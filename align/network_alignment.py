import os
import torch
from random import randint
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene.cameras import Camera
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
import json
import pytorch3d.transforms as p3d_transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.colmap_loader import torch_rotmat2qvec, torch_qvec2rotmat
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from scene_target import RobotUtil 
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def get_aspin_render(robot_util):
    render_list = []
    for camera in robot_util.cameras:
        camera.take_picture()
        # segment robot
        seg_labels = camera.get_picture('Segmentation')  # [H, W, 4]
        label_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
        label_image[label_image>0] = 1
        render_list.append(label_image)
    render_list = np.stack(render_list, axis=0) # [N, H, W]
    return render_list

class CameraPoseOptimizer(nn.Module):
    def __init__(self, gaussians, pipe, bg, transformed_matrix, scale, use_trained_exp=True, separate_sh=True):
        super().__init__()
        pose_params = np.loadtxt(transformed_matrix, delimiter=',')
        initial_alignment_vector = np.concatenate([pose_params, [scale]])
        initial_alignment_tensor = torch.tensor(initial_alignment_vector, dtype=torch.float32).unsqueeze(0)
        self.transformed_matrix = nn.Parameter(initial_alignment_tensor, requires_grad=True)
        self.gaussian_model = gaussians
        self.pipe = pipe
        self.bg = bg
        self.use_trained_exp = use_trained_exp
        self.separate_sh = separate_sh
        
        self.set_frozen_parameters()

    def set_frozen_parameters(self):
        self.gaussian_model._xyz.requires_grad = False
        self.gaussian_model._features_dc.requires_grad = False
        self.gaussian_model._features_rest.requires_grad = False
        self.gaussian_model._scaling.requires_grad = False
        self.gaussian_model._opacity.requires_grad = False
        self.gaussian_model._rotation.requires_grad = False 
        self.gaussian_model.xyz_gradient_accum.requires_grad = False
        self.gaussian_model.denom.requires_grad = False
    
    def transform_camera(self):
        quaternoin = self.transformed_matrix[:, 3:7] # 1 * 4 (x,y,z,w)
        quaternoin = quaternoin[:, [3, 0, 1, 2]] # convert to wxyz
        pos = self.transformed_matrix[0, :3] # 1 * 3
        scale = self.transformed_matrix[0, 7:] # 1 * 1
        scale = torch.eye(3) * scale # 3 * 3

        quaternoin = quaternoin / torch.norm(quaternoin, dim=-1, keepdim=True)
        rotmat = p3d_transforms.quaternion_to_matrix(quaternoin).squeeze() # wxyz 1 * 3 * 3
        transform44 = torch.eye(4, device=quaternoin.device)# 4 * 4
        transform44[:3, :3] = rotmat
        transform44[:3, 3] = pos
        # no_name = torch.matmul(scale, transform44[:3, :].clone())
        # transform44[:3, :] = no_name 
        return transform44.to("cuda")

    def forward(self, viewpoint_cam):

        transform44 = self.transform_camera()
        gaussian_render = render(viewpoint_cam, self.gaussian_model, self.pipe, self.bg, 
                                 use_trained_exp=self.use_trained_exp, separate_sh=self.separate_sh, 
                                 transformed_matrix=transform44, scales_ratio=self.transformed_matrix[:, 7:])
        
        """
        visualization code for the gaussian model
        import open3d as o3d
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(self.gaussian_model._xyz.detach().cpu().numpy())
        o3d.visualization.draw_geometries([target])
        """
        return gaussian_render

def set_scene_camera(camera_list, cameras_info):
    for i in range(len(cameras_info)):
        camera_list.append(Camera(cameras_info[i]))
    return camera_list

def training(dataset, opt, pipe, args, cameras_info=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = SummaryWriter('./output/network_align/loss')

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # scene = Scene(dataset, gaussians)
    gaussians.load_ply(args.gs_path)
    
    # gaussians.training_setup(opt)
    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    # use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    # depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # initialize camera
    camera_list = []
    # viewpoint_stack = scene.getTrainCameras().copy()
    # read camera_file to load cameras_info
    
    viewpoint_stacks = set_scene_camera(camera_list, cameras_info)

    robot_util = RobotUtil(cameras_info, control_hz=22, timestep=1/180)
    robot_util.robot.set_qpos(robot_util.init_qpos)
    robot_util.scene.update_render()
    gt_render = get_aspin_render(robot_util)
    matrix_optim = CameraPoseOptimizer(gaussians, pipe, background, args.transformed_matrix, args.scale,
                                       use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)

    optimizer = torch.optim.Adam(matrix_optim.parameters(), lr=0.0001, eps=1e-15)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    for iteration in range(first_iter, opt.iterations + 1):
        
        iter_start.record()
        torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
        images = []
        for i in range(len(viewpoint_stacks)):

            viewpoint_cam = viewpoint_stacks[i]

            render_pkg = matrix_optim(viewpoint_cam)
            # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            image = image.mean(dim=0)
            render_img = image.contiguous().detach().cpu().numpy()  # [H, W, C]
            images.append(image)
            # Loss
            # gt_image = viewpoint_cam.original_image.cuda()
        images = torch.stack(images, dim=0)
        loss = l1_loss(images, torch.from_numpy(gt_render).cuda())
        loss.backward()

        iter_end.record()
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)
        with torch.no_grad():
            # Progress bar
            loss_for_log = loss.item()
            training_report(tb_writer, iteration, loss_for_log, iter_start.elapsed_time(iter_end))
            if iteration % 1 == 0:
                progress_bar.set_postfix({"Loss": f"{loss_for_log:.{7}f}"})
                progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % 1000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                
                ckpt_path = os.path.join(args.save_path, f"ckpt_{iteration}.pth")
                torch.save(matrix_optim.transformed_matrix, ckpt_path)

def training_report(tb_writer, iteration, loss_for_log, elapsed):
    tb_writer.add_scalar("Loss", loss_for_log, iteration)
    tb_writer.add_scalar('iter_time', elapsed, iteration)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__== "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ptvsd", action="store_true", help="Whether to start ptvsd debugging.")
    parser.add_argument('--ip', type=str, default="127.0.0.0")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gs_path", type=str, help="Path to the Gaussian Splatting point cloud file you want to align.")
    parser.add_argument("--transformed_matrix", type=str, default = "transformation_icp.txt")
    parser.add_argument("--scale", type=float, default=0.32, help="Used to normalize the source_pcd")
    parser.add_argument("--save_path", type=str, default="./output/network_align/ckpts", help="Path to save the checkpoints.")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    torch.set_default_device("cuda:0")
    torch.cuda.set_device(0)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    with open('/mnt/xjtu/GS/code/gaussian-splatting/camera.json', 'r') as f:
        camera_info = json.load(f)

    # Start GUI server, configure and run training
    # if not args.disable_viewer:
    #     network_gui.init(args.ip, args.port)
    if args.ptvsd:    
        import ptvsd
        ptvsd.enable_attach(address =('127.0.0.1', 10010), redirect_output=True)
        ptvsd.wait_for_attach()
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args, camera_info)
