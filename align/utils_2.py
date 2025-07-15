import sys
sys.path.append('/mnt/xjtu/GS/code/gaussian-splatting/')
from scene.cameras import Camera
from scene import GaussianModel
import numpy as np
import torch
from scene_target import RobotUtil 
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from e3nn import o3
import einops
from torch import einsum
import argparse

def transform_shs(shs_feat, rotation_matrix):
    ## rotate shs
    P = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).cuda() # switch axes: yzx -> xyz
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
    
    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

    #rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
        "... i j, ... j -> ... i",
        D_1,
        one_degree_shs,  
    )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
        "... i j, ... j -> ... i",
        D_2,
        two_degree_shs,
    )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
        "... i j, ... j -> ... i",
        D_3,
        three_degree_shs,
    )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat

def get_sapien_render(cameras_info):

    """
    get images rendered from cameras in sapien simulation.
    """
    robot_util = RobotUtil(cameras_info, control_hz=22, timestep=1/180)
    robot_util.robot.set_qpos(robot_util.init_qpos)
    robot_util.scene.update_render()

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

def set_scene_camera(camera_list, cameras_info):

    """
    set camera parameters in reality
    """
    for i in range(len(cameras_info)):
        camera_list.append(Camera(cameras_info[i]))
    return camera_list

def initialize_gs_model(dataset, opt, gs_arm_path):

    """
    initialize gaussian splatting model
    """
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians.load_ply(gs_arm_path)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    return gaussians, background

# def rotate_by_matrix(gs, rotation_matrix, keep_sh_degree: bool = True):
#     # rotate xyz
#     gs._xyz = torch.matmul(gs._xyz, rotation_matrix.T)

#     # rotate gaussian
#     # rotate via quaternions
#     def quat_multiply(quaternion0, quaternion1):
#         w0, x0, y0, z0 = torch.split(quaternion0, [1,1,1,1], dim=-1)
#         w1, x1, y1, z1 = torch.split(quaternion1, [1,1,1,1], dim=-1)
#         return torch.cat((
#             -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
#             x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
#             -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
#             x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
#         ), dim=-1)

#     quaternions = matrix_to_quaternion(rotation_matrix)[None, ...]
#     rotations_from_quats = quat_multiply(gs._rotation, quaternions)
#     rotations = rotations_from_quats / torch.linalg.norm(rotations_from_quats, dim=-1, keepdims=True)
#     gs._rotation = rotations

def transform_gs(gaussian_points, quat, tran, scale):

    """
    transform gaussian points according to optimized quaternion, translation and scale
    """
    torch_rot = quaternion_to_matrix(quat)
    transform44 = torch.eye(4, 4).cuda()
    transform44[:3, :3] = torch_rot
    transform44[:3, 3] = tran

    gaussian_points._xyz = gaussian_points.get_xyz * scale
    gaussian_points._scaling = gaussian_points._scaling + torch.log((scale.clone()))
    # rotate_by_matrix(gaussian_points, transform44[:3, :3])
    # gaussian_points._xyz += tran
    xyz = gaussian_points.get_xyz
    tran = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=1)
    tran = torch.matmul(tran, transform44.T)[:, :3]
    rot = gaussian_points.get_rotation
    rotation = quaternion_to_matrix(rot)
    rotation = torch.matmul(torch_rot, rotation)
    rot = matrix_to_quaternion(rotation)
    
    gaussian_points._xyz = tran
    gaussian_points._rotation = rot 
    
    # shs = gaussian_points.get_features_rest
    # gaussian_points._features_rest = transform_shs(shs, torch_rot[0, :3, :3])
    return gaussian_points


def main(args):
    matrix = torch.load(args.ckpt_path)
    tran, quant, scale = matrix[0, :3], matrix[0, 3:7], matrix[0, 7]
    print(quant)
    quant = quant[[3,0,1,2]]
    gaussians = GaussianModel(3, 'default')
    gaussians.load_ply(args.gs_path)

    gaussian_points = transform_gs(gaussians, quant, tran, scale)
    gaussian_points.save_ply(args.save_path)
    print(f"Completed! Saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="./output/network_align/ckpts/ckpt_1000.pth")
    parser.add_argument("--gs_path", type=str, default="./point_cloud/pcd_fps5_high_re/desk_m.ply")
    parser.add_argument("--save_path", type=str, default="./point_cloud/pcd_fps5_high_re/desk_m_posed.ply")
    args = parser.parse_args()

    main(args)
