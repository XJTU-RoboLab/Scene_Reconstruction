import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/mnt/xjtu/GS/code/gaussian-splatting/gaussian_renderer")
from gaussian_renderer import render
import pytorch3d.transforms as p3d_transforms

class MatrixOptimizer(nn.Module):
    def __init__(self, gaussians, pipe, bg, use_trained_exp=True, separate_sh=True):
        super().__init__()

        """
        args:
            gaussians: GS points of robot arm
            pipe: render pipeline configuration
            bg: background
            use_trained_exp: use trained experiment
            separate_sh: separate sh
        """
        # [ 0.12127015  0.83774023 -0.50975773 -0.15372673] [0.22867865 0.22098053 0.67912668]
        # transformed_matrix is the result from icp_alignment.py included wxyz, xyz and scale(sequentially).
        self.transformed_matrix = nn.Parameter(
            # torch.tensor([[-0.1545,  0.1231,  0.8375, -0.5098,  0.2282,  0.2153,  0.6728,  0.2278]]), 
            torch.tensor([[-0.31787468, 0.38655424,  0.64750253, -0.57469294,   0.45318359, -0.2272101 ,  0.74428139, 0.23]]),
            requires_grad=True)
        self.gaussian_model = gaussians
        self.pipe = pipe
        self.bg = bg
        self.use_trained_exp = use_trained_exp
        self.separate_sh = separate_sh
        
        self.set_frozen_parameters()

    def set_frozen_parameters(self):

        """
        frozen gs model parameters
        """
        self.gaussian_model._xyz.requires_grad = False
        self.gaussian_model._features_dc.requires_grad = False
        self.gaussian_model._features_rest.requires_grad = False
        self.gaussian_model._scaling.requires_grad = False
        self.gaussian_model._opacity.requires_grad = False
        self.gaussian_model._rotation.requires_grad = False 
        self.gaussian_model.xyz_gradient_accum.requires_grad = False
        self.gaussian_model.denom.requires_grad = False
    
    def get_transform(self):

        """
        get initial transformation matrix
        """
        quaternoin = self.transformed_matrix[:, :4] # 1 * 4
        pos = self.transformed_matrix[0, 4:7] # 1 * 3

        quaternoin = quaternoin / torch.norm(quaternoin, dim=-1, keepdim=True)
        rotmat = p3d_transforms.quaternion_to_matrix(quaternoin).squeeze() # wxyz 1 * 3 * 3
        transform44 = torch.eye(4, device=quaternoin.device)# 4 * 4
        transform44[:3, :3] = rotmat
        transform44[:3, 3] = pos
        return transform44.to("cuda")

    def forward(self, viewpoint_cam):

        """
        args:
            viewpoint_cam: cameras from multi viewpoints

        return:
            images: render images from multi cameras
        """
        transform44 = self.get_transform()
        # transform44 = None
        gaussian_render = render(viewpoint_cam, self.gaussian_model, self.pipe, self.bg, 
                                 use_trained_exp=self.use_trained_exp, separate_sh=self.separate_sh, 
                                 transformed_matrix=transform44, scales_ratio=self.transformed_matrix[0, 7:])
        
        return gaussian_render