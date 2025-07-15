import sys, json
import open3d as o3d
import numpy as np
from scene_target import RobotUtil 

with open('/mnt/xjtu/GS/code/gaussian-splatting/camera.json', 'r') as f:
    camera_info = json.load(f)
robot_util = RobotUtil(camera_info, control_hz=22, timestep=1/180)
robot_util.init_qpos = np.array([0.16333782,  0.60923889,  0.04159661, -1.5864757 , -0.080347  ,  2.24162222,  0.05934149,  0.03437125,  0.025]) # grasp pose (qpose)
robot_util.robot.set_qpos(robot_util.init_qpos)
print(robot_util.get_ee_pose())
robot_util.scene.update_render()

# Test getting point cloud
pc = robot_util.get_pc(num_point=100000)

# visualize the point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
# print(pc[:, 3:6])
# pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6])  # 颜色值需要归一化到 [0, 1]
o3d.visualization.draw_geometries([pcd])

# ----------------------------------
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append('/mnt/xjtu/GS/code/gaussian-splatting/')
from utils.graphics_utils import BasicPointCloud
# TODO: 注释 GaussianModel: line 173
from scene import GaussianModel

simulator_gauss = GaussianModel(3, 'default')
a = BasicPointCloud(pc[:,:3], pc[:,3:], 0.01)
simulator_gauss.create_from_pcd(a, camera_info, 0.01)
# simulator_gauss._xyz = torch.from_numpy(pc[:, :3]).cuda()
simulator_gauss.save_ply('/mnt/xjtu/GS/code/gaussian-splatting/point_cloud/simulator_gaussian/simulator_gauss_2.ply')