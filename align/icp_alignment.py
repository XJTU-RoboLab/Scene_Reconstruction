import sys, json
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from scene_target import RobotUtil
import argparse


def get_urdf_and_gs_pcd(camera_path, gs_path, scale):
    """
    args:
        scale: scale gs points to align with urdf

    return: source, target
    """
    with open(camera_path, 'r') as f:
        camera_info = json.load(f)
    robot_util = RobotUtil(camera_info, control_hz=22, timestep=1/180)
    robot_util.robot.set_qpos(robot_util.init_qpos)
    print(robot_util.get_ee_pose())
    robot_util.scene.update_render()

    pc = robot_util.get_pc(num_point=100000)

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    # o3d.visualization.draw_geometries([tgt_pcd])

    source = o3d.io.read_point_cloud(gs_path)
    source.points = o3d.utility.Vector3dVector(np.asarray(source.points)*scale)
    # o3d.visualization.draw_geometries([source, pcd])
    return source, tgt_pcd

def check_align(source, reg_p2p, pcd):
    """
    args:
        source: Gaussian points before alignment
        reg_p2p: result of ICP algrothm
        pcd: urdf Gaussian points

    """
    points = np.asarray(source.points)
    transformation = reg_p2p.transformation

    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1) # convert to homogeneous coordinates (N, 3) -> (N, 4)
    transformed_points = np.dot(points, transformation.T)[:, :3]

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_pcd.colors = o3d.utility.Vector3dVector(np.asarray(source.colors))
    o3d.visualization.draw_geometries([transformed_pcd, pcd])

def ICP(source, pcd, threshold, trans_init):
    """
    args:
        source: Gaussian points before alignment
        pcd: urdf Gaussian points
        threshold: distance threshold
        trans_init: initial transformation

    return: result of ICP algrothm
    """
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return reg_p2p

def matrix_to_quat(reg_p2p):
    """
    args:
        reg_p2p: The result of ICP algorithm
    
    return:
        transformation: xyz + quaternion
    """
    transformation = np.zeros((7, ))
    transformation[3:] = R.from_matrix(reg_p2p.transformation[:3, :3]).as_quat() # (x,y,z,w)
    transformation[:3] = reg_p2p.transformation[:3, 3]
    return transformation # (xyz + xyzw)

def main(args):
    trans_init = np.loadtxt(args.trans_init_path, delimiter=',')
    source, pcd = get_urdf_and_gs_pcd(args.camera_path, args.gs_path, args.scale)
    reg_p2p = ICP(source, pcd, args.threshold, trans_init)
    check_align(source, reg_p2p, pcd)
    transformation = matrix_to_quat(reg_p2p)   
    np.savetxt("transformation_icp.txt", transformation.reshape(1, -1), delimiter=",", fmt="%.8f") 
    print(transformation)


if __name__ == "__main__":
    gs_path = "/mnt/xjtu/GS/code/gaussian-splatting/point_cloud/pcd_fps5_high_re/fr3.ply"

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_path", type=str, default="/mnt/xjtu/GS/code/gaussian-splatting/camera.json")
    parser.add_argument("--gs_path", type=str, default=gs_path, help="Path to the Gaussian Splatting point cloud file")
    parser.add_argument("--threshold", type=float, default=10, help="Distance threshold for ICP")
    parser.add_argument("--trans_init_path", type=str, default="transformation_initial.txt", help="Initial transformation matrix file")
    parser.add_argument("--scale", type=float, default=0.32, help="Used to normalize the source_pcd")
    args = parser.parse_args()

    main(args)
    
