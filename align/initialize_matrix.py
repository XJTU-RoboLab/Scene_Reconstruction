import copy
import open3d as o3d
import numpy as np
from icp_alignment import get_urdf_and_gs_pcd
import argparse


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, camera_path, gs_path, scale):
    print(":: Load two point clouds and disturb initial pose.")

    source, target = get_urdf_and_gs_pcd(camera_path, gs_path, scale)
    # source = o3d.io.read_point_cloud("/mnt/xjtu/GS/code/gaussian-splatting/point_cloud/new_gripper_arm.ply")
    # source.points = o3d.utility.Vector3dVector(np.array(source.points)*0.23)
    # target = pcd

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def main(args):
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        args.voxel_size, args.camera_path, args.gs_path, args.scale)
    best_result =  None
    best_fitness = 0.0
    for i in range(1):
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    args.voxel_size)
        if result_ransac.fitness > best_fitness:
            best_result = result_ransac
            best_fitness = result_ransac.fitness
            
        print("fitness:", best_result.fitness)

    print("result_ransac.transformation:", best_result.transformation)
    draw_registration_result(source_down, target_down, result_ransac.transformation)
    np.savetxt("transformation_initial.txt", best_result.transformation, delimiter=", ", fmt="%.8f")
    

if __name__ == "__main__":
    gs_path = "/mnt/xjtu/GS/code/gaussian-splatting/point_cloud/pcd_fps5_high_re/fr3.ply"
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=0.32, help="Scale factor for the point cloud")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for downsampling") # 0.01
    parser.add_argument("--camera_path", type=str, default="/mnt/xjtu/GS/code/gaussian-splatting/align_ws/camera.json", help="Path to the camera JSON file")
    parser.add_argument("--gs_path", type=str, default=gs_path, help="Path to the Gaussian Splatting point cloud file")
    args = parser.parse_args()

    main(args)
