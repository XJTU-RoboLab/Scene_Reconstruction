from typing import List
import argparse
## TODO clean up the code here, too many functions that are plurals of one or the other and confusing naming
import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import trimesh, json
import trimesh.creation
import sys, torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
sys.path.append('/mnt/xjtu/GS')
from scene_target import RobotUtil
sys.path.append('/mnt/xjtu/GS/code/gaussian-splatting')
from scene import GaussianModel
from scipy.spatial.transform import Rotation as R

def transform_gaussian(gaussian, T, scale=1.0, transform_sh=False):
    # transform xyz
    xyz = gaussian.get_xyz
    xyz = xyz * scale
    xyz = xyz @ T[:3, :3].T + T[:3, 3]
    gaussian._xyz = xyz
    # tranform rotation
    rotation = gaussian.get_rotation
    rotation = quaternion_to_matrix(rotation)
    rotation = T[:3, :3] @ rotation
    rotation = matrix_to_quaternion(rotation)
    gaussian._rotation = rotation
    # transform scale
    gaussian._scaling = torch.log(torch.exp(gaussian._scaling) * scale)

    if transform_sh:
        # transform sh
        sh = gaussian._features_rest
        sh = transform_shs(sh, T[:3, :3])
        gaussian._features_rest = sh

    return gaussian

def get_component_meshes(component: physx.PhysxRigidBaseComponent):
    """Get component (collision) meshes in the component's frame."""
    meshes = []
    for geom in component.get_collision_shapes():
        if isinstance(geom, physx.PhysxCollisionShapeBox):
            mesh = trimesh.creation.box(extents=2 * geom.half_size)
        elif isinstance(geom, physx.PhysxCollisionShapeCapsule):
            mesh = trimesh.creation.capsule(
                height=2 * geom.half_length, radius=geom.radius
            )

        elif isinstance(geom, physx.PhysxCollisionShapeCylinder):
            mesh = trimesh.creation.cylinder(
                radius=geom.radius, height=2 * geom.half_length
            )
        elif isinstance(geom, physx.PhysxCollisionShapeSphere):
            mesh = trimesh.creation.icosphere(radius=geom.radius)
        elif isinstance(geom, physx.PhysxCollisionShapePlane):
            continue
        elif isinstance(geom, (physx.PhysxCollisionShapeConvexMesh)):
            vertices = geom.vertices  # [n, 3]
            faces = geom.get_triangles()
            vertices = vertices * geom.scale
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        elif isinstance(geom, physx.PhysxCollisionShapeTriangleMesh):
            vertices = geom.vertices
            faces = geom.get_triangles()
            vertices = vertices * geom.scale
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            raise TypeError(type(geom))
        mesh.apply_transform(geom.get_local_pose().to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def get_render_body_meshes(visual_body: sapien.render.RenderBodyComponent):
    meshes = []
    for render_shape in visual_body.render_shapes:
        meshes += get_render_shape_meshes(render_shape)
    return meshes


def get_render_shape_meshes(render_shape: sapien.render.RenderShape):
    meshes = []
    if type(render_shape) == sapien.render.RenderShapeTriangleMesh:
        for part in render_shape.parts:
            vertices = part.vertices * render_shape.scale  # [n, 3]
            faces = part.triangles
            # faces = render_shape.mesh.indices.reshape(-1, 3)  # [m * 3]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.apply_transform(render_shape.local_pose.to_transformation_matrix())
            meshes.append(mesh)
    return meshes


def get_actor_visual_meshes(actor: sapien.Entity):
    """Get actor (visual) meshes in the actor frame."""
    meshes = []
    comp = actor.find_component_by_type(sapien.render.RenderBodyComponent)
    if comp is not None:
        meshes.extend(get_render_body_meshes(comp))
    return meshes


def merge_meshes(meshes: List[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None


def get_component_mesh(component: physx.PhysxRigidBaseComponent, to_world_frame=True):
    mesh = merge_meshes(get_component_meshes(component))
    if mesh is None:
        return None
    if to_world_frame:
        T = component.pose.to_transformation_matrix()
        mesh.apply_transform(T)
    return mesh


def get_actor_visual_mesh(actor: sapien.Entity):
    mesh = merge_meshes(get_actor_visual_meshes(actor))
    if mesh is None:
        return None
    return mesh


def get_articulation_meshes(
    articulation: physx.PhysxArticulation, exclude_link_names=()
):
    """Get link meshes in the world frame."""
    meshes = []
    names = []
    m = {}
    link_names = ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand', 'leftfinger', 'rightfinger']
    for link in articulation.get_links():
        if link.name in exclude_link_names:
            continue
        mesh = get_component_mesh(link, True)
        if mesh is None:
            continue
        if 'hand' in link.name:
            m['link7'] = mesh if 'link7' not in m else merge_meshes([m['link7'], mesh])
            continue

        for link_name in link_names: 
            if link_name in link.name:
                m[link_name] = mesh if link_name not in m else merge_meshes([m[link_name], mesh])

    meshes.append(mesh)
    names.append(link.name)
    return m.values(), m.keys()


def main(agrs):
    with open("./camera.json", "r") as f:
        cameras_info = json.load(f)
    sc = ('fr3_link0_sc', 'fr3_link1_sc', 'fr3_link2_sc', 'fr3_link3_sc', 'fr3_link4_sc', 'fr3_link5_sc', 'fr3_link6_sc', 'fr3_link7_sc', 'fr3_hand_sc')
    robot_util = RobotUtil(cameras_info, control_hz=22, timestep=1/180)
    robot_util.robot.set_qpos(robot_util.init_qpos)
    robot_util.scene.update_render()
    # pc = robot_util.get_pc(num_point=10000)
    T = []
    for i in range(13):
        if 'hand' in robot_util.robot.get_links()[i].name or 'link8' in robot_util.robot.get_links()[i].name:
            continue
        a = robot_util.robot.get_links()[i].entity_pose
        pos, rot = a.p, a.q
        rot3_3 = R.from_quat(rot, scalar_first=True).as_matrix()
        trans = np.eye(4)
        trans[:3,:3] = rot3_3
        trans[:3, 3] = pos
        inv_trans = np.linalg.inv(trans)
        T.append(inv_trans)
    T = np.stack(T, axis=0)

    meshes, names = get_articulation_meshes(robot_util.robot, sc)

    gaussians = GaussianModel(3, 'default')
    gaussians.load_ply(args.ply_path) # points to check whether it is in the meshes
    i = 0
    link = {}
    points = gaussians.get_xyz.detach().cpu().numpy()
    flag = np.zeros(points.shape[0], dtype=bool)
    gaussian_component = GaussianModel(3, 'default')
    for mesh, name in zip(meshes, names): 
        if 'sc' in name:
            continue
        con = mesh.contains(points)
        gaussian_component._xyz = gaussians._xyz[con]
        gaussian_component._features_dc = gaussians._features_dc[con]
        gaussian_component._features_rest = gaussians._features_rest[con]
        gaussian_component._scaling = gaussians._scaling[con]
        gaussian_component._rotation = gaussians._rotation[con]
        gaussian_component._opacity = gaussians._opacity[con]
        flag[con] = True
        gaussian_component = transform_gaussian(gaussian_component, torch.from_numpy(T[i]).float().to('cuda:0'))
        gaussian_component.save_ply(f'/mnt/xjtu/GS/code/RoboSplat/data_hw/gaussian/robot/{name}_default.ply')
        i += 1
    # gaussian_background = GaussianModel(3, 'default')
    # gaussian_background._xyz = gaussians.get_xyz[con]
    # gaussian_component._features_dc = gaussians.get_features_dc[con]
    # gaussian_component._features_rest = gaussians.get_features_rest[con]
    # gaussian_component._scaling = gaussians.get_scaling[con]
    # gaussian_component._rotation = gaussians.get_rotation[con]
    # gaussian_component._opacity = gaussians.get_opacity[con]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_path", type=str, default="/mnt/xjtu/GS/code/gaussian-splatting/point_cloud/pcd_fps5_high_re/fr3.ply")
    args = parser.parse_args()

    main(args)
