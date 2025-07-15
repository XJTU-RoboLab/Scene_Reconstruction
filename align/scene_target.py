import os
import mplib.sapien_utils
import sapien
import numpy as np
import torch
import pytorch3d.ops as pytorch3d_ops
import mplib,sys
import json
from scipy.spatial.transform import Rotation as R
from mplib import Pose
import roboticstoolbox as rtb
from roboticstoolbox.robot.ERobot import ERobot

# Define the initial joint positions for the Franka Research3 robot
INIT_QPOS = [-0.030938, 0.185544, -0.0125273, -1.9622, -0.0490948, 2.2383, 0.788477, 0.04, 0.04]

def get_RT(camera_info):

    cam_pos = np.array(camera_info["cam_pos"])
    look_at_point = np.array(camera_info["look_at_point"])
    cam_rel_pos = cam_pos - look_at_point
    forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    
    return forward, left, up

class FrankaEmikaPanda(ERobot):
    '''
    Class that imports a FrankaEmikaPanda Robot
    '''
    def __init__(self):
        origin_dir = os.getcwd()
        links, name, urdf_string, urdf_filepath = super().URDF_read(
            rtb.rtb_path_to_datafile("/mnt/xjtu/GS/code/RoboSplat/data_our/asset/robot/fr3/gazebo_fr3.urdf"), tld=FrankaEmikaPanda.load_my_path())
        os.chdir(origin_dir)
        super().__init__(
            links,
            name=name,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath)
        self.default_joint_pos = np.array(INIT_QPOS)
        self.addconfiguration('qr', self.default_joint_pos)

    @staticmethod
    def load_my_path():
        os.chdir(os.path.dirname(__file__))

class RobotUtil:
    def __init__(self, cameras_info=None, control_hz=22, timestep=1/180):
        self.control_hz = control_hz
        frame_skip = int(1/timestep/control_hz)
        self.frame_skip = frame_skip
        self.setup_scene(cameras_info=cameras_info, timestep=timestep)
        self.load_robot()
        self.setup_planner()
        self.scene.update_render()

    def setup_scene(self, cameras_info, timestep, ray_tracing=True):
        self.scene = sapien.Scene()
        self.scene.set_timestep(timestep)
        self.scene.default_physical_material = self.scene.create_physical_material(1, 1, 0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        self.setup_camera(cameras_info)
        if ray_tracing:
            sapien.render.set_camera_shader_dir('rt')
            sapien.render.set_viewer_shader_dir('rt')
            sapien.render.set_ray_tracing_samples_per_pixel(4)  # change to 256 for less noise
            sapien.render.set_ray_tracing_denoiser('oidn') # change to 'optix' or 'oidn'

    def load_robot(self, urdf_path="/mnt/xjtu/GS/code/RoboSplat/data_our/asset/robot/fr3/gazebo_fr3.urdf"):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(urdf_path)
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=100, force_limit=100, mode='force')
            joint.set_friction(0.0)
        self.init_qpos = INIT_QPOS
        self.robot.set_qpos(self.init_qpos)
        self.end_effector = self.robot.get_links()[10]
        for link in self.robot.links:
            link.disable_gravity = True

    def setup_planner(self, urdf_path="/mnt/xjtu/GS/code/RoboSplat/data_our/asset/robot/fr3/gazebo_fr3.urdf", srdf_path="/mnt/xjtu/GS/code/RoboSplat/data_our/asset/robot/fr3/fr3.srdf", move_group='fr3_hand_tcp'):
        self.planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group=move_group)
        self.plannerworld = mplib.sapien_utils.SapienPlanningWorld(self.scene)
        # self.planner = mplib.sapien_utils.SapienPlanner(self.plannerworld, move_group=move_group)

    def setup_camera(self, cameras_info=None):
        self.cameras = []

        for i in range(len(cameras_info)):
            forward, left, up = get_RT(cameras_info[i])
            mat44 = np.eye(4)
            mat44[:3, :3] = np.stack([forward, left, up], axis=1)
            mat44[:3, 3] = cameras_info[i]["cam_pos"]
            camera_0 = self.scene.add_camera(
                name=cameras_info[i]["image_name"],
                width=cameras_info[i]["width"],
                height=cameras_info[i]["height"],
                fovy=np.deg2rad(cameras_info[i]["fov"]),
                near=cameras_info[i]["near"],
                far=cameras_info[i]["far"],
            )
            camera_0.entity.set_pose(sapien.Pose(mat44))
            self.cameras.append(camera_0)

    def get_ee_pose(self):
        ee_pose = self.end_effector.get_pose()
        return np.concatenate([ee_pose.p, ee_pose.q])

    def get_qpos(self):
        return self.robot.get_qpos()

    def get_pc(self, num_point=None):
        pc_list = []
        for camera in self.cameras:
            camera.take_picture()
            rgb = camera.get_picture('Color')[:,:,:3]
            position = camera.get_picture('Position')

            # segment robot
            seg_labels = camera.get_picture('Segmentation')  # [H, W, 4]
            label_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
            points_opengl = position[..., :3][(position[..., 3] < 1) & (label_image > 0)]  # position[..., :3][(position[..., 3] < 1)]
            points_color = rgb[(position[..., 3] < 1) & (label_image > 0)]

            model_matrix = camera.get_model_matrix()
            points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
            points_color = np.clip(points_color, 0, 1)  # (np.clip(points_color, 0, 1) * 255).astype(np.uint8)
            pc_list.append(np.concatenate([points_world, points_color], axis=-1))

        pc = np.concatenate(pc_list, axis=0)
        
        if num_point is not None:
            _, fps_idx = pytorch3d_ops.sample_farthest_points(points=torch.from_numpy(pc[:,:3]).cuda().unsqueeze(0), K=num_point)
            pc = pc[fps_idx[0].cpu().numpy()]

        return pc

    def get_pc_at_qpos(self, qpos, num_point=None):
        self.robot.set_qpos(qpos)
        self.scene.update_render()
        pc = self.get_pc(num_point)

        return pc

    def step(self, action):
        # position control
        for i in range(7):
            self.active_joints[i].set_drive_target(action[i])
        if action[-1] > 0.5:
            self.active_joints[-1].set_drive_target(0.00)
            self.active_joints[-2].set_drive_target(0.00)
        elif action[-1] <= 0.5:
            self.active_joints[-1].set_drive_target(0.04)
            self.active_joints[-2].set_drive_target(0.04)

        for _ in range(self.frame_skip):
            self.scene.step()
        self.scene.update_render()

    def get_trajectory(self, init_qpos, poses, gripper_action, control_hz=None):
        self.robot.set_qpos(init_qpos)
        ee_pose_list, qpos_list, action_list, n_step_list = [], [], [], [] # ee_pose and qpos are obs, i.e., they are the states before taking action

        for pose_idx, target_pose in enumerate(poses):
            result = self.planner.plan_pose(
                target_pose,
                self.get_qpos(),
                time_step=1/self.control_hz if control_hz is None else 1/control_hz[pose_idx],
            )
            result_pos = result['position']
            # combine the small action
            # delta_action = abs(result_pos[1:] - result_pos[:-1])
            # delta_action.sum(axis=-1)
            result_pos[-3] = result_pos[-1]
            result_pos = result_pos[2:-2]

            n_step = result_pos.shape[0]
            action = np.zeros([8,])
            for i in range(n_step):
                qpos_list.append(self.get_qpos())
                if gripper_action[pose_idx] == 0:
                    action = np.zeros([8,])
                elif gripper_action[pose_idx] == 1:
                    action = np.ones([8,])
                action[:7] = result_pos[i]
                action_list.append(action)
                self.step(action)
                ee_pose_list.append(self.get_ee_pose())
            n_step_list.append(n_step)

        return ee_pose_list, qpos_list, action_list, n_step_list
    
    def visualization(self):
        viewer = self.scene.create_viewer()

        # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
        # The principle axis of the camera is the x-axis
        viewer.set_camera_xyz(x=-4, y=0, z=2)
        # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
        # The camera now looks at the origin
        viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        while not viewer.closed:  # Press key q to quit
            # self.scene.step()  # Simulate the world
            self.scene.update_render()  # Update the world to the renderer
            viewer.render()

    def add_scene_object(self):
        self.scene.add_ground(altitude=0)  # Add a ground
        actor_builder = self.scene.create_actor_builder()
        actor_builder.add_box_collision(half_size=[0.025, 0.025, 0.025])
        actor_builder.add_box_visual(half_size=[0.025, 0.025, 0.025], material=[1., 0., 0.])
        box = actor_builder.build(name='box')  # Add a box
        box.set_pose(sapien.Pose(p=[0.542, -0.1113, 0.025]))

if __name__ == "__main__":
    camera_info = {}
    camera_list = []
    with open("/mnt/xjtu/GS/code/gaussian-splatting/camera.json", "r") as f:
        camera_list = json.load(f)

    robot_util = RobotUtil(camera_list, control_hz=22, timestep=1/180)
    # robot_util.robot.set_qpos(robot_util.init_qpos)
    # robot_util.scene.update_render()

    # Test getting point cloud
    # pc = robot_util.get_pc(num_point=10000)
    # print("Point cloud shape:", pc.shape)

    # Test getting end effector pose
    ee_pose = robot_util.get_ee_pose()
    # print("End effector pose:", ee_pose)

    # Test getting qpos
    qpos = robot_util.get_qpos()

    # robot_util.visualization()
    # import open3d as o3d
    # path = '/home/xjtu/Franka/gaussian/object/red_block_pcd.ply'
    # pcd = o3d.io.read_point_cloud(path)
    # points = pcd.points
    # robot_util.plannerworld.add_point_cloud(name="red_block", vertices=points)
    # robot_util.add_scene_object()

    # [ 0.58239448,  0.23053403,  0.03956759, -0.06326989,  0.73726702, 0.18528803, -0.64660854]
    init_pose = np.array([robot_util.init_qpos,[0.0446, 0.5960, -0.2111, -2.1772, 0.1554, 2.7961, 0.4752, 0.04, 0.04]])
    print(robot_util.init_qpos)
    end_pose = [Pose([0.58239448,  0.23053403,  0.03956759], [0.0337, 0.9992, -0.006, 0.0193]),
                Pose([0.5428, -0.1135, 0.0176], [0.0337, 0.9992, -0.006, 0.0193]),
                Pose([0.471, -0.028, 0.199], [0.015, 0.997, -0.072, 0.019]),]
    # trajectory = robot_util.planner.plan_pose(end_pose[0], init_pose[0], time_step=0.03)
    # print(trajectory['position'])
    ee_pose_list, qpos_list, action_list, n_step_list = robot_util.get_trajectory(init_pose[0], end_pose, [0,1])