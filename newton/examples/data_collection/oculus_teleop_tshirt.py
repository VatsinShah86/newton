###########################################################################
# Franka Oculus Teleop T-Shirt
#
# Franka Panda robot on a table, driven by Oculus Quest controller
# teleoperation via QuestStream. End-effector pose targets are accumulated
# from per-frame deltas and solved via IK; joint positions are tracked by
# Featherstone PD control. A cloth t-shirt is simulated with VBD.
#
# Simulation runs in centimetre scale for better VBD numerical behaviour.
# A viz_state converts back to metre scale for visualisation.
#
# Command: python newton/examples/data_collection/oculus_teleop_tshirt.py
#
###########################################################################

from __future__ import annotations

import atexit
import csv
import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton import ModelBuilder, eval_fk
from newton.sensors import SensorTiledCamera
from newton.solvers import SolverFeatherstone, SolverVBD
from newton.examples.data_collection.oculus_teleop import QuestStream

# Simulation ↔ world scale
SIM_TO_WORLD_SCALE = np.float32(0.01)   # cm → m
WORLD_TO_SIM_SCALE = np.float32(100.0)  # m  → cm

# Gripper (cm / cm·s⁻¹)
GRIPPER_SPEED = 8.0    # cm/s — full stroke in ~0.5 s
GRIPPER_MAX   = 4.0    # cm, fully open
GRIPPER_MIN   = 0.0    # cm, fully closed

# Camera (metre-space values used for visualisation only)
CAMERA_AXIS_LEN = 0.05   # m, length of each displayed frame axis
CAMERA_WIDTH    = 640
CAMERA_HEIGHT   = 480
CAMERA_FOV_DEG  = 60.0

# Cloth (centimetre scale)
CLOTH_PARTICLE_RADIUS       = 0.8     # cm
CLOTH_BODY_CONTACT_MARGIN   = 0.8    # cm
CLOTH_SELF_CONTACT_RADIUS   = 0.2    # cm
CLOTH_SELF_CONTACT_MARGIN   = 0.2    # cm
CLOTH_TRI_KE                = 1e4
CLOTH_TRI_KA                = 1e4
CLOTH_TRI_KD                = 1.5e-6
CLOTH_BENDING_KE            = 5.0
CLOTH_BENDING_KD            = 1e-2
CLOTH_SIZE_CM               = 20.0   # cm per side
CLOTH_CELLS                 = 20     # grid subdivisions per axis

# Gripper close limit when gripping cloth — don't crush below one particle radius
GRIPPER_CLOTH_MIN = 0 # CLOTH_PARTICLE_RADIUS

IK_ITERS = 24


@wp.kernel
def _compute_joint_qd(
    target_q: wp.array(dtype=float),
    current_q: wp.array(dtype=float),
    out_qd: wp.array(dtype=float),
    inv_frame_dt: float,
):
    """qd = (target - q) / frame_dt — kinematic tracking in one frame."""
    i = wp.tid()
    out_qd[i] = (target_q[i] - current_q[i]) * inv_frame_dt


@wp.kernel
def _update_camera_transform(
    body_q: wp.array(dtype=wp.transform),
    body_id: int,
    offset: wp.transform,
    out: wp.array(dtype=wp.transformf, ndim=2),
):
    """Write the camera world transform into the (camera, world) sensor array."""
    out[0, 0] = body_q[body_id] * offset


@wp.kernel
def _compute_frame_lines(
    body_q: wp.array(dtype=wp.transform),
    body_id: int,
    offset: wp.transform,
    axis_len: float,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
):
    """One thread per axis (0=X, 1=Y, 2=Z): start/end in world space."""
    i = wp.tid()
    cam_tf = body_q[body_id] * offset
    origin = wp.transform_get_translation(cam_tf)
    starts[i] = origin
    if i == 0:
        ends[i] = origin + wp.transform_vector(cam_tf, wp.vec3(axis_len, 0.0, 0.0))
    elif i == 1:
        ends[i] = origin + wp.transform_vector(cam_tf, wp.vec3(0.0, axis_len, 0.0))
    else:
        ends[i] = origin + wp.transform_vector(cam_tf, wp.vec3(0.0, 0.0, axis_len))


@wp.kernel
def _depth_to_point_cloud(
    depth: wp.array(dtype=wp.float32, ndim=4),     # (worlds, cameras, H, W)
    rays: wp.array(dtype=wp.vec3f, ndim=4),          # (cameras, H, W, 2)
    cam_tf: wp.array(dtype=wp.transformf, ndim=2),   # (cameras, worlds)
    out: wp.array(dtype=wp.vec3f, ndim=2),           # (H, W)
):
    y, x = wp.tid()
    d = depth[0, 0, y, x]
    if d <= 0.0:
        out[y, x] = wp.vec3f(0.0, 0.0, 0.0)
        return
    tf = cam_tf[0, 0]
    ray_dir_world = wp.transform_vector(tf, rays[0, y, x, 1])
    out[y, x] = wp.transform_get_translation(tf) + d * ray_dir_world


@wp.kernel
def _scale_positions(src: wp.array(dtype=wp.vec3), scale: float, dst: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def _scale_body_transforms(src: wp.array(dtype=wp.transform), scale: float, dst: wp.array(dtype=wp.transform)):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


class Example:
    def __init__(self, viewer, args=None, oculus_ip: str | None = None, debug_replay: str | None = None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viz_scale = float(SIM_TO_WORLD_SCALE)

        self.viewer = viewer

        # cm-scale scene (gravity in cm/s²)
        self.scene = ModelBuilder(gravity=-981.0)

        # Robot
        franka = ModelBuilder()
        self.create_articulation(franka)
        self.scene.add_world(franka)

        # Cloth t-shirt (cm scale) resting on the ground
        cloth_cell = CLOTH_SIZE_CM / CLOTH_CELLS
        cloth_origin_x = -0.5 * CLOTH_SIZE_CM
        cloth_origin_y = -50.0 - 0.5 * CLOTH_SIZE_CM
        cloth_top_z    = CLOTH_PARTICLE_RADIUS + 0.5
        self.scene.add_cloth_grid(
            pos=wp.vec3(cloth_origin_x, cloth_origin_y, cloth_top_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_CELLS,
            dim_y=CLOTH_CELLS,
            cell_x=cloth_cell,
            cell_y=cloth_cell,
            mass=0.02,
            tri_ke=CLOTH_TRI_KE,
            tri_ka=CLOTH_TRI_KA,
            tri_kd=CLOTH_TRI_KD,
            edge_ke=CLOTH_BENDING_KE,
            edge_kd=CLOTH_BENDING_KD,
            particle_radius=CLOTH_PARTICLE_RADIUS,
        )

        self.scene.color()
        self.scene.add_ground_plane()
        self.model = self.scene.finalize(requires_grad=False)

        # Contact materials
        self.model.soft_contact_ke  = 1e4
        self.model.soft_contact_kd  = 1e-2
        self.model.soft_contact_mu  = 0.25   # cloth self-contact friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[...] = 5e4
        shape_kd[...] = 1e-3
        shape_mu[...] = 1.5
        self.model.shape_material_ke = wp.array(shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.device)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.device)
        self.model.shape_material_mu = wp.array(shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.device)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Collision pipeline (cm margin)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=CLOTH_BODY_CONTACT_MARGIN,
        )
        self.contacts = self.collision_pipeline.contacts()

        # Featherstone solver for the robot
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)

        # VBD solver for cloth (full self-contact enabled)
        self.model.edge_rest_angle.zero_()
        self.cloth_solver = SolverVBD(
            self.model,
            iterations=5,
            integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=CLOTH_SELF_CONTACT_RADIUS,
            particle_self_contact_margin=CLOTH_SELF_CONTACT_MARGIN,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=self.model.soft_contact_ke,
        )

        # Gravity arrays for swapping: Featherstone runs gravity-free, VBD gets full gravity
        self.gravity_zero  = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.setup_ik()

        # CUDA graph capture of the simulate loop
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

        # Oculus / debug-replay state
        self.oculus = QuestStream(ip=oculus_ip) if oculus_ip is not None else None
        self._debug_replay_rows: list[dict] = []
        self._debug_replay_idx: int = 0
        self._debug_csv_file = None
        self._debug_csv_writer = None
        self._debug_frame: int = 0

        if debug_replay is not None:
            with open(debug_replay, newline="") as f:
                self._debug_replay_rows = list(csv.DictReader(f))
            n_dofs = self.model.joint_dof_count
            out_path = os.path.splitext(os.path.abspath(debug_replay))[0] + "_debug.csv"
            self._debug_csv_file = open(out_path, "w", newline="")
            fieldnames = (
                ["frame", "sim_time"]
                + ["input_delta_pos_x", "input_delta_pos_y", "input_delta_pos_z"]
                + ["input_delta_rot_x", "input_delta_rot_y", "input_delta_rot_z"]
                + ["input_gripper"]
                + ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
                + ["ee_quat_x", "ee_quat_y", "ee_quat_z", "ee_quat_w"]
                + ["ee_body_ang_vel_x", "ee_body_ang_vel_y", "ee_body_ang_vel_z"]
                + ["ee_body_lin_vel_x", "ee_body_lin_vel_y", "ee_body_lin_vel_z"]
                + ["gripper_pos"]
                + ["ee_target_pos_x", "ee_target_pos_y", "ee_target_pos_z"]
                + ["ee_target_quat_x", "ee_target_quat_y", "ee_target_quat_z", "ee_target_quat_w"]
                + [f"joint_q_{i}" for i in range(n_dofs)]
                + [f"joint_target_q_{i}" for i in range(n_dofs)]
            )
            self._debug_csv_writer = csv.DictWriter(self._debug_csv_file, fieldnames=fieldnames)
            self._debug_csv_writer.writeheader()
            atexit.register(self._close_debug_log)
            print(f"Debug replay: {len(self._debug_replay_rows)} frames → {out_path}")

        # Camera frame visualisation buffers
        self._cam_starts = wp.zeros(3, dtype=wp.vec3)
        self._cam_ends   = wp.zeros(3, dtype=wp.vec3)
        self._cam_colors = wp.array(
            [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)],
            dtype=wp.vec3,
        )
        self._cam_tf = wp.zeros((1, 1), dtype=wp.transformf)

        self.camera_sensor = SensorTiledCamera(
            model=self.model,
            config=SensorTiledCamera.Config(
                default_light=True,
                default_light_shadows=True,
                backface_culling=True,
            ),
        )
        self._cam_rays = self.camera_sensor.compute_pinhole_camera_rays(
            CAMERA_WIDTH, CAMERA_HEIGHT, [math.radians(CAMERA_FOV_DEG)]
        )
        self._cam_depth = self.camera_sensor.create_depth_image_output(
            CAMERA_WIDTH, CAMERA_HEIGHT, camera_count=1
        )
        self.cam_point_cloud = wp.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=wp.vec3f)
        self._first_cloud_np = None
        atexit.register(self._save_first_cloud)

        # Visualisation state (cm → m)
        self.viz_state = self.model.state()

        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale     = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # Scale the viewer's cached shape instance data (base / GL viewer path)
        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)
                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

    def create_articulation(self, builder):
        urdf_path = os.path.join(
            newton.examples.get_asset_directory(),
            "assets", "urdf", "franka_description", "robots", "franka_panda_gripper.urdf",
        )
        builder.add_urdf(
            urdf_path,
            xform=wp.transform((-50.0, -50.0, -10.0), wp.quat_identity()),
            floating=False,
            scale=100.0,   # URDF is in metres; simulation is in cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:7] = [0, 0, 0, -1.57079, 0, 1.57079, 0.7853]
        builder.joint_armature[:7] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.endeffector_id = builder.body_label.index("panda/panda_link7")

        # Camera mount offset from panda_link7 local frame (cm scale)
        tf_hand = wp.transform(
            wp.vec3(0.0, 0.0, 10.7),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -math.pi / 4),
        )
        tf_cam = wp.transform(
            wp.vec3(3.0, 0.0, 5.87),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi),
        )
        self.camera_offset = wp.transform_multiply(tf_hand, tf_cam)

        # Metre-scale version of camera_offset used for visualisation
        self.camera_offset_viz = wp.transform(
            wp.transform_get_translation(self.camera_offset) * float(SIM_TO_WORLD_SCALE),
            wp.transform_get_rotation(self.camera_offset),
        )

    def setup_ik(self):
        """Set up IK solver targeting panda_grip_site (collapsed into panda_link7).

        Grip-site offset in panda_link7 local frame:
          panda_hand_joint:    xyz=(0, 0, 10.7 cm),  rpy=(0, 0, -π/4)
          panda_grip_vis_joint: xyz=(0, 0, 10.25 cm), rpy=(0, 0, 0)
        Both translations along Z give link_offset=(0, 0, 20.95 cm).
        """
        n_dofs = self.model.joint_dof_count
        self._n_arm_dofs = n_dofs - 2   # 7 arm joints; last 2 DOFs are the fingers
        self._finger_dof1 = n_dofs - 2
        self._finger_dof2 = n_dofs - 1
        self.gripper_pos = GRIPPER_MAX
        n_coords = self.model.joint_coord_count

        self.target_joint_q = wp.clone(self.model.joint_q[:n_coords])
        tq = self.target_joint_q.numpy()
        tq[self._finger_dof1] = self.gripper_pos
        tq[self._finger_dof2] = self.gripper_pos
        self.target_joint_q.assign(tq)
        self.target_joint_qd = wp.zeros(n_dofs, dtype=float)

        # Grip-site offset from panda_link7 (cm)
        grip_link_offset = wp.vec3(0.0, 0.0, 20.95)
        grip_rot_offset  = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -math.pi / 4)

        body_q_np = self.state_0.body_q.numpy()
        ee_tf = body_q_np[self.endeffector_id]
        ee_pos = wp.vec3(float(ee_tf[0]), float(ee_tf[1]), float(ee_tf[2]))
        ee_rot = wp.quat(float(ee_tf[3]), float(ee_tf[4]), float(ee_tf[5]), float(ee_tf[6]))

        self._ee_target_pos = ee_pos + wp.quat_rotate(ee_rot, grip_link_offset)
        self._ee_target_rot = wp.normalize(ee_rot * grip_rot_offset)

        self._pos_obj = ik.IKObjectivePosition(
            link_index=self.endeffector_id,
            link_offset=grip_link_offset,
            target_positions=wp.array([self._ee_target_pos], dtype=wp.vec3),
        )
        q = self._ee_target_rot
        self._rot_obj = ik.IKObjectiveRotation(
            link_index=self.endeffector_id,
            link_offset_rotation=grip_rot_offset,
            target_rotations=wp.array([wp.vec4(q[0], q[1], q[2], q[3])], dtype=wp.vec4),
        )
        self._joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.clone(self.model.joint_limit_lower),
            joint_limit_upper=wp.clone(self.model.joint_limit_upper),
        )

        self.joint_q_ik = wp.clone(self.model.joint_q[:n_coords].reshape((1, n_coords)))

        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self._pos_obj, self._rot_obj, self._joint_limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def apply_ee_delta(self, delta_pos: np.ndarray, delta_rot: np.ndarray):
        """Accumulate EE pose target from world-frame deltas and solve IK.

        Args:
            delta_pos: World-frame EE position displacement [m], shape (3,).
            delta_rot: World-frame EE rotation displacement (rotation vector) [rad], shape (3,).
        """
        # Oculus reports deltas in metres; simulation is in cm
        self._ee_target_pos = self._ee_target_pos + wp.vec3(
            float(delta_pos[0] * WORLD_TO_SIM_SCALE),
            float(delta_pos[1] * WORLD_TO_SIM_SCALE),
            float(delta_pos[2] * WORLD_TO_SIM_SCALE),
        )

        angle = float(np.linalg.norm(delta_rot))
        if angle > 1e-6:
            axis = delta_rot / angle
            dq = wp.quat_from_axis_angle(
                wp.vec3(float(axis[0]), float(axis[1]), float(axis[2])), angle
            )
            self._ee_target_rot = wp.normalize(dq * self._ee_target_rot)

        self._pos_obj.set_target_position(0, self._ee_target_pos)
        q = self._ee_target_rot
        self._rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=IK_ITERS)
        wp.copy(self.target_joint_q[:self._n_arm_dofs], self.joint_q_ik[0, :self._n_arm_dofs])

    def apply_gripper(self, gripper_cmd: float):
        """Step gripper toward open or closed and write position target.

        Args:
            gripper_cmd: 1.0 = close, 0.0 = open.
        """
        if gripper_cmd > 0.5:
            self.gripper_pos = max(GRIPPER_CLOTH_MIN, self.gripper_pos - GRIPPER_SPEED * self.frame_dt)
        else:
            self.gripper_pos = min(GRIPPER_MAX, self.gripper_pos + GRIPPER_SPEED * self.frame_dt)

        tq = self.target_joint_q.numpy()
        tq[self._finger_dof1] = self.gripper_pos
        tq[self._finger_dof2] = self.gripper_pos
        self.target_joint_q.assign(tq)

    def _save_first_cloud(self):
        if self._first_cloud_np is None:
            return
        path = "first_point_cloud.npy"
        np.save(path, self._first_cloud_np)
        n_valid = (self._first_cloud_np != 0).any(axis=-1).sum()
        print(f"Saved first point cloud: {self._first_cloud_np.shape}, {n_valid} valid points → {path}")

    def step(self):
        if self._debug_replay_rows:
            if self._debug_replay_idx < len(self._debug_replay_rows):
                row = self._debug_replay_rows[self._debug_replay_idx]
                self._debug_replay_idx += 1
                delta_pos   = np.array([float(row["delta_pos_x"]),   float(row["delta_pos_y"]),   float(row["delta_pos_z"])])
                delta_rot   = np.array([float(row["delta_rot_x"]),   float(row["delta_rot_y"]),   float(row["delta_rot_z"])])
                gripper_cmd = float(row["gripper"])
            else:
                delta_pos, delta_rot, gripper_cmd = np.zeros(3), np.zeros(3), 0.0
        elif self.oculus is not None:
            action = self.oculus.get_action()
            delta_pos, delta_rot, gripper_cmd = action[:3], action[3:6], action[6]
        else:
            delta_pos, delta_rot, gripper_cmd = np.zeros(3), np.zeros(3), 0.0

        self.apply_ee_delta(delta_pos, delta_rot)
        self.apply_gripper(gripper_cmd)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self._sense()
        self.sim_time += self.frame_dt

        if self._debug_csv_writer is not None:
            self._write_debug_row(delta_pos, delta_rot, gripper_cmd)

    def _write_debug_row(self, delta_pos: np.ndarray, delta_rot: np.ndarray, gripper_cmd: float):
        body_q_np  = self.state_0.body_q.numpy()
        body_qd_np = self.state_0.body_qd.numpy()
        ee_tf  = body_q_np[self.endeffector_id]
        ee_vel = body_qd_np[self.endeffector_id]
        joint_q_np     = self.state_0.joint_q.numpy()
        joint_target_np = self.target_joint_q.numpy()
        tgt_pos = self._ee_target_pos
        tgt_rot = self._ee_target_rot

        row: dict = {
            "frame":    self._debug_frame,
            "sim_time": f"{self.sim_time:.6f}",
            "input_delta_pos_x": f"{delta_pos[0]:.7f}",
            "input_delta_pos_y": f"{delta_pos[1]:.7f}",
            "input_delta_pos_z": f"{delta_pos[2]:.7f}",
            "input_delta_rot_x": f"{delta_rot[0]:.7f}",
            "input_delta_rot_y": f"{delta_rot[1]:.7f}",
            "input_delta_rot_z": f"{delta_rot[2]:.7f}",
            "input_gripper":    f"{gripper_cmd:.1f}",
            "ee_pos_x": f"{ee_tf[0]:.6f}",
            "ee_pos_y": f"{ee_tf[1]:.6f}",
            "ee_pos_z": f"{ee_tf[2]:.6f}",
            "ee_quat_x": f"{ee_tf[3]:.6f}",
            "ee_quat_y": f"{ee_tf[4]:.6f}",
            "ee_quat_z": f"{ee_tf[5]:.6f}",
            "ee_quat_w": f"{ee_tf[6]:.6f}",
            "ee_body_ang_vel_x": f"{ee_vel[0]:.6f}",
            "ee_body_ang_vel_y": f"{ee_vel[1]:.6f}",
            "ee_body_ang_vel_z": f"{ee_vel[2]:.6f}",
            "ee_body_lin_vel_x": f"{ee_vel[3]:.6f}",
            "ee_body_lin_vel_y": f"{ee_vel[4]:.6f}",
            "ee_body_lin_vel_z": f"{ee_vel[5]:.6f}",
            "gripper_pos": f"{self.gripper_pos:.6f}",
            "ee_target_pos_x": f"{tgt_pos[0]:.6f}",
            "ee_target_pos_y": f"{tgt_pos[1]:.6f}",
            "ee_target_pos_z": f"{tgt_pos[2]:.6f}",
            "ee_target_quat_x": f"{tgt_rot[0]:.6f}",
            "ee_target_quat_y": f"{tgt_rot[1]:.6f}",
            "ee_target_quat_z": f"{tgt_rot[2]:.6f}",
            "ee_target_quat_w": f"{tgt_rot[3]:.6f}",
        }
        for i, v in enumerate(joint_q_np):
            row[f"joint_q_{i}"] = f"{v:.6f}"
        for i, v in enumerate(joint_target_np):
            row[f"joint_target_q_{i}"] = f"{v:.6f}"

        self._debug_csv_writer.writerow(row)
        self._debug_csv_file.flush()
        self._debug_frame += 1

    def _close_debug_log(self):
        if self._debug_csv_file is not None:
            self._debug_csv_file.close()
            self._debug_csv_file = None

    def _sense(self):
        wp.launch(
            _update_camera_transform,
            dim=1,
            inputs=[self.state_0.body_q, self.endeffector_id, self.camera_offset],
            outputs=[self._cam_tf],
        )
        self.camera_sensor.update(
            self.state_0,
            self._cam_tf,
            self._cam_rays,
            color_image=None,
            depth_image=self._cam_depth,
        )
        wp.launch(
            _depth_to_point_cloud,
            dim=(CAMERA_HEIGHT, CAMERA_WIDTH),
            inputs=[self._cam_depth, self._cam_rays, self._cam_tf],
            outputs=[self.cam_point_cloud],
        )
        if self._first_cloud_np is None:
            # Convert cm → m before saving
            self._first_cloud_np = self.cam_point_cloud.numpy().copy() * float(SIM_TO_WORLD_SCALE)

    def simulate(self):
        # Compute joint velocities so the robot tracks the IK target in one frame
        wp.launch(
            _compute_joint_qd,
            dim=self.model.joint_dof_count,
            inputs=[self.target_joint_q, self.state_0.joint_q, self.target_joint_qd, 1.0 / self.frame_dt],
        )

        self.cloth_solver.rebuild_bvh(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Robot step: kinematic FK integration, particles hidden, gravity disabled
            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)

            # Cloth step: VBD with full gravity and cloth-body contacts
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        if self.viewer is None:
            return

        # Scale cm → m into viz_state for rendering
        wp.launch(
            _scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                _scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        # Camera frame lines computed in m-space from viz_state
        wp.launch(
            _compute_frame_lines,
            dim=3,
            inputs=[self.viz_state.body_q, self.endeffector_id, self.camera_offset_viz, CAMERA_AXIS_LEN],
            outputs=[self._cam_starts, self._cam_ends],
        )

        # Swap shape data to m scale for rendering
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale     = self.viz_shape_scale

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.log_lines("/camera_frame", self._cam_starts, self._cam_ends, self._cam_colors)
        self.viewer.end_frame()

        # Restore cm-scale shape data
        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale     = self.sim_shape_scale

    def test_final(self):
        p_lower = wp.vec3(-100.0, -120.0, -5.0)
        p_upper = wp.vec3(100.0, 20.0, 80.0)
        newton.examples.test_particle_state(
            self.state_0,
            "cloth particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "cloth particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 200.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 70.0,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1000)
    parser.add_argument("--oculus-ip",    type=str, default=None, help="Quest IP address for teleoperation (omit to run without Oculus)")
    parser.add_argument("--debug-replay", type=str, default=None, metavar="CSV",
                        help="Replay actions from a recorded teleop CSV and write a debug log")
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args, oculus_ip=args.oculus_ip, debug_replay=args.debug_replay)
    newton.examples.run(example, args)
