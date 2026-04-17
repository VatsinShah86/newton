###########################################################################
# Franka Oculus Teleop
#
# Franka Panda robot on a table, driven by Oculus Quest controller
# teleoperation via QuestStream. End-effector pose targets are accumulated
# from per-frame deltas and solved via IK; joint positions are tracked by
# Featherstone PD control.
#
# Command: python newton/examples/data_collection/franka_oculus_teleop.py
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

GRIPPER_SPEED = 0.05   # m/s — full stroke (0 → 0.04 m) in ~0.8 s
GRIPPER_MAX   = 0.04   # m, fully open (Franka finger travel limit)
GRIPPER_MIN   = 0.0    # m, fully closed

CAMERA_AXIS_LEN = 0.05  # m, length of each displayed frame axis
CAMERA_WIDTH    = 640
CAMERA_HEIGHT   = 480
CAMERA_FOV_DEG  = 60.0

ROPE_N_PARTICLES  = 25          # number of particles in the rope chain
ROPE_LENGTH       = 0.5         # m, total rope length
ROPE_RADIUS       = 0.008       # m, particle collision radius (~8 mm)
ROPE_PARTICLE_MASS = 0.02       # kg per particle
ROPE_KE           = 5e3         # N/m, spring stretch stiffness
ROPE_KD           = 10.0        # N·s/m, spring damping
ROPE_CONTACT_MARGIN = 0.015     # m, particle–body soft contact detection distance

# IK iterations per step
IK_ITERS = 24


@wp.kernel
def _compute_joint_qd(
    target_q: wp.array(dtype=float),
    current_q: wp.array(dtype=float),
    out_qd: wp.array(dtype=float),
    inv_frame_dt: float,
):
    """Compute joint velocity to reach target in one frame: qd = (target - q) / frame_dt."""
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
    """One thread per axis (0=X, 1=Y, 2=Z): compute start/end in world space."""
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
    depth: wp.array(dtype=wp.float32, ndim=4),    # (worlds, cameras, H, W)
    rays: wp.array(dtype=wp.vec3f, ndim=4),        # (cameras, H, W, 2) — [..,0]=origin [..,1]=dir
    cam_tf: wp.array(dtype=wp.transformf, ndim=2), # (cameras, worlds)
    out: wp.array(dtype=wp.vec3f, ndim=2),         # (H, W) — zero vec = no hit
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
def _compute_rope_segment_xforms(
    particle_q: wp.array(dtype=wp.vec3),
    particle_offset: int,
    out_xforms: wp.array(dtype=wp.transform),
):
    """One thread per rope segment: compute midpoint transform oriented along the segment."""
    i = wp.tid()
    p0 = particle_q[particle_offset + i]
    p1 = particle_q[particle_offset + i + 1]
    mid = (p0 + p1) * 0.5
    seg = p1 - p0
    seg_len = wp.length(seg)
    if seg_len < 1.0e-6:
        q = wp.quat_identity()
    else:
        d = seg / seg_len
        z = wp.vec3(0.0, 0.0, 1.0)
        c = wp.dot(z, d)
        if c > 0.9999:
            q = wp.quat_identity()
        elif c < -0.9999:
            # 180° rotation around X axis
            q = wp.quat(1.0, 0.0, 0.0, 0.0)
        else:
            axis = wp.normalize(wp.cross(z, d))
            q = wp.quat_from_axis_angle(axis, wp.acos(c))
    out_xforms[i] = wp.transform(mid, q)


class Example:
    def __init__(self, viewer, args=None, oculus_ip: str | None = None, debug_replay: str | None = None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.scene = ModelBuilder(gravity=-9.81)
        self.viewer = viewer

        # create robot
        franka = ModelBuilder()
        self.create_articulation(franka)
        self.scene.add_world(franka)

        # Rope: particle chain lying along X at a height just above the ground,
        # positioned within the robot's reachable workspace.
        seg_len = ROPE_LENGTH / (ROPE_N_PARTICLES - 1)
        rope_start = wp.vec3(-0.2, -0.3, ROPE_RADIUS + 0.002)
        self._rope_particle_offset = self.scene.particle_count
        for i in range(ROPE_N_PARTICLES):
            self.scene.add_particle(
                pos=rope_start + wp.vec3(i * seg_len, 0.0, 0.0),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=ROPE_PARTICLE_MASS,
                radius=ROPE_RADIUS,
            )
        for i in range(ROPE_N_PARTICLES - 1):
            self.scene.add_spring(
                i=self._rope_particle_offset + i,
                j=self._rope_particle_offset + i + 1,
                ke=ROPE_KE,
                kd=ROPE_KD,
                control=0.0,
            )

        # color() must be called after all particles/springs and before finalize
        self.scene.color()
        self.scene.add_ground_plane()
        self.model = self.scene.finalize(requires_grad=False)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)

        # Particle–body contact material (applied to all robot shapes)
        self.model.soft_contact_ke = 1e4
        self.model.soft_contact_kd = 1e-2
        self.model.soft_contact_mu = 0.8

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[...] = 5e4
        shape_kd[...] = 1e-3
        shape_mu[...] = 1.0
        self.model.shape_material_ke = wp.array(shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.device)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.device)
        self.model.shape_material_mu = wp.array(shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.device)

        # VBD solver for rope particles; Featherstone handles robot rigid bodies
        # particle_enable_tile_solve=False: the tiled CUDA kernel assumes mesh/tet
        # topology and crashes on pure particle+spring systems like this rope.
        self.rope_solver = SolverVBD(
            self.model,
            iterations=5,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
            rigid_contact_k_start=self.model.soft_contact_ke,
        )

        # Explicit collision pipeline for particle–shape contacts
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=ROPE_CONTACT_MARGIN,
        )
        self.contacts = self.collision_pipeline.contacts()

        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Gravity arrays for swapping: Featherstone step runs with zero gravity
        # (PD control provides torques; adding gravity makes dynamics unstable without
        # a gravity-compensation term). VBD rope step gets full earth gravity.
        self.gravity_zero  = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)

        self.setup_ik()

        self.oculus = QuestStream(ip=oculus_ip) if oculus_ip is not None else None

        # Debug replay state (None when not active)
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

        # Camera frame visualization buffers
        self._cam_starts = wp.zeros(3, dtype=wp.vec3)
        self._cam_ends   = wp.zeros(3, dtype=wp.vec3)
        self._cam_colors = wp.array(
            [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)],
            dtype=wp.vec3,
        )

        # (camera_count=1, world_count=1) transform buffer for the sensor
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

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # Rope cable rendering: represent each spring segment as a capsule
        self.viewer.show_particles = False
        n_segs = ROPE_N_PARTICLES - 1
        seg_half_len = (ROPE_LENGTH / (ROPE_N_PARTICLES - 1)) * 0.5
        self._rope_seg_xforms = wp.zeros(n_segs, dtype=wp.transform)
        self._rope_seg_scales = wp.full(n_segs, wp.vec3(ROPE_RADIUS, ROPE_RADIUS, seg_half_len), dtype=wp.vec3)
        brown = wp.vec3(0.45, 0.27, 0.08)
        self._rope_seg_colors = wp.full(n_segs, brown, dtype=wp.vec3)

    def create_articulation(self, builder):
        urdf_path = os.path.join(newton.examples.get_asset_directory(), "assets", "urdf", "franka_description", "robots", "franka_panda_gripper.urdf")
        builder.add_urdf(
            urdf_path,
            xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        # rest pose: arm up, slightly bent
        builder.joint_q[:7] = [0, 0, 0, -1.57079, 0, 1.57079, 0.7853]

        # Kinematic control: no PD gains needed. Featherstone is used as a pure
        # kinematic integrator — joint velocities are injected each substep so the
        # robot tracks IK targets without any force-based dynamics.
        # Small armature stabilises the mass-matrix inversion for FK/IK purposes.
        builder.joint_armature[:7] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.endeffector_id = builder.body_label.index("panda/panda_link7")

        # Camera offset: chain of fixed joints from panda_link7 to camera_link
        # (fixed joints are collapsed, so camera_link is merged into panda_link7's body)
        # panda_hand_joint: xyz=(0, 0, 0.107), rpy=(0, 0, -π/4)
        # cam joint:        xyz=(0.03, 0, 0.0587), rpy=(π, 0, 0)
        tf_hand = wp.transform(
            wp.vec3(0.0, 0.0, 0.107),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -math.pi / 4),
        )
        tf_cam = wp.transform(
            wp.vec3(0.03, 0.0, 0.0587),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi),
        )
        self.camera_offset = wp.transform_multiply(tf_hand, tf_cam)

    def setup_ik(self):
        """Set up IK solver targeting panda_grip_site.

        panda_grip_site is a fixed link collapsed into panda_link7. Its pose in
        panda_link7's local frame is derived by chaining the two fixed joints:
          panda_hand_joint: xyz=(0, 0, 0.107), rpy=(0, 0, -π/4)
          panda_grip_vis_joint: xyz=(0, 0, 0.1025), rpy=(0, 0, 0)
        Since both translations are along Z, the Z-rotation doesn't change the
        offset magnitude, giving link_offset=(0, 0, 0.2095) and
        link_offset_rotation=Rz(-π/4).
        """
        n_dofs = self.model.joint_dof_count
        self._n_arm_dofs = n_dofs - 1   # 7 arm joints; last DOF is the finger
        self._finger_dof = n_dofs - 1
        self.gripper_pos = GRIPPER_MAX
        n_coords = self.model.joint_coord_count

        # Kinematic target buffers (1-D, DOF-indexed).
        # target_joint_q: desired joint positions updated by IK + gripper control.
        # target_joint_qd: joint velocities injected into state_0 each substep so
        #   Featherstone acts as a pure kinematic integrator.
        self.target_joint_q = wp.clone(self.model.joint_q[:n_coords])
        tq = self.target_joint_q.numpy()
        tq[self._finger_dof] = self.gripper_pos
        self.target_joint_q.assign(tq)
        self.target_joint_qd = wp.zeros(n_dofs, dtype=float)

        # Grip-site offset from panda_link7's local frame
        grip_link_offset = wp.vec3(0.0, 0.0, 0.2095)
        grip_rot_offset  = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -math.pi / 4)

        # Initial grip-site world pose from FK state
        body_q_np = self.state_0.body_q.numpy()
        ee_tf = body_q_np[self.endeffector_id]  # [px,py,pz, qx,qy,qz,qw]
        ee_pos = wp.vec3(float(ee_tf[0]), float(ee_tf[1]), float(ee_tf[2]))
        ee_rot = wp.quat(float(ee_tf[3]), float(ee_tf[4]), float(ee_tf[5]), float(ee_tf[6]))

        # Grip-site world position: body_pos + body_rot * link_offset
        self._ee_target_pos = ee_pos + wp.quat_rotate(ee_rot, grip_link_offset)
        # Grip-site world rotation: body_rot * rot_offset
        self._ee_target_rot = wp.normalize(ee_rot * grip_rot_offset)

        # IK objectives
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

        # Full joint coord array shaped (1, n_coords) for n_problems=1
        self.joint_q_ik = wp.clone(
            self.model.joint_q[:n_coords].reshape((1, n_coords))
        )

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
        # Accumulate position
        self._ee_target_pos = self._ee_target_pos + wp.vec3(
            float(delta_pos[0]), float(delta_pos[1]), float(delta_pos[2])
        )

        # Accumulate rotation: convert rotation vector to quaternion, pre-multiply
        angle = float(np.linalg.norm(delta_rot))
        if angle > 1e-6:
            axis = delta_rot / angle
            dq = wp.quat_from_axis_angle(
                wp.vec3(float(axis[0]), float(axis[1]), float(axis[2])), angle
            )
            self._ee_target_rot = wp.normalize(dq * self._ee_target_rot)

        # Push updated targets into IK objectives
        self._pos_obj.set_target_position(0, self._ee_target_pos)
        q = self._ee_target_rot
        self._rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

        # Solve IK and write arm joint targets into the kinematic target buffer
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=IK_ITERS)
        wp.copy(self.target_joint_q[:self._n_arm_dofs], self.joint_q_ik[0, :self._n_arm_dofs])

    def apply_gripper(self, gripper_cmd: float):
        """Step gripper toward open or closed and write position target.

        Args:
            gripper_cmd: 1.0 = close, 0.0 = open.
        """
        if gripper_cmd > 0.5:
            self.gripper_pos = max(GRIPPER_MIN, self.gripper_pos - GRIPPER_SPEED * self.frame_dt)
        else:
            self.gripper_pos = min(GRIPPER_MAX, self.gripper_pos + GRIPPER_SPEED * self.frame_dt)

        tq = self.target_joint_q.numpy()
        tq[self._finger_dof] = self.gripper_pos
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
                delta_pos = np.array([float(row["delta_pos_x"]), float(row["delta_pos_y"]), float(row["delta_pos_z"])])
                delta_rot = np.array([float(row["delta_rot_x"]), float(row["delta_rot_y"]), float(row["delta_rot_z"])])
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
        self.simulate()
        self._sense()
        self.sim_time += self.frame_dt

        if self._debug_csv_writer is not None:
            self._write_debug_row(delta_pos, delta_rot, gripper_cmd)

    def _write_debug_row(self, delta_pos: np.ndarray, delta_rot: np.ndarray, gripper_cmd: float):
        # body_q layout: [px, py, pz, qx, qy, qz, qw] per body
        # body_qd layout: [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z] per body (spatial vector)
        body_q_np = self.state_0.body_q.numpy()
        body_qd_np = self.state_0.body_qd.numpy()
        ee_tf = body_q_np[self.endeffector_id]
        ee_vel = body_qd_np[self.endeffector_id]
        joint_q_np = self.state_0.joint_q.numpy()
        joint_target_np = self.target_joint_q.numpy()
        tgt_pos = self._ee_target_pos
        tgt_rot = self._ee_target_rot

        row: dict = {
            "frame": self._debug_frame,
            "sim_time": f"{self.sim_time:.6f}",
            "input_delta_pos_x": f"{delta_pos[0]:.7f}",
            "input_delta_pos_y": f"{delta_pos[1]:.7f}",
            "input_delta_pos_z": f"{delta_pos[2]:.7f}",
            "input_delta_rot_x": f"{delta_rot[0]:.7f}",
            "input_delta_rot_y": f"{delta_rot[1]:.7f}",
            "input_delta_rot_z": f"{delta_rot[2]:.7f}",
            "input_gripper": f"{gripper_cmd:.1f}",
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
            self._first_cloud_np = self.cam_point_cloud.numpy().copy()

    def simulate(self):
        # Compute joint velocities once per frame so the robot reaches the IK target
        # in exactly one frame_dt (kinematic tracking, always stable).
        wp.launch(
            _compute_joint_qd,
            dim=self.model.joint_dof_count,
            inputs=[self.target_joint_q, self.state_0.joint_q, self.target_joint_qd, 1.0 / self.frame_dt],
        )

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # --- Robot step (kinematic, no particles, no contacts) ---
            # Inject the pre-computed joint velocity so Featherstone acts as a
            # pure kinematic integrator (q += qd*dt). Zero gravity so no dynamics
            # forces perturb the injected motion.
            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)

            # --- Rope step (VBD with gravity and particle–body contacts) ---
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.rope_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        if self.viewer is None:
            return

        wp.launch(
            _compute_frame_lines,
            dim=3,
            inputs=[self.state_0.body_q, self.endeffector_id, self.camera_offset, CAMERA_AXIS_LEN],
            outputs=[self._cam_starts, self._cam_ends],
        )

        wp.launch(
            _compute_rope_segment_xforms,
            dim=ROPE_N_PARTICLES - 1,
            inputs=[self.state_0.particle_q, self._rope_particle_offset],
            outputs=[self._rope_seg_xforms],
        )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("/camera_frame", self._cam_starts, self._cam_ends, self._cam_colors)
        self.viewer.log_capsules("/rope/segments", "", self._rope_seg_xforms, self._rope_seg_scales, self._rope_seg_colors, None)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 0.7,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1000)
    parser.add_argument("--oculus-ip", type=str, default=None, help="Quest IP address for teleoperation (omit to run without Oculus)")
    parser.add_argument("--debug-replay", type=str, default=None, metavar="CSV", help="Replay actions from a recorded teleop CSV and write a debug log (e.g. teleop.csv)")
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args, oculus_ip=args.oculus_ip, debug_replay=args.debug_replay)
    newton.examples.run(example, args)
