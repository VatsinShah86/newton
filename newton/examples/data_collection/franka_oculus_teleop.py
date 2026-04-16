###########################################################################
# Franka Oculus Teleop
#
# Franka Panda robot on a table, driven by Oculus Quest controller
# teleoperation via QuestStream. End-effector position and orientation
# deltas are mapped to joint velocities via the pseudoinverse Jacobian.
# Setting all deltas to zero holds the robot in position.
#
# Command: python newton/examples/data_collection/franka_oculus_teleop.py
#
###########################################################################

from __future__ import annotations

import atexit
import math
import os
import time

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ModelBuilder, State, eval_fk
from newton.math import transform_twist
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


@wp.kernel
def _compute_ee_body_out(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    ee_id: int,
    ee_offset: wp.transform,
    body_out: wp.array(dtype=float),
):
    """Compute end-effector body velocity projected through TCP offset.

    body_qd is stored in world frame by Newton's FK, so the TCP offset
    translation must be rotated to world frame before computing the twist.
    """
    body_tf = body_q[ee_id]
    p_local = wp.transform_get_translation(ee_offset)
    p_world = wp.transform_vector(body_tf, p_local)
    ee_world_offset = wp.transform(p_world, wp.quat_identity())
    mv = transform_twist(ee_world_offset, body_qd[ee_id])
    for i in range(6):
        body_out[i] = mv[i]


class Example:
    def __init__(self, viewer, args=None, oculus_ip: str | None = None):
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

        self.set_up_control()

        # gravity arrays for swapping during simulate()
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array([wp.vec3(0.0, 0.0, -9.81)], dtype=wp.vec3)

        self.oculus = QuestStream(ip=oculus_ip) if oculus_ip is not None else None
        self._last_step_time: float | None = None

        # Camera mount frame: placed at the TCP (0.22 m along body-7 z-axis).
        # Adjust this transform to move/orient the camera relative to the TCP.
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

        # tool-center-point offset along the gripper z-axis [m]
        self.endeffector_offset = wp.transform(wp.vec3(0.0, 0.0, 0.22), wp.quat_identity())

    def set_up_control(self):
        n_dofs = self.model.joint_dof_count

        # target joint velocities; zero = hold position
        self.target_joint_qd = wp.zeros(n_dofs, dtype=float)

        # finger DOF is the single prismatic joint (panda_finger_joint1)
        self._finger_dof = n_dofs - 1
        self.gripper_pos = GRIPPER_MAX  # start fully open

        # Jacobian buffers: 6 rows (linear + angular EE velocity) x n_dofs columns
        self._n_dofs = n_dofs
        self._J_flat = wp.empty(6 * n_dofs, dtype=float)
        self._body_out = wp.empty(6, dtype=float, requires_grad=True)
        self._temp_state = self.model.state(requires_grad=True)

        # one-hot gradient seeds for each of the 6 EE velocity components
        self._one_hots = [
            wp.array([1.0 if j == i else 0.0 for j in range(6)], dtype=float)
            for i in range(6)
        ]

    def _compute_jacobian(self, state: State) -> np.ndarray:
        """Compute the 6 x n_dofs end-effector Jacobian via autodiff.

        Differentiates EE velocity (linear + angular) w.r.t. joint velocities.

        Args:
            state: Current simulation state.

        Returns:
            J: ndarray of shape (6, n_dofs).
        """
        joint_q = state.joint_q
        joint_qd = state.joint_qd
        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        tape = wp.Tape()
        with tape:
            eval_fk(self.model, joint_q, joint_qd, self._temp_state)
            wp.launch(
                _compute_ee_body_out,
                dim=1,
                inputs=[self._temp_state.body_q, self._temp_state.body_qd, self.endeffector_id, self.endeffector_offset],
                outputs=[self._body_out],
            )

        for i in range(6):
            tape.backward(grads={self._body_out: self._one_hots[i]})
            wp.copy(self._J_flat[i * self._n_dofs : (i + 1) * self._n_dofs], joint_qd.grad)
            tape.zero()

        return self._J_flat.numpy().reshape(6, self._n_dofs)

    def apply_ee_delta(self, delta_pos: np.ndarray, delta_rot: np.ndarray, dt: float):
        """Map an end-effector delta command to joint velocities.

        Uses the pseudoinverse Jacobian to convert desired EE displacement into
        joint velocities. Call with zero arrays to hold the current position.

        Args:
            delta_pos: Desired EE position displacement this frame [m], shape (3,).
            delta_rot: Desired EE rotation displacement this frame [rad], shape (3,).
            dt: Actual elapsed wall-clock time since the last step [s].
        """
        ee_delta = np.concatenate([delta_pos, delta_rot]).astype(np.float32)

        if np.allclose(ee_delta, 0.0):
            self.target_joint_qd.zero_()
            return

        J = self._compute_jacobian(self.state_0)
        # Divide by dt so Featherstone's q += qd * sim_dt integration
        # accumulates exactly the desired EE displacement over the real frame time.
        joint_qd = (np.linalg.pinv(J) @ ee_delta / dt).astype(np.float32)
        self.target_joint_qd.assign(joint_qd)

    def _save_first_cloud(self):
        if self._first_cloud_np is None:
            return
        path = "first_point_cloud.npy"
        np.save(path, self._first_cloud_np)
        n_valid = (self._first_cloud_np != 0).any(axis=-1).sum()
        print(f"Saved first point cloud: {self._first_cloud_np.shape}, {n_valid} valid points → {path}")

    def apply_gripper(self, gripper_cmd: float, dt: float):
        """Move fingers toward open or closed at a fixed speed.

        Args:
            gripper_cmd: 1.0 = close, 0.0 = open.
            dt: Actual elapsed wall-clock time since the last step [s].
        """
        if gripper_cmd > 0.5:
            target = max(GRIPPER_MIN, self.gripper_pos - GRIPPER_SPEED * dt)
        else:
            target = min(GRIPPER_MAX, self.gripper_pos + GRIPPER_SPEED * dt)

        finger_vel = float((target - self.gripper_pos) / dt)
        self.gripper_pos = target

        # Override the finger DOF — must come after apply_ee_delta so the
        # pseudoinverse result for that column is replaced.
        qd = self.target_joint_qd.numpy()
        qd[self._finger_dof] = finger_vel
        self.target_joint_qd.assign(qd)

    def step(self):
        now = time.monotonic()
        if self._last_step_time is None:
            dt = self.frame_dt
        else:
            dt = now - self._last_step_time
        self._last_step_time = now

        if self.oculus is not None:
            action = self.oculus.get_action()
            delta_pos, delta_rot, gripper_cmd = action[:3], action[3:6], action[6]
        else:
            delta_pos, delta_rot, gripper_cmd = np.zeros(3), np.zeros(3), 0.0
        self.apply_ee_delta(delta_pos, delta_rot, dt)
        self.apply_gripper(gripper_cmd, dt)
        self.simulate()
        self._sense()
        self.sim_time += self.frame_dt

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
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # --- Robot step (kinematic, no particles, no gravity, no contacts) ---
            # Hide particles from Featherstone so it acts as a pure kinematic FK
            # integrator: integrates joint_q from joint_qd with no dynamics.
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
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args, oculus_ip=args.oculus_ip)
    newton.examples.run(example, args)