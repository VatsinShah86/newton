###########################################################################
# Franka DP3 Eval — T-Shirt
#
# Franka Panda robot with a T-shirt on the ground, driven by a trained DP3
# policy. Observations are built from the same state + point-cloud format as
# the teleoperation dataset, then fed into the policy for chunked action
# inference. Cloth is simulated by VBD with full self-contact.
#
# Command:
#   python third_party/newton/newton/examples/data_collection/tshirt_eval.py \
#       --checkpoint path/to/latest.ckpt --viewer null --video-output-path out.mp4
#
###########################################################################

from __future__ import annotations

import atexit
import csv
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import warp as wp
import zarr
from pxr import Usd

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[5]
DP3_ROOT = REPO_ROOT / "3D-Diffusion-Policy"

if str(DP3_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_ROOT))

import newton
import newton.examples
import newton.ik as ik
import newton.usd
from train import TrainDP3Workspace
from newton import ModelBuilder, eval_fk
from newton.sensors import SensorTiledCamera
from newton.solvers import SolverFeatherstone, SolverVBD

GRIPPER_SPEED = 0.08   # m/s — full stroke (0 → 0.04 m) in ~0.5 s
GRIPPER_MAX   = 0.04   # m, fully open (Franka finger travel limit)
GRIPPER_MIN   = 0.002  # m, close limit for cloth (prevents finger interpenetration)

CAMERA_AXIS_LEN = 0.05  # m, length of each displayed frame axis
CAMERA_WIDTH    = 640
CAMERA_HEIGHT   = 480
CAMERA_FOV_DEG  = 60.0

# Point cloud subsampling: sample a 32×32 uniform grid from the full 640×480 depth
# image → exactly 1024 points while preserving the full 4:3 field of view.
PC_COLS     = 32
PC_ROWS     = 32
PC_STRIDE_X = CAMERA_WIDTH  // PC_COLS   # 20
PC_STRIDE_Y = CAMERA_HEIGHT // PC_ROWS   # 15
PC_N_POINTS = PC_COLS * PC_ROWS          # 1024

# Shirt mesh (unisex_shirt.usd) is stored in centimetres; scale=0.01 converts to metres.
SHIRT_MESH_SCALE         = 0.006
SHIRT_POS                = wp.vec3(0.05, 0.2, 0.0025)   # m, above ground in robot workspace
SHIRT_ROT                = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi)
SHIRT_DENSITY            = 0.02    # kg/m²
SHIRT_PARTICLE_RADIUS    = 0.008   # m, cloth collision radius
SHIRT_BODY_CONTACT_MARGIN = 0.008  # m, cloth-body contact detection distance
SHIRT_SELF_CONTACT_RADIUS = 0.002  # m
SHIRT_SELF_CONTACT_MARGIN = 0.002  # m
SHIRT_TRI_KE             = 1.0e3   # N/m, in-plane stretch stiffness
SHIRT_TRI_KA             = 1.0e3   # N/m, area preservation stiffness
SHIRT_TRI_KD             = 1.0e-6  # damping
SHIRT_BENDING_KE         = 0.01     # N·m/rad, bending stiffness
SHIRT_BENDING_KD         = 1.0e-4  # N·m·s/rad, bending damping

# Default zarr dataset path (created when --record is passed)
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "teleop_dataset.zarr")

# IK iterations per step
IK_ITERS = 24


def _stack_history(history: deque[np.ndarray], n_steps: int) -> np.ndarray:
    """Pad a deque of observations by repeating the oldest available frame."""
    items = list(history)
    if not items:
        raise ValueError("Cannot stack an empty observation history.")

    latest = items[-1]
    result = np.zeros((n_steps,) + latest.shape, dtype=latest.dtype)
    count = min(n_steps, len(items))
    start = n_steps - count
    result[start:] = np.asarray(items[-count:], dtype=latest.dtype)
    if start > 0:
        result[:start] = result[start]
    return result


def _resolve_repo_path(path_str: str) -> Path:
    """Resolve a user path against cwd first, then the repository root."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (REPO_ROOT / path).resolve()


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
    stride_y: int,
    stride_x: int,
    n_cols: int,
    out: wp.array(dtype=wp.vec3f),                 # flat (PC_N_POINTS,) — zero vec = no hit
):
    """Sample one depth pixel per thread at a uniform stride and unproject to world space.

    Launch with dim=(PC_ROWS, PC_COLS) so every thread maps to one of the 1024 output
    points while still reading from the full-resolution depth image.
    """
    row, col = wp.tid()
    src_y = row * stride_y
    src_x = col * stride_x
    d = depth[0, 0, src_y, src_x]
    idx = row * n_cols + col
    if d <= 0.0:
        out[idx] = wp.vec3f(0.0, 0.0, 0.0)
        return
    tf = cam_tf[0, 0]
    ray_dir_world = wp.transform_vector(tf, rays[0, src_y, src_x, 1])
    out[idx] = wp.transform_get_translation(tf) + d * ray_dir_world


class Example:
    def __init__(
        self,
        viewer,
        args=None,
        checkpoint: str | None = None,
        policy_device: str = "cuda:0",
        use_ema: bool = True,
        debug_rope: bool = False,
        record: bool = False,
    ):
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

        # T-shirt: cloth mesh lying on the ground in the robot's reachable workspace.
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
        shirt_mesh = newton.usd.get_mesh(usd_prim)
        vertices = [wp.vec3(v) for v in shirt_mesh.vertices]

        self._cloth_particle_offset = self.scene.particle_count
        self.scene.add_cloth_mesh(
            vertices=vertices,
            indices=shirt_mesh.indices,
            pos=SHIRT_POS,
            rot=SHIRT_ROT,
            vel=wp.vec3(0.0, 0.0, 0.0),
            scale=SHIRT_MESH_SCALE,
            density=SHIRT_DENSITY,
            tri_ke=SHIRT_TRI_KE,
            tri_ka=SHIRT_TRI_KA,
            tri_kd=SHIRT_TRI_KD,
            edge_ke=SHIRT_BENDING_KE,
            edge_kd=SHIRT_BENDING_KD,
            particle_radius=SHIRT_PARTICLE_RADIUS,
        )

        # color() must be called after all particles and before finalize
        self.scene.color()
        self.scene.add_ground_plane()
        self.model = self.scene.finalize(requires_grad=False)

        # Flat rest angles so the shirt wants to be a flat sheet
        self.model.edge_rest_angle.zero_()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)

        # Cloth–ground contact: moderate stiffness, low friction so the shirt
        # slides freely on the ground but is still grippable by the fingers.
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e-2
        self.model.soft_contact_mu = 0.4

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[...] = 5.0e4
        shape_kd[...] = 1.0e-3
        shape_mu[...] = 1.5   # finger–cloth friction
        self.model.shape_material_ke = wp.array(shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.device)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.device)
        self.model.shape_material_mu = wp.array(shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.device)

        # VBD solver for cloth with full self-contact.
        # integrate_with_external_rigid_solver=True: Featherstone owns the robot bodies.
        self.cloth_solver = SolverVBD(
            self.model,
            iterations=5,
            integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=SHIRT_SELF_CONTACT_RADIUS,
            particle_self_contact_margin=SHIRT_SELF_CONTACT_MARGIN,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.005,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=self.model.soft_contact_ke,
        )

        # Explicit collision pipeline for cloth–body contacts
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=SHIRT_BODY_CONTACT_MARGIN,
        )
        self.contacts = self.collision_pipeline.contacts()

        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Gravity arrays for swapping: Featherstone step runs with zero gravity
        # (PD control provides torques; adding gravity makes dynamics unstable without
        # a gravity-compensation term). VBD rope step gets full earth gravity.
        self.gravity_zero  = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)

        self.setup_ik()

        # Debug replay state (None when not active)
        self._debug_csv_file = None
        self._debug_csv_writer = None
        self._debug_frame: int = 0

        # Rope force debug logging
        self._debug_rope_csv_file = None
        self._debug_rope_csv_writer = None
        self._debug_rope_frame = 0
        if debug_rope:
            # Build shape→body label lookup now, before any state is modified
            shape_body_np = self.model.shape_body.numpy()
            body_labels = self.model.body_label  # list[str], one per body
            self._shape_body_name: list[str] = []
            for shape_idx in range(len(shape_body_np)):
                b = int(shape_body_np[shape_idx])
                if b < 0 or b >= len(body_labels):
                    self._shape_body_name.append("world/ground")
                else:
                    self._shape_body_name.append(body_labels[b])

            out_path = "rope_forces_debug.csv"
            self._debug_rope_csv_file = open(out_path, "w", newline="")
            fieldnames = [
                "frame", "sim_time",
                "particle_idx",           # 0-indexed within rope
                "pos_x", "pos_y", "pos_z",
                "net_force_x", "net_force_y", "net_force_z", "net_force_mag",
                "n_contacts",             # number of active contacts for this particle
                "contact_body",           # body applying this contact force (one row per contact)
                "contact_normal_x", "contact_normal_y", "contact_normal_z",
            ]
            self._debug_rope_csv_writer = csv.DictWriter(
                self._debug_rope_csv_file, fieldnames=fieldnames
            )
            self._debug_rope_csv_writer.writeheader()
            atexit.register(self._close_rope_debug_log)
            print(f"Rope force debug logging → {out_path}")

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
        self._cam_color = self.camera_sensor.create_color_image_output(
            CAMERA_WIDTH, CAMERA_HEIGHT, camera_count=1
        )
        self.cam_point_cloud = wp.zeros(PC_N_POINTS, dtype=wp.vec3f)

        # Dataset recording (activated by --record flag)
        self._record = record
        self._record_actions: list[np.ndarray] = []
        self._record_states: list[np.ndarray] = []
        self._record_clouds: list[np.ndarray] = []
        self._latest_point_cloud = np.zeros((PC_N_POINTS, 6), dtype=np.float32)

        self._load_policy(checkpoint=checkpoint, policy_device=policy_device, use_ema=use_ema)
        self._action_queue: deque[np.ndarray] = deque()
        self._obs_history_states: deque[np.ndarray] = deque(maxlen=self.n_obs_steps)
        self._obs_history_clouds: deque[np.ndarray] = deque(maxlen=self.n_obs_steps)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)
        self._sense()
        self._append_current_observation()

    def create_articulation(self, builder):
        urdf_path = os.path.join(newton.examples.get_asset_directory(), "assets", "urdf", "franka_description", "robots", "franka_panda_gripper.urdf")
        builder.add_urdf(
            urdf_path,
            xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )
        # rest pose: arm up, slightly bent
        builder.joint_q[:7] = [0, 0, 0, -1.57079, 0, 1.57079, 0.7853]

        # Small armature stabilises the mass-matrix inversion for FK/IK purposes.
        builder.joint_armature[:7] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.endeffector_id = builder.body_label.index("panda/panda_link7")

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
        self._n_arm_dofs = n_dofs - 2   # 7 arm joints; last 2 DOFs are the fingers
        self._finger_dof1 = n_dofs - 2  # panda_finger_joint1 (left, +Y)
        self._finger_dof2 = n_dofs - 1  # panda_finger_joint2 (right, -Y)
        self.gripper_pos = GRIPPER_MAX
        n_coords = self.model.joint_coord_count

        # Kinematic target buffers (1-D, DOF-indexed).
        # target_joint_q: desired joint positions updated by IK + gripper control.
        # target_joint_qd: joint velocities injected into state_0 each substep so
        #   Featherstone acts as a pure kinematic integrator.
        self.target_joint_q = wp.clone(self.model.joint_q[:n_coords])
        tq = self.target_joint_q.numpy()
        tq[self._finger_dof1] = self.gripper_pos
        tq[self._finger_dof2] = self.gripper_pos
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
        tq[self._finger_dof1] = self.gripper_pos
        tq[self._finger_dof2] = self.gripper_pos
        self.target_joint_q.assign(tq)

    @staticmethod
    def _quat_to_euler(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [qx, qy, qz, qw] to ZYX Euler angles [roll, pitch, yaw] in radians."""
        qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        roll  = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
        yaw   = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _load_policy(self, checkpoint: str | None, policy_device: str, use_ema: bool):
        if checkpoint is None:
            raise ValueError("--checkpoint is required for tshirt_eval.py")

        checkpoint_path = _resolve_repo_path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        workspace = TrainDP3Workspace.create_from_checkpoint(checkpoint_path)
        policy = workspace.ema_model if use_ema and workspace.ema_model is not None else workspace.model
        self.policy = policy
        self.policy.eval()
        self.policy.to(torch.device(policy_device))
        self.policy.reset()
        self.policy_device = torch.device(policy_device)
        self.n_obs_steps = int(self.policy.n_obs_steps)
        self.n_action_steps = int(self.policy.n_action_steps)
        self.checkpoint_path = checkpoint_path
        self.policy_source = "ema_model" if policy is workspace.ema_model else "model"

        print(
            f"Loaded DP3 {self.policy_source} from {self.checkpoint_path} "
            f"(n_obs_steps={self.n_obs_steps}, n_action_steps={self.n_action_steps}, device={self.policy_device})"
        )

    def _current_agent_state(self) -> np.ndarray:
        body_q_np = self.state_0.body_q.numpy()
        ee_tf = body_q_np[self.endeffector_id]
        ee_euler = self._quat_to_euler(ee_tf[3:7])
        return np.concatenate([ee_tf[:3], ee_euler, [self.gripper_pos]]).astype(np.float32)

    def _append_current_observation(self):
        self._obs_history_states.append(self._current_agent_state())
        self._obs_history_clouds.append(self._latest_point_cloud.copy())

    def _query_policy(self) -> np.ndarray:
        obs_dict = {
            "point_cloud": torch.from_numpy(_stack_history(self._obs_history_clouds, self.n_obs_steps))
            .unsqueeze(0)
            .to(self.policy_device),
            "agent_pos": torch.from_numpy(_stack_history(self._obs_history_states, self.n_obs_steps))
            .unsqueeze(0)
            .to(self.policy_device),
        }

        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)

        action_chunk = action_dict["action"].detach().to("cpu").numpy().squeeze(0)
        return np.atleast_2d(action_chunk).astype(np.float32)

    def _save_dataset(self):
        """Flush recorded frames to DATASET_PATH zarr store (appends across multiple runs)."""
        n = len(self._record_actions)
        if n == 0:
            return

        actions = np.stack(self._record_actions)   # (N, 7)
        states  = np.stack(self._record_states)    # (N, 7)
        clouds  = np.stack(self._record_clouds)    # (N, 1024, 6)

        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        store = zarr.open_group(DATASET_PATH, mode="a")
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

        if "data" not in store:
            data_grp = store.create_group("data")
            meta_grp = store.create_group("meta")
            data_grp.create_dataset("action",      data=actions, chunks=(100, 7),       dtype="float32", compressor=compressor)
            data_grp.create_dataset("state",       data=states,  chunks=(100, 7),       dtype="float32", compressor=compressor)
            data_grp.create_dataset("point_cloud", data=clouds,  chunks=(100, 1024, 6), dtype="float32", compressor=compressor)
            meta_grp.create_dataset("episode_ends", data=np.array([n], dtype=np.int64), chunks=(100,),   compressor=compressor)
        else:
            data_grp = store["data"]
            meta_grp = store["meta"]
            prev_end = int(meta_grp["episode_ends"][-1])
            data_grp["action"].append(actions)
            data_grp["state"].append(states)
            data_grp["point_cloud"].append(clouds)
            meta_grp["episode_ends"].append(np.array([prev_end + n], dtype=np.int64))

        total = int(store["meta"]["episode_ends"][-1])
        n_episodes = len(store["meta"]["episode_ends"])
        print(f"Saved {n} frames → {DATASET_PATH}  ({n_episodes} episode(s), {total} frames total)")

    def step(self):
        if not self._action_queue:
            for action in self._query_policy():
                self._action_queue.append(action)

        if self._action_queue:
            action = self._action_queue.popleft()
            delta_pos, delta_rot = action[:3], action[3:6]
            gripper_cmd = float(action[6])
        else:
            delta_pos, delta_rot, gripper_cmd = np.zeros(3), np.zeros(3), 0.0

        self.apply_ee_delta(delta_pos, delta_rot)
        self.apply_gripper(gripper_cmd)
        self.simulate()
        self._sense()
        self._append_current_observation()
        self.sim_time += self.frame_dt

        if self._record:
            self._record_actions.append(np.concatenate([delta_pos, delta_rot, [gripper_cmd]]).astype(np.float32))
            self._record_states.append(self._current_agent_state())

        if self._debug_csv_writer is not None:
            self._write_debug_row(delta_pos, delta_rot, gripper_cmd)
        if self._debug_rope_csv_writer is not None:
            self._log_rope_forces()

    def _write_debug_row(self, delta_pos: np.ndarray, delta_rot: np.ndarray, gripper_cmd: float):
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

    def _close_rope_debug_log(self):
        if self._debug_rope_csv_file is not None:
            self._debug_rope_csv_file.close()
            self._debug_rope_csv_file = None

    def _log_rope_forces(self):
        """Write one CSV row per (cloth particle, contact) for the current frame."""
        n_contacts_total = int(self.contacts.soft_contact_count.numpy()[0])
        particle_q_np = self.state_0.particle_q.numpy()
        particle_f_np = self.state_0.particle_f.numpy()

        offset = self._cloth_particle_offset
        n_cloth = self.model.particle_count - offset
        n_rope = n_cloth  # variable reused below for iteration count

        per_particle_contacts: dict[int, list[tuple[str, np.ndarray]]] = {i: [] for i in range(n_rope)}

        if n_contacts_total > 0:
            c_particles = self.contacts.soft_contact_particle.numpy()[:n_contacts_total]
            c_shapes    = self.contacts.soft_contact_shape.numpy()[:n_contacts_total]
            c_normals   = self.contacts.soft_contact_normal.numpy()[:n_contacts_total]
            for ci in range(n_contacts_total):
                global_pidx = int(c_particles[ci])
                rope_pidx = global_pidx - offset
                if rope_pidx < 0 or rope_pidx >= n_rope:
                    continue
                shape_idx = int(c_shapes[ci])
                if 0 <= shape_idx < len(self._shape_body_name):
                    body_name = self._shape_body_name[shape_idx]
                else:
                    body_name = "world/ground"
                per_particle_contacts[rope_pidx].append((body_name, c_normals[ci]))

        for rope_pidx in range(n_rope):
            global_pidx = offset + rope_pidx
            pos = particle_q_np[global_pidx]
            frc = particle_f_np[global_pidx]
            fmag = float(np.linalg.norm(frc))
            contacts_for_p = per_particle_contacts[rope_pidx]
            n_c = len(contacts_for_p)

            if contacts_for_p:
                for body_name, normal in contacts_for_p:
                    self._debug_rope_csv_writer.writerow({
                        "frame": self._debug_rope_frame,
                        "sim_time": f"{self.sim_time:.6f}",
                        "particle_idx": rope_pidx,
                        "pos_x": f"{pos[0]:.6f}",
                        "pos_y": f"{pos[1]:.6f}",
                        "pos_z": f"{pos[2]:.6f}",
                        "net_force_x": f"{frc[0]:.6f}",
                        "net_force_y": f"{frc[1]:.6f}",
                        "net_force_z": f"{frc[2]:.6f}",
                        "net_force_mag": f"{fmag:.6f}",
                        "n_contacts": n_c,
                        "contact_body": body_name,
                        "contact_normal_x": f"{normal[0]:.6f}",
                        "contact_normal_y": f"{normal[1]:.6f}",
                        "contact_normal_z": f"{normal[2]:.6f}",
                    })
            else:
                # No contact this frame — still log the particle so every particle
                # appears every frame, making the CSV easy to pivot on particle_idx.
                self._debug_rope_csv_writer.writerow({
                    "frame": self._debug_rope_frame,
                    "sim_time": f"{self.sim_time:.6f}",
                    "particle_idx": rope_pidx,
                    "pos_x": f"{pos[0]:.6f}",
                    "pos_y": f"{pos[1]:.6f}",
                    "pos_z": f"{pos[2]:.6f}",
                    "net_force_x": f"{frc[0]:.6f}",
                    "net_force_y": f"{frc[1]:.6f}",
                    "net_force_z": f"{frc[2]:.6f}",
                    "net_force_mag": f"{fmag:.6f}",
                    "n_contacts": 0,
                    "contact_body": "",
                    "contact_normal_x": "",
                    "contact_normal_y": "",
                    "contact_normal_z": "",
                })

        self._debug_rope_csv_file.flush()
        self._debug_rope_frame += 1

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
            color_image=self._cam_color,
            depth_image=self._cam_depth,
        )
        wp.launch(
            _depth_to_point_cloud,
            dim=(PC_ROWS, PC_COLS),
            inputs=[self._cam_depth, self._cam_rays, self._cam_tf, PC_STRIDE_Y, PC_STRIDE_X, PC_COLS],
            outputs=[self.cam_point_cloud],
        )
        xyz_np = self.cam_point_cloud.numpy()                    # (1024, 3) float32
        color_raw = self._cam_color.numpy()[0, 0]                # (H, W) uint32, packed RGBA
        rows_idx = np.arange(PC_ROWS, dtype=np.int32) * PC_STRIDE_Y
        cols_idx = np.arange(PC_COLS, dtype=np.int32) * PC_STRIDE_X
        sampled = color_raw[np.ix_(rows_idx, cols_idx)].reshape(PC_N_POINTS)  # (1024,) uint32
        r = (sampled & 0xFF).astype(np.float32) / 255.0
        g = ((sampled >> 8) & 0xFF).astype(np.float32) / 255.0
        b = ((sampled >> 16) & 0xFF).astype(np.float32) / 255.0
        rgb_np = np.stack([r, g, b], axis=-1)                    # (1024, 3) float32
        self._latest_point_cloud = np.concatenate([xyz_np, rgb_np], axis=-1).astype(np.float32)
        if self._record:
            self._record_clouds.append(self._latest_point_cloud.copy())

    def simulate(self):
        # Compute joint velocities once per frame so the robot reaches the IK target
        # in exactly one frame_dt (kinematic tracking, always stable).
        wp.launch(
            _compute_joint_qd,
            dim=self.model.joint_dof_count,
            inputs=[self.target_joint_q, self.state_0.joint_q, self.target_joint_qd, 1.0 / self.frame_dt],
        )

        # Rebuild self-contact BVH once per frame before the substep loop.
        self.cloth_solver.rebuild_bvh(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Robot step: kinematic FK integration, no particles, no gravity.
            # Temporarily hide particles and contacts from Featherstone so it acts
            # as a pure kinematic integrator.
            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)

            # Cloth step: VBD with gravity, cloth–body and cloth–ground contacts.
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

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

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("/camera_frame", self._cam_starts, self._cam_ends, self._cam_colors)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 0.7,
        )


if __name__ == "__main__":
    import argparse

    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=2000)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained DP3 checkpoint (.ckpt).")
    parser.add_argument("--policy-device", type=str, default="cuda:0", help="Torch device used for DP3 inference.")
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the checkpoint EMA policy if present; otherwise fall back to the base model.",
    )
    parser.add_argument("--debug-cloth", action="store_true", default=False, help="Log per-particle cloth forces and contact bodies to rope_forces_debug.csv every frame")
    parser.add_argument("--record", action="store_true", default=False, help=f"Record rollout data to {DATASET_PATH} (appends across runs)")
    viewer, args = newton.examples.init(parser)

    example = Example(
        viewer,
        args,
        checkpoint=args.checkpoint,
        policy_device=args.policy_device,
        use_ema=args.use_ema,
        debug_rope=args.debug_cloth,
        record=args.record,
    )
    newton.examples.run(example, args)

    if args.record and example._record_actions:
        try:
            answer = input(f"\nSave {len(example._record_actions)} recorded frames to dataset? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer in ("y", "yes"):
            example._save_dataset()
        else:
            print("Discarding recording.")
