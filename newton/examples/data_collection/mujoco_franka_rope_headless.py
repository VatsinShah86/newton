###########################################################################
# MuJoCo Franka Rope Headless
#
# Headless MuJoCo scene that uses a dedicated Franka Panda gripper MJCF asset
# derived from the Newton teleoperation robot definition and adds a rope-like
# object with the same total length as franka_oculus_teleop.py.
#
# The robot control loop mirrors the Newton teleop structure at a high level:
# each frame reads an action, applies end-effector/gripper updates, advances
# several simulation substeps, and writes a rendered frame to an MP4. For now
# the action is always zero so the script only verifies scene creation and
# stability.
#
# The rope is modeled as a chain of capsule bodies connected by ball joints.
# This is a conservative first step for validating the MuJoCo scene with the
# explicit Franka MJCF asset before attempting a more elaborate flex-based
# rope.
#
# Command:
#   python newton/examples/data_collection/mujoco_franka_rope_headless.py
#
###########################################################################

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")

import mujoco

if os.environ["MUJOCO_GL"].lower() == "egl":
    import mujoco.egl as mujoco_gl
else:
    import mujoco.osmesa as mujoco_gl


GRIPPER_SPEED = 0.02
GRIPPER_MAX = 0.04
CAMERA_FOV_DEG = 60.0
PC_ROWS = 32
PC_COLS = 32
APPROACH_HEIGHT = 0.10
PICK_READY_HEIGHT = 0.004
LIFT_HEIGHT = 0.10
EE_POS_DELTA_PER_STEP = 0.003
POSE_REACHED_TOL = 0.006
GRIP_REACHED_TOL = 0.004
GRIP_HOLD_FRAMES = 45
STEP_TARGET_GAIN = 0.35
IK_POS_WEIGHT = 5.0
IK_ROT_WEIGHT = 4.0
IK_REG_WEIGHT = 0.02
ARM_ACTUATOR_KP = 3500.0
ARM_ACTUATOR_KV = 300.0
FINGER_ACTUATOR_KP = 700.0
FINGER_ACTUATOR_KV = 48.0
FINGER_FORCE_LIMIT = 60.0

ROPE_N_PARTICLES = 12
ROPE_LENGTH = 0.5 / 2.0
ROPE_RADIUS = 0.008
GRIPPER_MIN = ROPE_RADIUS - 0.003

ROBOT_BASE_POS = np.array([-0.5, -0.5, -0.1], dtype=float)
ROPE_START = np.array([-0.2, -0.3, ROPE_RADIUS + 0.002], dtype=float)

ARM_REST_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853], dtype=float)
ROBOT_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]
WRIST_CAMERA_NAME = "wrist_camera"
FINGER_BOX_BASE_GAP = 0.01
GRIPPER_VISUAL_GEOM_NAMES = (
    "panda_leftfinger_visual",
    "panda_rightfinger_visual",
)
GRIPPER_COLLISION_GEOM_NAMES = (
    "panda_leftfinger_collision",
    "panda_rightfinger_collision",
)

def _asset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assets"


def _source_mjcf_path() -> Path:
    return _asset_root() / "assets" / "mjcf" / "franka_description" / "robots" / "franka_panda_gripper.xml"


def _clip_vec(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm < 1.0e-12:
        return vec
    return vec * (max_norm / norm)


def _rotation_error(desired_rot: np.ndarray, current_rot: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(current_rot[:, 0], desired_rot[:, 0])
        + np.cross(current_rot[:, 1], desired_rot[:, 1])
        + np.cross(current_rot[:, 2], desired_rot[:, 2])
    )


class VideoWriter:
    def __init__(self, output_path: Path, width: int, height: int, fps: int):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg was not found on PATH. It is required to write MP4 output.")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        stderr = b""
        if self._proc.stderr is not None:
            stderr = self._proc.stderr.read()
            self._proc.stderr.close()
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed with code {rc}: {stderr.decode(errors='replace')}")


class Example:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.spec = self._build_spec()
        self.model = self.spec.compile()
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)

        self.target_joint_q = np.concatenate([ARM_REST_Q, [GRIPPER_MAX, GRIPPER_MAX]]).astype(float)
        self._robot_qpos_adr = np.array(
            [self.model.jnt_qposadr[self.model.joint(name).id] for name in ROBOT_JOINT_NAMES],
            dtype=int,
        )
        self._arm_joint_names = ROBOT_JOINT_NAMES[:7]
        self._finger_joint_names = ROBOT_JOINT_NAMES[7:]
        self._arm_joint_ids = np.array([self.model.joint(name).id for name in self._arm_joint_names], dtype=int)
        self._arm_qpos_adr = self._robot_qpos_adr[:7].copy()
        self._finger_qpos_adr = self._robot_qpos_adr[7:].copy()
        self._arm_joint_limits = self.model.jnt_range[self._arm_joint_ids].copy()
        self._robot_dof_adr = np.array(
            [self.model.jnt_dofadr[self.model.joint(name).id] for name in ROBOT_JOINT_NAMES],
            dtype=int,
        )
        self._arm_dof_adr = self._robot_dof_adr[:7].copy()
        self._finger_dof_adr = self._robot_dof_adr[7:].copy()
        self._arm_actuator_ids = np.array([self.model.actuator(f"act_{name}").id for name in self._arm_joint_names], dtype=int)
        self._finger_actuator_id = self.model.actuator("act_panda_finger_joint1").id
        self._wrist_camera_id = self.model.camera(WRIST_CAMERA_NAME).id
        self._grip_site_id = self.model.site("panda_grip_site").id
        self._ground_geom_id = self.model.geom("ground").id
        self._left_finger_collision_geom_id = self.model.geom("panda_leftfinger_collision").id
        self._right_finger_collision_geom_id = self.model.geom("panda_rightfinger_collision").id
        self._rope_geom_ids = np.array([self.model.geom(f"rope_geom_{i}").id for i in range(ROPE_N_PARTICLES - 1)], dtype=int)
        self._rope_geom_id_set = {int(geom_id) for geom_id in self._rope_geom_ids.tolist()}
        self._rope_root_geom_id = int(self._rope_geom_ids[0])
        self._rope_mid_geom_id = int(self._rope_geom_ids[len(self._rope_geom_ids) // 2])
        self._rope_tip_geom_id = int(self._rope_geom_ids[-1])
        self._first_cloud_saved = False
        self._last_phase_name = ""
        self._last_gripper_cmd = 0.0
        self._scene_option: mujoco.MjvOption | None = None
        self._debug_prev_positions: dict[str, np.ndarray] = {}

        self._initialize_robot_state()
        self._initialize_scripted_teleop()
        self._configure_gripper_debug_rendering()

        self.gl_context = mujoco_gl.GLContext(args.width, args.height)
        self.gl_context.make_current()
        self.renderer = mujoco.Renderer(self.model, width=args.width, height=args.height)
        self.camera = self._make_camera(args.camera)

    def _build_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(_source_mjcf_path()))

        robot_root = spec.worldbody.bodies[0]
        robot_root.pos = ROBOT_BASE_POS.tolist()

        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0.0, 0.0, 0.0],
            size=[2.0, 2.0, 0.1],
            friction=[0.8, 0.005, 0.0001],
            rgba=[0.2, 0.2, 0.2, 1.0],
        )

        self._add_position_actuators(spec)
        self._add_rope_chain(spec)
        return spec

    def _add_position_actuators(self, spec: mujoco.MjSpec) -> None:
        for joint_name in ROBOT_JOINT_NAMES:
            if joint_name == "panda_finger_joint2":
                continue
            is_finger = "finger" in joint_name
            actuator_args = {
                "name": f"act_{joint_name}",
                "trntype": mujoco.mjtTrn.mjTRN_JOINT,
                "target": joint_name,
            }
            if is_finger:
                actuator_args["forcerange"] = [-FINGER_FORCE_LIMIT, FINGER_FORCE_LIMIT]
            actuator = spec.add_actuator(**actuator_args)
            actuator.set_to_position(
                kp=ARM_ACTUATOR_KP if not is_finger else FINGER_ACTUATOR_KP,
                kv=ARM_ACTUATOR_KV if not is_finger else FINGER_ACTUATOR_KV,
            )

    def _add_rope_chain(self, spec: mujoco.MjSpec) -> None:
        seg_len = ROPE_LENGTH / (ROPE_N_PARTICLES - 1)
        density = 200.0
        friction = [1.5, 0.03, 0.01]
        color = [0.45, 0.27, 0.08, 1.0]

        parent = spec.worldbody.add_body(name="rope_0", pos=ROPE_START.tolist())
        parent.add_joint(
            name="rope_free",
            type=mujoco.mjtJoint.mjJNT_FREE,
            damping=0.01,
            armature=1.0e-5,
        )
        parent.add_geom(
            name="rope_geom_0",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.0, 0.0, 0.0, seg_len, 0.0, 0.0],
            size=[ROPE_RADIUS],
            condim=6,
            density=density,
            friction=friction,
            rgba=color,
        )

        for i in range(1, ROPE_N_PARTICLES - 1):
            child = parent.add_body(name=f"rope_{i}", pos=[seg_len, 0.0, 0.0])
            child.add_joint(
                name=f"rope_ball_{i}",
                type=mujoco.mjtJoint.mjJNT_BALL,
                damping=0.02,
                armature=1.0e-5,
            )
            child.add_geom(
                name=f"rope_geom_{i}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0.0, 0.0, 0.0, seg_len, 0.0, 0.0],
                size=[ROPE_RADIUS],
                condim=6,
                density=density,
                friction=friction,
                rgba=color,
            )
            parent = child

    def _initialize_robot_state(self) -> None:
        for qpos_adr, qval in zip(self._robot_qpos_adr, self.target_joint_q, strict=True):
            self.data.qpos[qpos_adr] = qval
        self._set_ctrl_targets()
        mujoco.mj_forward(self.model, self.data)

    def _set_ctrl_targets(self, arm_q: np.ndarray | None = None, finger_q: float | None = None) -> None:
        if arm_q is None:
            arm_q = self.target_joint_q[:7]
        if finger_q is None:
            finger_q = float(self.target_joint_q[-2])
        self.data.ctrl[self._arm_actuator_ids] = np.asarray(arm_q, dtype=float)
        self.data.ctrl[self._finger_actuator_id] = float(finger_q)

    def _ee_site_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._grip_site_id].astype(np.float64, copy=True)

    def _ee_site_rot(self) -> np.ndarray:
        return self.data.site_xmat[self._grip_site_id].reshape(3, 3).astype(np.float64, copy=True)

    def _initialize_scripted_teleop(self) -> None:
        grip_site_pos = self._ee_site_pos()
        grip_site_rot = self._ee_site_rot()
        rope_center = ROPE_START + np.array([0.5 * ROPE_LENGTH, 0.0, 0.0], dtype=float)

        self.ee_target_pos = grip_site_pos.copy()
        self.ee_target_rot = grip_site_rot.copy()
        self._scripted_steps = [
            {
                "name": "approach",
                "target_pos": rope_center + np.array([0.0, 0.0, APPROACH_HEIGHT], dtype=float),
                "gripper_cmd": 0.0,
            },
            {
                "name": "pick_ready",
                "target_pos": rope_center + np.array([0.0, 0.0, PICK_READY_HEIGHT], dtype=float),
                "gripper_cmd": 0.0,
            },
            {
                "name": "grip",
                "target_pos": rope_center + np.array([0.0, 0.0, PICK_READY_HEIGHT], dtype=float),
                "gripper_cmd": 1.0,
            },
            {
                "name": "lift",
                "target_pos": rope_center + np.array([0.0, 0.0, PICK_READY_HEIGHT + LIFT_HEIGHT], dtype=float),
                "gripper_cmd": 1.0,
            },
        ]
        self._scripted_step_idx = 0
        self._scripted_step_hold = 0
        print(f"Scripted teleop phase: {self._scripted_steps[self._scripted_step_idx]['name']}")

    def _solve_arm_ik(self, target_pos: np.ndarray, target_rot: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
        saved_qpos = self.data.qpos.copy()
        saved_ctrl = self.data.ctrl.copy()

        def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            self.data.qpos[self._arm_qpos_adr] = q
            self._set_ctrl_targets(arm_q=q)
            mujoco.mj_forward(self.model, self.data)
            pos = self._ee_site_pos()
            rot = self._ee_site_rot()
            return pos, rot

        def residual(q: np.ndarray) -> np.ndarray:
            pos, rot = fk(q)
            pos_err = IK_POS_WEIGHT * (pos - target_pos)
            rot_err = IK_ROT_WEIGHT * _rotation_error(target_rot, rot)
            reg_err = IK_REG_WEIGHT * (q - q_seed)
            return np.concatenate([pos_err, rot_err, reg_err], axis=0)

        result = least_squares(
            residual,
            np.asarray(q_seed, dtype=np.float64),
            bounds=(self._arm_joint_limits[:, 0], self._arm_joint_limits[:, 1]),
            max_nfev=100,
            xtol=1.0e-8,
            ftol=1.0e-8,
            gtol=1.0e-8,
        )
        pos, rot = fk(result.x)
        pos_err = float(np.linalg.norm(pos - target_pos))
        rot_err = float(np.linalg.norm(_rotation_error(target_rot, rot)))
        if (not result.success) or pos_err > 2.0e-3 or rot_err > 2.0e-3:
            raise RuntimeError(
                f"Failed to solve scripted IK for target {target_pos}: "
                f"position error {pos_err:.6f}, rotation error {rot_err:.6f}"
            )

        self.data.qpos[:] = saved_qpos
        self.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(self.model, self.data)
        return result.x.astype(np.float64, copy=False)

    def _make_camera(self, camera_name: str) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        if camera_name == "free":
            mujoco.mjv_defaultFreeCamera(self.model, camera)
            camera.azimuth = 135.0
            camera.elevation = -25.0
            camera.distance = 1.6
            camera.lookat[:] = [-0.25, -0.25, 0.2]
            return camera

        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = self.model.camera(WRIST_CAMERA_NAME).id
        return camera

    def _configure_gripper_debug_rendering(self) -> None:
        if not self.args.render_gripper_collision:
            return

        self._scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self._scene_option)
        self._scene_option.geomgroup[:] = 1

        # Fade out the decorative finger meshes and render the actual collision
        # boxes so contact alignment is visible in the output video.
        for geom_name in GRIPPER_VISUAL_GEOM_NAMES:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                self.model.geom_rgba[geom_id] = np.array([0.95, 0.45, 0.08, 0.0], dtype=np.float32)

        debug_colors = (
            np.array([0.1, 0.9, 0.1, 1.0], dtype=np.float32),
            np.array([0.1, 0.6, 1.0, 1.0], dtype=np.float32),
        )
        for geom_name, rgba in zip(GRIPPER_COLLISION_GEOM_NAMES, debug_colors, strict=True):
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                self.model.geom_rgba[geom_id] = rgba

    def _update_renderer_scene(self, camera: str | mujoco.MjvCamera) -> None:
        if self._scene_option is None:
            self.renderer.update_scene(self.data, camera=camera)
            return
        self.renderer.update_scene(self.data, camera=camera, scene_option=self._scene_option)

    def _compute_camera_rays(self, width: int, height: int, fovy_deg: float) -> np.ndarray:
        fovy_rad = np.deg2rad(fovy_deg)
        half_height = np.tan(0.5 * fovy_rad)
        aspect = float(width) / float(height)

        px = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
        py = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
        u = px * 2.0 - 1.0
        v = py * 2.0 - 1.0

        x_cam = u[None, :] * half_height * aspect
        y_cam = -v[:, None] * half_height
        z_cam = -np.ones((height, width), dtype=np.float32)
        rays = np.stack(
            [
                np.broadcast_to(x_cam, (height, width)),
                np.broadcast_to(y_cam, (height, width)),
                z_cam,
            ],
            axis=-1,
        )
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays / np.maximum(norms, 1.0e-8)

    def _capture_wrist_point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        self.renderer.enable_depth_rendering()
        self._update_renderer_scene(WRIST_CAMERA_NAME)
        depth_m = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        self._update_renderer_scene(WRIST_CAMERA_NAME)
        color_rgb = self.renderer.render().copy()

        valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)

        fovy_deg = float(self.model.cam_fovy[self._wrist_camera_id] or CAMERA_FOV_DEG)
        ray_dirs_camera = self._compute_camera_rays(self.args.width, self.args.height, fovy_deg)
        cam_rot = self.data.cam_xmat[self._wrist_camera_id].reshape(3, 3)
        cam_pos = self.data.cam_xpos[self._wrist_camera_id].astype(np.float32, copy=True)

        # MuJoCo depth is planar depth along the optical axis, not Euclidean
        # distance along the normalized camera ray. Convert it to range first.
        ray_depth_scale = -ray_dirs_camera[..., 2]
        valid_mask &= ray_depth_scale > 1.0e-8
        ray_range = np.zeros_like(depth_m, dtype=np.float32)
        ray_range[valid_mask] = depth_m[valid_mask] / ray_depth_scale[valid_mask]

        ray_dirs_world = ray_dirs_camera @ cam_rot.T
        points_world = cam_pos[None, None, :] + ray_range[..., None] * ray_dirs_world
        points_world[~valid_mask] = 0.0

        row_idx = np.linspace(0, self.args.height - 1, PC_ROWS, dtype=np.int32)
        col_idx = np.linspace(0, self.args.width - 1, PC_COLS, dtype=np.int32)
        sampled_xyz = points_world[np.ix_(row_idx, col_idx)].reshape(-1, 3).astype(np.float32, copy=False)
        sampled_rgb = (color_rgb[np.ix_(row_idx, col_idx)].reshape(-1, 3).astype(np.float32, copy=False) / 255.0)
        sampled_cloud = np.concatenate([sampled_xyz, sampled_rgb], axis=-1)
        return sampled_cloud, depth_m.astype(np.float32, copy=False)

    def _save_first_point_cloud(self, output_path: Path) -> None:
        if self._first_cloud_saved:
            return
        cloud, depth_m = self._capture_wrist_point_cloud()
        prefix = output_path.with_suffix("")
        cloud_path = prefix.with_name(f"{prefix.name}_first_point_cloud.npy")
        depth_path = prefix.with_name(f"{prefix.name}_first_depth.npy")
        np.save(cloud_path, cloud)
        np.save(depth_path, depth_m)
        valid_points = int((cloud[:, :3] != 0.0).any(axis=1).sum())
        print(f"Saved first point cloud: {cloud.shape}, {valid_points} valid points -> {cloud_path}")
        print(f"Saved first depth map: {depth_m.shape} -> {depth_path}")
        self._first_cloud_saved = True

    def _advance_scripted_step(self) -> None:
        if self._scripted_step_idx >= len(self._scripted_steps) - 1:
            return
        self._scripted_step_idx += 1
        self._scripted_step_hold = 0
        print(f"Scripted teleop phase: {self._scripted_steps[self._scripted_step_idx]['name']}")

    def get_action(self) -> tuple[np.ndarray, np.ndarray, float]:
        step = self._scripted_steps[self._scripted_step_idx]
        self._last_phase_name = str(step["name"])
        current_site_pos = self._ee_site_pos()
        pos_error = step["target_pos"] - current_site_pos
        delta_pos = _clip_vec(STEP_TARGET_GAIN * pos_error.astype(float, copy=False), EE_POS_DELTA_PER_STEP)
        delta_rot = np.zeros(3, dtype=float)
        gripper_cmd = float(step["gripper_cmd"])
        self._last_gripper_cmd = gripper_cmd

        pos_reached = float(np.linalg.norm(pos_error)) < POSE_REACHED_TOL
        grip_reached = abs(float(self.target_joint_q[-1]) - GRIPPER_MIN) < GRIP_REACHED_TOL

        if step["name"] in {"approach", "pick_ready"}:
            if pos_reached:
                self._advance_scripted_step()
        elif step["name"] == "grip":
            delta_pos[:] = 0.0
            if pos_reached and grip_reached:
                self._scripted_step_hold += 1
                if self._scripted_step_hold >= GRIP_HOLD_FRAMES:
                    self._advance_scripted_step()
        elif step["name"] == "lift" and pos_reached:
            delta_pos[:] = 0.0

        return delta_pos, delta_rot, gripper_cmd

    def apply_ee_delta(self, delta_pos: np.ndarray, delta_rot: np.ndarray) -> None:
        self.ee_target_pos = self.ee_target_pos + delta_pos
        if float(np.linalg.norm(delta_rot)) > 1.0e-12:
            raise ValueError("Scripted teleop should not generate rotational deltas")

        arm_q = self.data.qpos[self._arm_qpos_adr].astype(np.float64, copy=True)
        solved_q = self._solve_arm_ik(self.ee_target_pos, self.ee_target_rot, arm_q)
        self.target_joint_q[:7] = np.clip(solved_q, self._arm_joint_limits[:, 0], self._arm_joint_limits[:, 1])

    def apply_gripper(self, gripper_cmd: float) -> None:
        gripper_pos = float(self.target_joint_q[-1])
        if gripper_cmd > 0.5:
            gripper_pos = max(GRIPPER_MIN, gripper_pos - GRIPPER_SPEED * self.frame_dt)
        else:
            gripper_pos = min(GRIPPER_MAX, gripper_pos + GRIPPER_SPEED * self.frame_dt)

        self.target_joint_q[-2:] = gripper_pos

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            # Track the IK-generated joint targets through MuJoCo actuators so
            # arm-rope contact remains fully dynamic.
            self._set_ctrl_targets()
            mujoco.mj_step(self.model, self.data)

    def step(self) -> None:
        delta_pos, delta_rot, gripper_cmd = self.get_action()
        self.apply_ee_delta(delta_pos, delta_rot)
        self.apply_gripper(gripper_cmd)
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> np.ndarray:
        self._update_renderer_scene(self.camera)
        return self.renderer.render()

    def _geom_world_pos(self, geom_id: int) -> np.ndarray:
        return self.data.geom_xpos[geom_id].astype(np.float64, copy=True)

    def _finite_difference_velocity(self, key: str, position: np.ndarray) -> np.ndarray:
        prev = self._debug_prev_positions.get(key)
        self._debug_prev_positions[key] = position.copy()
        if prev is None or self.frame_dt <= 0.0:
            return np.zeros(3, dtype=np.float64)
        return (position - prev) / self.frame_dt

    def _contact_debug_fieldnames(self) -> list[str]:
        return [
            "frame",
            "sim_time",
            "contact_id",
            "geom1_id",
            "geom1_name",
            "geom2_id",
            "geom2_name",
            "is_left_finger",
            "is_right_finger",
            "is_rope",
            "is_ground",
            "is_finger_rope",
            "is_rope_ground",
            "is_finger_ground",
            "rope_is_geom1",
            "rope_is_geom2",
            "left_is_geom1",
            "left_is_geom2",
            "right_is_geom1",
            "right_is_geom2",
            "body1_id",
            "body1_name",
            "body2_id",
            "body2_name",
            "dist",
            "pos_x",
            "pos_y",
            "pos_z",
            "normal_x",
            "normal_y",
            "normal_z",
            "tangent1_x",
            "tangent1_y",
            "tangent1_z",
            "tangent2_x",
            "tangent2_y",
            "tangent2_z",
            "fn_local",
            "ft1_local",
            "ft2_local",
            "torque_n_local",
            "torque_t1_local",
            "torque_t2_local",
            "tangent_force_mag",
            "force_world_x_raw",
            "force_world_y_raw",
            "force_world_z_raw",
            "torque_world_x_raw",
            "torque_world_y_raw",
            "torque_world_z_raw",
            "body1_point_vel_x",
            "body1_point_vel_y",
            "body1_point_vel_z",
            "body2_point_vel_x",
            "body2_point_vel_y",
            "body2_point_vel_z",
            "rel_point_vel_x",
            "rel_point_vel_y",
            "rel_point_vel_z",
            "rel_normal_vel",
            "rel_tangent1_vel",
            "rel_tangent2_vel",
            "rel_tangent_speed",
        ]

    def _point_velocity_world(self, body_id: int, point_world: np.ndarray) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jac(self.model, self.data, jacp, None, point_world, body_id)
        return jacp @ self.data.qvel

    def _collect_contact_debug(
        self,
        frame_idx: int,
    ) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
        summary: dict[str, float | int] = {
            "ncon_total": int(self.data.ncon),
            "ncon_finger_rope_total": 0,
            "ncon_finger_rope_left": 0,
            "ncon_finger_rope_right": 0,
            "ncon_rope_ground_total": 0,
            "ncon_finger_ground_total": 0,
            "finger_rope_normal_force_sum": 0.0,
            "finger_rope_normal_force_sum_left": 0.0,
            "finger_rope_normal_force_sum_right": 0.0,
            "finger_rope_tangent_force_sum": 0.0,
            "finger_rope_tangent_force_sum_left": 0.0,
            "finger_rope_tangent_force_sum_right": 0.0,
            "rope_ground_normal_force_sum": 0.0,
            "rope_ground_tangent_force_sum": 0.0,
            "finger_rope_min_dist": np.inf,
            "finger_rope_min_dist_left": np.inf,
            "finger_rope_min_dist_right": np.inf,
            "rope_ground_min_dist": np.inf,
            "finger_rope_mean_pos_x": 0.0,
            "finger_rope_mean_pos_y": 0.0,
            "finger_rope_mean_pos_z": 0.0,
            "finger_rope_mean_normal_x": 0.0,
            "finger_rope_mean_normal_y": 0.0,
            "finger_rope_mean_normal_z": 0.0,
        }
        finger_rope_weight = 0.0
        contact_rows: list[dict[str, float | int | str]] = []

        for contact_id in range(int(self.data.ncon)):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            geom1_name = self.model.geom(geom1).name or ""
            geom2_name = self.model.geom(geom2).name or ""
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            body1_name = self.model.body(body1).name or ""
            body2_name = self.model.body(body2).name or ""

            is_left_finger = geom1 == self._left_finger_collision_geom_id or geom2 == self._left_finger_collision_geom_id
            is_right_finger = geom1 == self._right_finger_collision_geom_id or geom2 == self._right_finger_collision_geom_id
            is_rope = geom1 in self._rope_geom_id_set or geom2 in self._rope_geom_id_set
            is_ground = geom1 == self._ground_geom_id or geom2 == self._ground_geom_id
            is_finger = is_left_finger or is_right_finger
            is_finger_rope = is_finger and is_rope
            is_rope_ground = is_rope and is_ground
            is_finger_ground = is_finger and is_ground

            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_id, contact_force)
            frame_axes = np.asarray(contact.frame, dtype=np.float64).reshape(-1)
            normal = frame_axes[0:3].copy()
            tangent1 = frame_axes[3:6].copy()
            tangent2 = frame_axes[6:9].copy()
            force_world = normal * contact_force[0] + tangent1 * contact_force[1] + tangent2 * contact_force[2]
            torque_world = normal * contact_force[3] + tangent1 * contact_force[4] + tangent2 * contact_force[5]
            tangent_force_mag = float(np.linalg.norm(contact_force[1:3]))
            body1_point_vel = self._point_velocity_world(body1, np.asarray(contact.pos, dtype=np.float64))
            body2_point_vel = self._point_velocity_world(body2, np.asarray(contact.pos, dtype=np.float64))
            rel_point_vel = body2_point_vel - body1_point_vel
            rel_normal_vel = float(np.dot(rel_point_vel, normal))
            rel_tangent1_vel = float(np.dot(rel_point_vel, tangent1))
            rel_tangent2_vel = float(np.dot(rel_point_vel, tangent2))
            rel_tangent_speed = float(np.linalg.norm([rel_tangent1_vel, rel_tangent2_vel]))

            if is_finger_rope:
                summary["ncon_finger_rope_total"] = int(summary["ncon_finger_rope_total"]) + 1
                summary["finger_rope_normal_force_sum"] = float(summary["finger_rope_normal_force_sum"]) + float(contact_force[0])
                summary["finger_rope_tangent_force_sum"] = float(summary["finger_rope_tangent_force_sum"]) + tangent_force_mag
                summary["finger_rope_min_dist"] = min(float(summary["finger_rope_min_dist"]), float(contact.dist))
                weight = max(abs(float(contact_force[0])), 1.0e-8)
                summary["finger_rope_mean_pos_x"] = float(summary["finger_rope_mean_pos_x"]) + weight * float(contact.pos[0])
                summary["finger_rope_mean_pos_y"] = float(summary["finger_rope_mean_pos_y"]) + weight * float(contact.pos[1])
                summary["finger_rope_mean_pos_z"] = float(summary["finger_rope_mean_pos_z"]) + weight * float(contact.pos[2])
                summary["finger_rope_mean_normal_x"] = float(summary["finger_rope_mean_normal_x"]) + weight * float(normal[0])
                summary["finger_rope_mean_normal_y"] = float(summary["finger_rope_mean_normal_y"]) + weight * float(normal[1])
                summary["finger_rope_mean_normal_z"] = float(summary["finger_rope_mean_normal_z"]) + weight * float(normal[2])
                finger_rope_weight += weight
                if is_left_finger:
                    summary["ncon_finger_rope_left"] = int(summary["ncon_finger_rope_left"]) + 1
                    summary["finger_rope_normal_force_sum_left"] = float(summary["finger_rope_normal_force_sum_left"]) + float(contact_force[0])
                    summary["finger_rope_tangent_force_sum_left"] = (
                        float(summary["finger_rope_tangent_force_sum_left"]) + tangent_force_mag
                    )
                    summary["finger_rope_min_dist_left"] = min(float(summary["finger_rope_min_dist_left"]), float(contact.dist))
                if is_right_finger:
                    summary["ncon_finger_rope_right"] = int(summary["ncon_finger_rope_right"]) + 1
                    summary["finger_rope_normal_force_sum_right"] = float(summary["finger_rope_normal_force_sum_right"]) + float(contact_force[0])
                    summary["finger_rope_tangent_force_sum_right"] = (
                        float(summary["finger_rope_tangent_force_sum_right"]) + tangent_force_mag
                    )
                    summary["finger_rope_min_dist_right"] = min(float(summary["finger_rope_min_dist_right"]), float(contact.dist))

            if is_rope_ground:
                summary["ncon_rope_ground_total"] = int(summary["ncon_rope_ground_total"]) + 1
                summary["rope_ground_normal_force_sum"] = float(summary["rope_ground_normal_force_sum"]) + float(contact_force[0])
                summary["rope_ground_tangent_force_sum"] = float(summary["rope_ground_tangent_force_sum"]) + tangent_force_mag
                summary["rope_ground_min_dist"] = min(float(summary["rope_ground_min_dist"]), float(contact.dist))

            if is_finger_ground:
                summary["ncon_finger_ground_total"] = int(summary["ncon_finger_ground_total"]) + 1

            contact_rows.append(
                {
                    "frame": frame_idx,
                    "sim_time": float(self.sim_time),
                    "contact_id": contact_id,
                    "geom1_id": geom1,
                    "geom1_name": geom1_name,
                    "geom2_id": geom2,
                    "geom2_name": geom2_name,
                    "is_left_finger": int(is_left_finger),
                    "is_right_finger": int(is_right_finger),
                    "is_rope": int(is_rope),
                    "is_ground": int(is_ground),
                    "is_finger_rope": int(is_finger_rope),
                    "is_rope_ground": int(is_rope_ground),
                    "is_finger_ground": int(is_finger_ground),
                    "rope_is_geom1": int(geom1 in self._rope_geom_id_set),
                    "rope_is_geom2": int(geom2 in self._rope_geom_id_set),
                    "left_is_geom1": int(geom1 == self._left_finger_collision_geom_id),
                    "left_is_geom2": int(geom2 == self._left_finger_collision_geom_id),
                    "right_is_geom1": int(geom1 == self._right_finger_collision_geom_id),
                    "right_is_geom2": int(geom2 == self._right_finger_collision_geom_id),
                    "body1_id": body1,
                    "body1_name": body1_name,
                    "body2_id": body2,
                    "body2_name": body2_name,
                    "dist": float(contact.dist),
                    "pos_x": float(contact.pos[0]),
                    "pos_y": float(contact.pos[1]),
                    "pos_z": float(contact.pos[2]),
                    "normal_x": float(normal[0]),
                    "normal_y": float(normal[1]),
                    "normal_z": float(normal[2]),
                    "tangent1_x": float(tangent1[0]),
                    "tangent1_y": float(tangent1[1]),
                    "tangent1_z": float(tangent1[2]),
                    "tangent2_x": float(tangent2[0]),
                    "tangent2_y": float(tangent2[1]),
                    "tangent2_z": float(tangent2[2]),
                    "fn_local": float(contact_force[0]),
                    "ft1_local": float(contact_force[1]),
                    "ft2_local": float(contact_force[2]),
                    "torque_n_local": float(contact_force[3]),
                    "torque_t1_local": float(contact_force[4]),
                    "torque_t2_local": float(contact_force[5]),
                    "tangent_force_mag": tangent_force_mag,
                    "force_world_x_raw": float(force_world[0]),
                    "force_world_y_raw": float(force_world[1]),
                    "force_world_z_raw": float(force_world[2]),
                    "torque_world_x_raw": float(torque_world[0]),
                    "torque_world_y_raw": float(torque_world[1]),
                    "torque_world_z_raw": float(torque_world[2]),
                    "body1_point_vel_x": float(body1_point_vel[0]),
                    "body1_point_vel_y": float(body1_point_vel[1]),
                    "body1_point_vel_z": float(body1_point_vel[2]),
                    "body2_point_vel_x": float(body2_point_vel[0]),
                    "body2_point_vel_y": float(body2_point_vel[1]),
                    "body2_point_vel_z": float(body2_point_vel[2]),
                    "rel_point_vel_x": float(rel_point_vel[0]),
                    "rel_point_vel_y": float(rel_point_vel[1]),
                    "rel_point_vel_z": float(rel_point_vel[2]),
                    "rel_normal_vel": rel_normal_vel,
                    "rel_tangent1_vel": rel_tangent1_vel,
                    "rel_tangent2_vel": rel_tangent2_vel,
                    "rel_tangent_speed": rel_tangent_speed,
                }
            )

        if finger_rope_weight > 0.0:
            summary["finger_rope_mean_pos_x"] = float(summary["finger_rope_mean_pos_x"]) / finger_rope_weight
            summary["finger_rope_mean_pos_y"] = float(summary["finger_rope_mean_pos_y"]) / finger_rope_weight
            summary["finger_rope_mean_pos_z"] = float(summary["finger_rope_mean_pos_z"]) / finger_rope_weight
            summary["finger_rope_mean_normal_x"] = float(summary["finger_rope_mean_normal_x"]) / finger_rope_weight
            summary["finger_rope_mean_normal_y"] = float(summary["finger_rope_mean_normal_y"]) / finger_rope_weight
            summary["finger_rope_mean_normal_z"] = float(summary["finger_rope_mean_normal_z"]) / finger_rope_weight
        else:
            summary["finger_rope_mean_pos_x"] = np.nan
            summary["finger_rope_mean_pos_y"] = np.nan
            summary["finger_rope_mean_pos_z"] = np.nan
            summary["finger_rope_mean_normal_x"] = np.nan
            summary["finger_rope_mean_normal_y"] = np.nan
            summary["finger_rope_mean_normal_z"] = np.nan

        for key in (
            "finger_rope_min_dist",
            "finger_rope_min_dist_left",
            "finger_rope_min_dist_right",
            "rope_ground_min_dist",
        ):
            if not np.isfinite(float(summary[key])):
                summary[key] = np.nan

        return summary, contact_rows

    def _gripper_debug_row(
        self,
        frame_idx: int,
        contact_summary: dict[str, float | int],
    ) -> dict[str, float | int | str]:
        finger_q = self.data.qpos[self._finger_qpos_adr].astype(np.float64, copy=False)
        finger_qd = self.data.qvel[self._finger_dof_adr].astype(np.float64, copy=False)
        ee_pos = self._ee_site_pos()
        ee_vel = self._finite_difference_velocity("ee_site", ee_pos)
        rope_root_pos = self._geom_world_pos(self._rope_root_geom_id)
        rope_mid_pos = self._geom_world_pos(self._rope_mid_geom_id)
        rope_tip_pos = self._geom_world_pos(self._rope_tip_geom_id)
        rope_root_vel = self._finite_difference_velocity("rope_root", rope_root_pos)
        rope_mid_vel = self._finite_difference_velocity("rope_mid", rope_mid_pos)
        rope_tip_vel = self._finite_difference_velocity("rope_tip", rope_tip_pos)
        target_gap = FINGER_BOX_BASE_GAP + float(self.target_joint_q[-2]) + float(self.target_joint_q[-1])
        actual_gap = FINGER_BOX_BASE_GAP + float(finger_q[0]) + float(finger_q[1])
        ee_err = self.ee_target_pos - ee_pos
        return {
            "frame": frame_idx,
            "sim_time": float(self.sim_time),
            "phase": self._last_phase_name,
            "gripper_cmd": float(self._last_gripper_cmd),
            "target_left": float(self.target_joint_q[-2]),
            "target_right": float(self.target_joint_q[-1]),
            "ctrl_left": float(self.data.ctrl[self._finger_actuator_id]),
            "ctrl_right": float(self.data.ctrl[self._finger_actuator_id]),
            "qpos_left": float(finger_q[0]),
            "qpos_right": float(finger_q[1]),
            "qvel_left": float(finger_qd[0]),
            "qvel_right": float(finger_qd[1]),
            "target_gap_est": target_gap,
            "actual_gap_est": actual_gap,
            "gap_error": target_gap - actual_gap,
            "ee_pos_x": float(ee_pos[0]),
            "ee_pos_y": float(ee_pos[1]),
            "ee_pos_z": float(ee_pos[2]),
            "ee_target_x": float(self.ee_target_pos[0]),
            "ee_target_y": float(self.ee_target_pos[1]),
            "ee_target_z": float(self.ee_target_pos[2]),
            "ee_err_x": float(ee_err[0]),
            "ee_err_y": float(ee_err[1]),
            "ee_err_z": float(ee_err[2]),
            "ee_vel_x": float(ee_vel[0]),
            "ee_vel_y": float(ee_vel[1]),
            "ee_vel_z": float(ee_vel[2]),
            "rope_root_pos_x": float(rope_root_pos[0]),
            "rope_root_pos_y": float(rope_root_pos[1]),
            "rope_root_pos_z": float(rope_root_pos[2]),
            "rope_mid_pos_x": float(rope_mid_pos[0]),
            "rope_mid_pos_y": float(rope_mid_pos[1]),
            "rope_mid_pos_z": float(rope_mid_pos[2]),
            "rope_tip_pos_x": float(rope_tip_pos[0]),
            "rope_tip_pos_y": float(rope_tip_pos[1]),
            "rope_tip_pos_z": float(rope_tip_pos[2]),
            "rope_root_vel_x": float(rope_root_vel[0]),
            "rope_root_vel_y": float(rope_root_vel[1]),
            "rope_root_vel_z": float(rope_root_vel[2]),
            "rope_mid_vel_x": float(rope_mid_vel[0]),
            "rope_mid_vel_y": float(rope_mid_vel[1]),
            "rope_mid_vel_z": float(rope_mid_vel[2]),
            "rope_tip_vel_x": float(rope_tip_vel[0]),
            "rope_tip_vel_y": float(rope_tip_vel[1]),
            "rope_tip_vel_z": float(rope_tip_vel[2]),
            **contact_summary,
        }

    def run(self) -> Path:
        output_path = Path(self.args.output).resolve()
        writer = VideoWriter(output_path, self.args.width, self.args.height, self.fps)
        debug_path = output_path.with_suffix("")
        debug_path = debug_path.with_name(f"{debug_path.name}_gripper_debug.csv")
        contact_debug_path = debug_path.with_name(f"{output_path.with_suffix('').name}_contact_debug.csv")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_file = debug_path.open("w", newline="")
        contact_debug_file = contact_debug_path.open("w", newline="")
        debug_writer: csv.DictWriter | None = None
        contact_debug_writer = csv.DictWriter(contact_debug_file, fieldnames=self._contact_debug_fieldnames())
        contact_debug_writer.writeheader()

        try:
            for frame_idx in range(self.args.num_frames):
                self.step()
                contact_summary, contact_rows = self._collect_contact_debug(frame_idx)
                frame_row = self._gripper_debug_row(frame_idx, contact_summary)
                if debug_writer is None:
                    debug_writer = csv.DictWriter(debug_file, fieldnames=list(frame_row.keys()))
                    debug_writer.writeheader()
                debug_writer.writerow(frame_row)
                for contact_row in contact_rows:
                    contact_debug_writer.writerow(contact_row)
                if frame_idx == 0:
                    self._save_first_point_cloud(output_path)
                writer.write(self.render())
                if (frame_idx + 1) % 60 == 0 or frame_idx + 1 == self.args.num_frames:
                    print(f"Rendered {frame_idx + 1}/{self.args.num_frames} frames")
        finally:
            debug_file.close()
            contact_debug_file.close()
            writer.close()
            self.renderer.close()
            self.gl_context.free()

        print(f"Saved gripper debug to {debug_path}")
        print(f"Saved contact debug to {contact_debug_path}")
        return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless MuJoCo Franka + rope scene capture")
    parser.add_argument("--num-frames", type=int, default=300, help="Number of video frames to render")
    parser.add_argument("--fps", type=int, default=60, help="Video frame rate")
    parser.add_argument("--sim-substeps", type=int, default=100, help="MuJoCo simulation substeps per video frame")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument(
        "--camera",
        choices=["wrist", "free"],
        default="wrist",
        help="Render camera: wrist camera from oculus_teleop_tshirt.py or the original free viewer camera",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().with_name("mujoco_franka_rope.mp4")),
        help="Output MP4 path",
    )
    parser.add_argument(
        "--render-gripper-collision",
        action="store_true",
        help="Render fingertip collision boxes for grasp debugging",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    example = Example(args)
    output_path = example.run()
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
