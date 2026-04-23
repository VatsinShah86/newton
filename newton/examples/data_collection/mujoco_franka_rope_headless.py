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
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")

import mujoco

if os.environ["MUJOCO_GL"].lower() == "egl":
    import mujoco.egl as mujoco_gl
else:
    import mujoco.osmesa as mujoco_gl


GRIPPER_SPEED = 0.08
GRIPPER_MAX = 0.04
GRIPPER_MIN = 0.0

ROPE_N_PARTICLES = 12
ROPE_LENGTH = 0.5 / 2.0
ROPE_RADIUS = 0.008

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

def _asset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assets"


def _source_mjcf_path() -> Path:
    return _asset_root() / "assets" / "mjcf" / "franka_description" / "robots" / "franka_panda_gripper.xml"


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
        self._robot_ctrl_slice = slice(0, len(ROBOT_JOINT_NAMES))

        self._initialize_robot_state()

        self.gl_context = mujoco_gl.GLContext(args.width, args.height)
        self.gl_context.make_current()
        self.renderer = mujoco.Renderer(self.model, width=args.width, height=args.height)
        self.camera = self._make_camera()

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
            is_finger = "finger" in joint_name
            actuator = spec.add_actuator(
                name=f"act_{joint_name}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                target=joint_name,
            )
            actuator.set_to_position(
                kp=2000.0 if not is_finger else 200.0,
                kv=200.0 if not is_finger else 20.0,
            )

    def _add_rope_chain(self, spec: mujoco.MjSpec) -> None:
        seg_len = ROPE_LENGTH / (ROPE_N_PARTICLES - 1)
        density = 200.0
        friction = [0.8, 0.01, 0.001]
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
                density=density,
                friction=friction,
                rgba=color,
            )
            parent = child

    def _initialize_robot_state(self) -> None:
        for qpos_adr, qval in zip(self._robot_qpos_adr, self.target_joint_q, strict=True):
            self.data.qpos[qpos_adr] = qval
        self.data.ctrl[self._robot_ctrl_slice] = self.target_joint_q
        mujoco.mj_forward(self.model, self.data)

    def _make_camera(self) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, camera)
        camera.azimuth = 135.0
        camera.elevation = -25.0
        camera.distance = 1.6
        camera.lookat[:] = [-0.25, -0.25, 0.2]
        return camera

    def get_action(self) -> tuple[np.ndarray, np.ndarray, float]:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float), 0.0

    def apply_ee_delta(self, delta_pos: np.ndarray, delta_rot: np.ndarray) -> None:
        del delta_pos, delta_rot
        # Placeholder for future operational-space control. The current script
        # deliberately keeps the robot target fixed so the scene can be verified
        # without added motion.

    def apply_gripper(self, gripper_cmd: float) -> None:
        gripper_pos = float(self.target_joint_q[-1])
        if gripper_cmd > 0.5:
            gripper_pos = max(GRIPPER_MIN, gripper_pos - GRIPPER_SPEED * self.frame_dt)
        else:
            gripper_pos = min(GRIPPER_MAX, gripper_pos + GRIPPER_SPEED * self.frame_dt)

        self.target_joint_q[-2:] = gripper_pos

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.data.ctrl[self._robot_ctrl_slice] = self.target_joint_q
            mujoco.mj_step(self.model, self.data)

    def step(self) -> None:
        delta_pos, delta_rot, gripper_cmd = self.get_action()
        self.apply_ee_delta(delta_pos, delta_rot)
        self.apply_gripper(gripper_cmd)
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def run(self) -> Path:
        output_path = Path(self.args.output).resolve()
        writer = VideoWriter(output_path, self.args.width, self.args.height, self.fps)

        try:
            for frame_idx in range(self.args.num_frames):
                self.step()
                writer.write(self.render())
                if (frame_idx + 1) % 60 == 0 or frame_idx + 1 == self.args.num_frames:
                    print(f"Rendered {frame_idx + 1}/{self.args.num_frames} frames")
        finally:
            writer.close()
            self.renderer.close()
            self.gl_context.free()

        return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless MuJoCo Franka + rope scene capture")
    parser.add_argument("--num-frames", type=int, default=300, help="Number of video frames to render")
    parser.add_argument("--fps", type=int, default=60, help="Video frame rate")
    parser.add_argument("--sim-substeps", type=int, default=10, help="MuJoCo simulation substeps per video frame")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().with_name("mujoco_franka_rope.mp4")),
        help="Output MP4 path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    example = Example(args)
    output_path = example.run()
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
