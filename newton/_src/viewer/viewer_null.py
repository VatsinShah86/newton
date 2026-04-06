# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import subprocess
import time as _time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton

from ..core.types import nparray, override
from .viewer import ViewerBase


class ViewerNull(ViewerBase):
    """
    A no-operation (no-op) viewer implementation for Newton.

    This class provides a minimal, non-interactive viewer that does not perform any rendering
    or visualization. It is intended for use in headless or automated worlds where
    visualization is not required. The viewer runs for a fixed number of frames and provides
    stub implementations for all logging and frame management methods.
    """

    def __init__(
        self,
        num_frames: int = 1000,
        benchmark: bool = False,
        benchmark_timeout: float | None = None,
        benchmark_start_frame: int = 3,
        video_output_path: str | None = None,
        video_width: int = 1280,
        video_height: int = 720,
        video_fps: int = 60,
        strict_recording: bool = False,
    ):
        """
        Initialize a no-op Viewer that runs for a fixed number of frames.

        Args:
            num_frames: The number of frames to run before stopping.
            benchmark: Enable benchmark timing (FPS measurement after warmup).
            benchmark_timeout: If set, stop after this many seconds of
                steady-state simulation (measured after warmup). Implicitly
                enables *benchmark*.
            benchmark_start_frame: Number of warmup frames before benchmark
                timing starts.
            video_output_path: Optional MP4 output path for offscreen recording.
            video_width: Recording width in pixels.
            video_height: Recording height in pixels.
            video_fps: Recording frame rate.
            strict_recording: Raise instead of warning if recording setup fails.
        """
        super().__init__()

        self.num_frames = num_frames
        self.frame_count = 0

        self.benchmark = benchmark or benchmark_timeout is not None
        self.benchmark_timeout = benchmark_timeout
        self.benchmark_start_frame = benchmark_start_frame
        self._bench_start_time: float | None = None
        self._bench_frames = 0
        self._bench_elapsed = 0.0
        self._video_recorder: _NullVideoRecorder | None = None

        if video_output_path is not None:
            try:
                self._video_recorder = _NullVideoRecorder(
                    output_path=video_output_path,
                    width=video_width,
                    height=video_height,
                    fps=video_fps,
                )
            except Exception as exc:
                if strict_recording:
                    raise
                warnings.warn(f"Failed to initialize null-viewer recording: {exc}", stacklevel=2)

    @override
    def set_model(self, model: newton.Model | None, max_worlds: int | None = None):
        if self._video_recorder is not None:
            self._video_recorder.set_model(model, max_worlds=max_worlds)
        super().set_model(model, max_worlds=max_worlds)

    @override
    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        if self._video_recorder is not None:
            self._video_recorder.set_camera(pos, pitch, yaw)

    @override
    def set_world_offsets(self, spacing: tuple[float, float, float] | list[float] | wp.vec3):
        super().set_world_offsets(spacing)
        if self._video_recorder is not None:
            self._video_recorder.set_world_offsets(spacing)

    @override
    def log_mesh(
        self,
        name: str,
        points: wp.array(dtype=wp.vec3),
        indices: wp.array(dtype=wp.int32) | wp.array(dtype=wp.uint32),
        normals: wp.array(dtype=wp.vec3) | None = None,
        uvs: wp.array(dtype=wp.vec2) | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
    ):
        """
        No-op implementation for logging a mesh.

        Args:
            name: Name of the mesh.
            points: Vertex positions.
            indices: Mesh indices.
            normals: Vertex normals (optional).
            uvs: Texture coordinates (optional).
            texture: Optional texture path/URL or image array.
            hidden: Whether the mesh is hidden.
            backface_culling: Whether to enable backface culling.
        """
        if self._video_recorder is not None:
            self._video_recorder.log_mesh(
                name,
                points,
                indices,
                normals=normals,
                uvs=uvs,
                texture=texture,
                hidden=hidden,
                backface_culling=backface_culling,
            )

    @override
    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array(dtype=wp.transform) | None,
        scales: wp.array(dtype=wp.vec3) | None,
        colors: wp.array(dtype=wp.vec3) | None,
        materials: wp.array(dtype=wp.vec4) | None,
        hidden: bool = False,
    ):
        """
        No-op implementation for logging mesh instances.

        Args:
            name: Name of the instance batch.
            mesh: Mesh object.
            xforms: Instance transforms.
            scales: Instance scales.
            colors: Instance colors.
            materials: Instance materials.
            hidden: Whether the instances are hidden.
        """
        if self._video_recorder is not None:
            self._video_recorder.log_instances(name, mesh, xforms, scales, colors, materials, hidden=hidden)

    @override
    def begin_frame(self, time: float):
        """
        No-op implementation for beginning a frame.

        Args:
            time: The current simulation time.
        """
        super().begin_frame(time)
        if self._video_recorder is not None:
            self._video_recorder.begin_frame(time)

    @override
    def end_frame(self):
        """
        Increment the frame count at the end of each frame.
        """
        self.frame_count += 1

        if self.benchmark:
            if self.frame_count == self.benchmark_start_frame:
                wp.synchronize()
                self._bench_start_time = _time.perf_counter()
            elif self._bench_start_time is not None:
                wp.synchronize()
                self._bench_frames = self.frame_count - self.benchmark_start_frame
                self._bench_elapsed = _time.perf_counter() - self._bench_start_time

        if self._video_recorder is not None:
            self._video_recorder.end_frame()

    @override
    def is_running(self) -> bool:
        """
        Check if the viewer should continue running.

        Returns:
            bool: True if the frame count is less than the maximum number of frames
            and the benchmark timeout (if any) has not been reached.
        """
        if self.frame_count >= self.num_frames:
            return False
        if (
            self.benchmark_timeout is not None
            and self._bench_start_time is not None
            and self._bench_elapsed >= self.benchmark_timeout
        ):
            return False
        return True

    def benchmark_result(self) -> dict[str, float | int] | None:
        """Return benchmark results, or ``None`` if benchmarking was not enabled.

        Returns:
            Dictionary with ``fps``, ``frames``, and ``elapsed`` keys,
            or ``None`` if benchmarking is not enabled.
        """
        if not self.benchmark:
            return None
        if self._bench_frames == 0 or self._bench_elapsed == 0.0:
            return {"fps": 0.0, "frames": 0, "elapsed": 0.0}
        return {
            "fps": self._bench_frames / self._bench_elapsed,
            "frames": self._bench_frames,
            "elapsed": self._bench_elapsed,
        }

    @override
    def close(self):
        """
        No-op implementation for closing the viewer.
        """
        if self._video_recorder is not None:
            self._video_recorder.close()

    @override
    def log_lines(
        self,
        name: str,
        starts: wp.array(dtype=wp.vec3) | None,
        ends: wp.array(dtype=wp.vec3) | None,
        colors: (
            wp.array(dtype=wp.vec3) | wp.array(dtype=wp.float32) | tuple[float, float, float] | list[float] | None
        ),
        width: float = 0.01,
        hidden: bool = False,
    ):
        """
        No-op implementation for logging lines.

        Args:
            name: Name of the line batch.
            starts: Line start points.
            ends: Line end points.
            colors: Line colors.
            width: Line width hint.
            hidden: Whether the lines are hidden.
        """
        if self._video_recorder is not None:
            self._video_recorder.log_lines(name, starts, ends, colors, width=width, hidden=hidden)

    @override
    def log_points(
        self,
        name: str,
        points: wp.array(dtype=wp.vec3) | None,
        radii: wp.array(dtype=wp.float32) | float | None = None,
        colors: (
            wp.array(dtype=wp.vec3) | wp.array(dtype=wp.float32) | tuple[float, float, float] | list[float] | None
        ) = None,
        hidden: bool = False,
    ):
        """
        No-op implementation for logging points.

        Args:
            name: Name of the point batch.
            points: Point positions.
            radii: Point radii.
            colors: Point colors.
            hidden: Whether the points are hidden.
        """
        if self._video_recorder is not None:
            self._video_recorder.log_points(name, points, radii=radii, colors=colors, hidden=hidden)

    @override
    def log_array(self, name: str, array: wp.array(dtype=Any) | nparray):
        """
        No-op implementation for logging a generic array.

        Args:
            name: Name of the array.
            array: The array data.
        """
        if self._video_recorder is not None:
            self._video_recorder.log_array(name, array)

    @override
    def log_scalar(self, name: str, value: int | float | bool | np.number):
        """
        No-op implementation for logging a scalar value.

        Args:
            name: Name of the scalar.
            value: The scalar value.
        """
        if self._video_recorder is not None:
            self._video_recorder.log_scalar(name, value)

    @override
    def apply_forces(self, state: newton.State):
        """
        No-op implementation for viewer-driven forces.

        Args:
            state: The current state.
        """
        del state


class _NullVideoRecorder:
    """Headless OpenGL renderer that records frames to an MP4 via ffmpeg."""

    def __init__(self, output_path: str, width: int, height: int, fps: int):
        from .viewer_gl import ViewerGL  # noqa: PLC0415

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self._process: subprocess.Popen | None = None
        self._render_viewer = ViewerGL(width=width, height=height, headless=True)
        self._gpu_frame: wp.array | None = None
        self._cpu_frame: wp.array | None = None

    def _start_process(self):
        if self._process is not None:
            return

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            str(self.output_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def set_model(self, model: newton.Model | None, max_worlds: int | None = None):
        self._render_viewer.set_model(model, max_worlds=max_worlds)

    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        self._render_viewer.set_camera(pos, pitch, yaw)

    def set_world_offsets(self, spacing: tuple[float, float, float] | list[float] | wp.vec3):
        self._render_viewer.set_world_offsets(spacing)

    def begin_frame(self, time: float):
        self._render_viewer.begin_frame(time)

    def end_frame(self):
        self._render_viewer.end_frame()
        frame = self._render_viewer.get_frame(target_image=self._get_render_buffer())
        frame_np = self._frame_to_numpy(frame)

        self._start_process()
        assert self._process is not None
        assert self._process.stdin is not None
        self._process.stdin.write(np.ascontiguousarray(frame_np).tobytes())
        self.frame_count += 1

    def close(self):
        process = self._process
        if process is not None:
            assert process.stdin is not None
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed while writing {self.output_path}: {stderr.strip()}")

        self._render_viewer.close()

    def _get_render_buffer(self) -> wp.array:
        if self._gpu_frame is None:
            self._gpu_frame = wp.empty((self.height, self.width, 3), dtype=wp.uint8, device=self._render_viewer.device)
        return self._gpu_frame

    def _frame_to_numpy(self, frame: wp.array) -> np.ndarray:
        if frame.device.is_cuda:
            if self._cpu_frame is None:
                self._cpu_frame = wp.empty(
                    (self.height, self.width, 3),
                    dtype=wp.uint8,
                    device="cpu",
                    pinned=True,
                )
            wp.copy(self._cpu_frame, frame)
            wp.synchronize()
            return self._cpu_frame.numpy()
        return frame.numpy()

    def log_mesh(
        self,
        name: str,
        points: wp.array(dtype=wp.vec3),
        indices: wp.array(dtype=wp.int32) | wp.array(dtype=wp.uint32),
        normals: wp.array(dtype=wp.vec3) | None = None,
        uvs: wp.array(dtype=wp.vec2) | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
    ):
        self._render_viewer.log_mesh(
            name,
            points,
            indices,
            normals=normals,
            uvs=uvs,
            texture=texture,
            hidden=hidden,
            backface_culling=backface_culling,
        )

    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array(dtype=wp.transform) | None,
        scales: wp.array(dtype=wp.vec3) | None,
        colors: wp.array(dtype=wp.vec3) | None,
        materials: wp.array(dtype=wp.vec4) | None,
        hidden: bool = False,
    ):
        self._render_viewer.log_instances(name, mesh, xforms, scales, colors, materials, hidden=hidden)

    def log_lines(
        self,
        name: str,
        starts: wp.array(dtype=wp.vec3) | None,
        ends: wp.array(dtype=wp.vec3) | None,
        colors: (
            wp.array(dtype=wp.vec3) | wp.array(dtype=wp.float32) | tuple[float, float, float] | list[float] | None
        ),
        width: float = 0.01,
        hidden: bool = False,
    ):
        self._render_viewer.log_lines(name, starts, ends, colors, width=width, hidden=hidden)

    def log_points(
        self,
        name: str,
        points: wp.array(dtype=wp.vec3) | None,
        radii: wp.array(dtype=wp.float32) | float | None = None,
        colors: (
            wp.array(dtype=wp.vec3) | wp.array(dtype=wp.float32) | tuple[float, float, float] | list[float] | None
        ) = None,
        hidden: bool = False,
    ):
        self._render_viewer.log_points(name, points, radii=radii, colors=colors, hidden=hidden)

    def log_array(self, name: str, array: wp.array(dtype=Any) | nparray):
        self._render_viewer.log_array(name, array)

    def log_scalar(self, name: str, value: int | float | bool | np.number):
        self._render_viewer.log_scalar(name, value)

    @override
    def apply_forces(self, state: newton.State):
        """Null backend does not apply interactive forces.

        Args:
            state: Current simulation state.
        """
        pass
