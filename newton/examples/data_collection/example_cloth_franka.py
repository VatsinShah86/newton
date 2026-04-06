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

###########################################################################
# Example Cloth Franka
#
# This simulation demonstrates a coupled robot-cloth simulation
# using the VBD solver for the cloth and Featherstone for the robot,
# showcasing its ability to handle complex contacts while ensuring it
# remains intersection-free.
#
# The simulation runs in centimeter scale for better numerical behavior
# of the VBD solver. A vis_state is used to convert back to meter scale
# for visualization.
#
# Command: python -m newton.examples.data_collection cloth_franka
#
###########################################################################

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.math import transform_twist
from newton.sensors import SensorTiledCamera
from newton.solvers import SolverFeatherstone, SolverVBD

SIM_TO_WORLD_SCALE = np.float32(0.01)
WORLD_TO_SIM_SCALE = np.float32(100.0)
DEFORMABLE_TRIANGLE_MESH_SHAPE_ID = np.uint32(0xFFFFFFFD)


@dataclass(frozen=True)
class SensorCameraConfig:
    name: str
    pos_m: tuple[float, float, float]
    pitch_deg: float
    yaw_deg: float
    fov_deg: float
    width_px: int
    height_px: int


def _camera_axes_z_up(pitch_deg: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)

    front = np.array(
        [
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
        ],
        dtype=np.float32,
    )
    front /= np.linalg.norm(front)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(front, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, front)
    up /= np.linalg.norm(up)

    return right, up, front


def _camera_rotation_matrix_z_up(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    right, up, front = _camera_axes_z_up(pitch_deg, yaw_deg)
    return np.column_stack((right, up, -front)).astype(np.float32)


def _camera_quaternion_z_up(pitch_deg: float, yaw_deg: float) -> wp.quatf:
    rotation = _camera_rotation_matrix_z_up(pitch_deg, yaw_deg)
    return wp.normalize(wp.quat_from_matrix(wp.mat33f(*rotation.flatten().tolist())))


def _next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    next_index = 1
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"run_(\d+)", child.name)
        if match is None:
            continue
        next_index = max(next_index, int(match.group(1)) + 1)

    run_dir = base_dir / f"run_{next_index}"
    (run_dir / "pc").mkdir(parents=True, exist_ok=True)
    return run_dir


def _merge_point_clouds(point_clouds: list[np.ndarray]) -> np.ndarray:
    merged = [cloud.reshape(-1, 3).astype(np.float32, copy=False) for cloud in point_clouds if cloud.size]
    if not merged:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(merged, axis=0)


def _extract_point_cloud_world_m(
    depth_image_cm: np.ndarray,
    shape_index_image: np.ndarray,
    camera_rays: np.ndarray,
    camera_config: SensorCameraConfig,
) -> np.ndarray:
    valid_mask = (depth_image_cm > 0.0) & (shape_index_image == DEFORMABLE_TRIANGLE_MESH_SHAPE_ID)
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float32)

    ray_dirs_camera = camera_rays[..., 1, :][valid_mask].astype(np.float32, copy=False)
    rotation = _camera_rotation_matrix_z_up(camera_config.pitch_deg, camera_config.yaw_deg)
    ray_dirs_world = ray_dirs_camera @ rotation.T

    depth_m = depth_image_cm[valid_mask].astype(np.float32, copy=False) * SIM_TO_WORLD_SCALE
    origin_world_m = np.asarray(camera_config.pos_m, dtype=np.float32)

    return origin_world_m + ray_dirs_world * depth_m[:, None]


@wp.kernel
def scale_positions(src: wp.array(dtype=wp.vec3), scale: float, dst: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array(dtype=wp.transform), scale: float, dst: wp.array(dtype=wp.transform)):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def compute_ee_delta(
    body_q: wp.array(dtype=wp.transform),
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    # compute pose difference between end effector and target
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


def compute_body_jacobian(
    model: Model,
    joint_q: wp.array,
    joint_qd: wp.array,
    body_id: int | str,  # Can be either body index or body name
    offset: wp.transform | None = None,
    velocity: bool = True,
    include_rotation: bool = False,
):
    if isinstance(body_id, str):
        body_id = model.body_name.get(body_id)
    if offset is None:
        offset = wp.transform_identity()

    joint_q.requires_grad = True
    joint_qd.requires_grad = True

    if velocity:

        @wp.kernel
        def compute_body_out(body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)):
            mv = transform_twist(offset, body_qd[body_id])
            if wp.static(include_rotation):
                for i in range(6):
                    body_out[i] = mv[i]
            else:
                for i in range(3):
                    body_out[i] = mv[3 + i]

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3
    else:

        @wp.kernel
        def compute_body_out(body_q: wp.array(dtype=wp.transform), body_out: wp.array(dtype=float)):
            tf = body_q[body_id] * offset
            if wp.static(include_rotation):
                for i in range(7):
                    body_out[i] = tf[i]
            else:
                for i in range(3):
                    body_out[i] = tf[i]

        in_dim = model.joint_coord_count
        out_dim = 7 if include_rotation else 3

    out_state = model.state(requires_grad=True)
    body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
        eval_fk(model, joint_q, joint_qd, out_state)
        wp.launch(compute_body_out, 1, inputs=[out_state.body_qd if velocity else out_state.body_q], outputs=[body_out])

    def onehot(i):
        x = np.zeros(out_dim, dtype=np.float32)
        x[i] = 1.0
        return wp.array(x)

    J = np.empty((out_dim, in_dim), dtype=wp.float32)
    for i in range(out_dim):
        tape.backward(grads={body_out: onehot(i)})
        J[i] = joint_qd.grad.numpy() if velocity else joint_q.grad.numpy()
        tape.zero()
    return J.astype(np.float32)


class Example:
    def __init__(self, viewer, args):
        # parameters
        #   simulation (centimeter scale)
        self.add_cloth = True
        self.add_robot = True
        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # visualization: simulation in cm, viewer in meters
        self.viz_scale = 0.01

        #   contact (cm scale)
        #       body-cloth contact
        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 0.8
        #       self-contact
        self.particle_self_contact_radius = 0.2
        self.particle_self_contact_margin = 0.2

        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 1e-2

        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 1e-3
        self.robot_contact_mu = 1.5

        self.self_contact_friction = 0.25

        #   elasticity
        self.tri_ke = 1e4
        self.tri_ka = 1e4
        self.tri_kd = 1.5e-6

        self.bending_ke = 5
        self.bending_kd = 1e-2

        self.sensor_cameras = [
            SensorCameraConfig(
                name="cam_0",
                pos_m=(1.244, -0.575, 0.696),
                pitch_deg=-20,
                yaw_deg=180,
                fov_deg=45.0,
                width_px=640,
                height_px=480,
            )
        ]
        self.capture_pointcloud = True
        self.capture_frame_stride = 1
        self.capture_frame_index = 1
        # Temporary camera debug. Remove these three lines and _print_camera_debug() later.
        self.debug_camera_enabled = True
        self.debug_camera_interval = 100
        self.frame_index = 0

        self.scene = ModelBuilder(gravity=-981.0)

        self.viewer = viewer

        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)

            self.scene.add_world(franka)
            self.bodies_per_world = franka.body_count
            self.dof_q_per_world = franka.joint_coord_count
            self.dof_qd_per_world = franka.joint_dof_count

        # add a table (cm scale)
        self.table_hx_cm = 40.0
        self.table_hy_cm = 40.0
        self.table_hz_cm = 10.0
        self.table_pos_cm = wp.vec3(0.0, -50.0, 10.0)
        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            wp.transform(
                self.table_pos_cm,
                wp.quat_identity(),
            ),
            hx=self.table_hx_cm,
            hy=self.table_hy_cm,
            hz=self.table_hz_cm,
        )

        if self.add_cloth:
            cloth_size_cm = 20.0
            cloth_cells_x = 20
            cloth_cells_y = 20
            cloth_cell_x = cloth_size_cm / cloth_cells_x
            cloth_cell_y = cloth_size_cm / cloth_cells_y
            cloth_origin_x = float(self.table_pos_cm[0]) - 0.5 * cloth_size_cm
            cloth_origin_y = float(self.table_pos_cm[1]) - 0.5 * cloth_size_cm
            cloth_top_z = float(self.table_pos_cm[2]) + self.table_hz_cm + 0.5

            self.scene.add_cloth_grid(
                pos=wp.vec3(cloth_origin_x, cloth_origin_y, cloth_top_z),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=cloth_cells_x,
                dim_y=cloth_cells_y,
                cell_x=cloth_cell_x,
                cell_y=cloth_cell_y,
                mass=0.02,
                tri_ke=self.tri_ke,
                tri_ka=self.tri_ka,
                tri_kd=self.tri_kd,
                edge_ke=self.bending_ke,
                edge_kd=self.bending_kd,
                particle_radius=self.cloth_particle_radius,
            )

            self.scene.color()

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)
        self._set_up_pointcloud_capture()

        # Hide the table box from automatic shape rendering -- the GL viewer
        # bakes primitive dimensions into the mesh and ignores shape_scale,
        # so we render it manually at meter scale in render() instead.
        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # Pre-compute meter-scale table viz data
        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(self.table_pos_cm[0]) * self.viz_scale,
                        float(self.table_pos_cm[1]) * self.viz_scale,
                        float(self.table_pos_cm[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            self.table_hx_cm * self.viz_scale,
            self.table_hy_cm * self.viz_scale,
            self.table_hz_cm * self.viz_scale,
        )
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()

        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu

        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self.control = self.model.control()

        # Explicit collision pipeline for cloth-body contacts with custom margin
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.sim_time = 0.0

        # initialize robot solver
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.cloth_solver: SolverVBD | None = None
        if self.add_cloth:
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=self.iterations,
                integrate_with_external_rigid_solver=True,
                particle_self_contact_radius=self.particle_self_contact_radius,
                particle_self_contact_margin=self.particle_self_contact_margin,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.5,
                particle_enable_self_contact=True,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                particle_collision_detection_interval=-1,
                rigid_contact_k_start=self.soft_contact_ke,
            )

        self.viewer.set_model(self.model)
        primary_sensor_camera = self.sensor_cameras[0]
        self.viewer.set_camera(
            wp.vec3(*primary_sensor_camera.pos_m),
            primary_sensor_camera.pitch_deg,
            primary_sensor_camera.yaw_deg,
        )
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = primary_sensor_camera.fov_deg

        # Visualization state for meter-scale rendering
        self.viz_state = self.model.state()

        # Pre-compute scaled shape data for meter-scale visualization.
        # Two paths need updating:
        #   1) The GL viewer's CUDA path reads model.shape_transform / model.shape_scale
        #      directly, so we swap them temporarily in render().
        #   2) The base viewer path caches shapes.xforms / shapes.scales during
        #      set_model(), so we permanently scale those cached copies here.
        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        # Scale the viewer's cached shape instance data (base viewer / GL fallback path)
        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)

                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        # gravity arrays for swapping during simulation
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        # gravity in cm/s²
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # graph capture
        if self.add_cloth:
            self.capture()

    def _set_up_pointcloud_capture(self):
        self.pointcloud_base_dir = Path(__file__).resolve().parent / "data"
        self.pointcloud_run_dir = _next_run_dir(self.pointcloud_base_dir)
        self.pointcloud_dir = self.pointcloud_run_dir / "pc"

        if not self.sensor_cameras:
            raise ValueError("At least one sensor camera must be configured.")

        sensor_width = self.sensor_cameras[0].width_px
        sensor_height = self.sensor_cameras[0].height_px
        for camera in self.sensor_cameras[1:]:
            if camera.width_px != sensor_width or camera.height_px != sensor_height:
                raise ValueError("All sensor cameras must use the same width and height.")

        self.tiled_camera_sensor = SensorTiledCamera(
            model=self.model,
            config=SensorTiledCamera.Config(
                default_light=True,
                default_light_shadows=True,
                backface_culling=True,
            ),
        )
        self.sensor_camera_rays = self.tiled_camera_sensor.compute_pinhole_camera_rays(
            sensor_width,
            sensor_height,
            [math.radians(camera.fov_deg) for camera in self.sensor_cameras],
        )
        self.sensor_camera_rays_np = self.sensor_camera_rays.numpy()
        self.sensor_depth_image = self.tiled_camera_sensor.create_depth_image_output(
            sensor_width,
            sensor_height,
            camera_count=len(self.sensor_cameras),
        )
        self.sensor_shape_index_image = self.tiled_camera_sensor.create_shape_index_image_output(
            sensor_width,
            sensor_height,
            camera_count=len(self.sensor_cameras),
        )

    def _get_sensor_camera_transforms_cm(self) -> wp.array(dtype=wp.transformf):
        return wp.array(
            [
                [
                    wp.transformf(
                        wp.vec3f(*(np.asarray(camera.pos_m, dtype=np.float32) * WORLD_TO_SIM_SCALE)),
                        _camera_quaternion_z_up(camera.pitch_deg, camera.yaw_deg),
                    )
                ]
                for camera in self.sensor_cameras
            ],
            dtype=wp.transformf,
            device=self.model.device,
        )

    def capture_point_cloud(self):
        if not self.capture_pointcloud:
            return

        if (self.capture_frame_index - 1) % self.capture_frame_stride != 0:
            self.capture_frame_index += 1
            return

        self.tiled_camera_sensor.update(
            self.state_0,
            self._get_sensor_camera_transforms_cm(),
            self.sensor_camera_rays,
            depth_image=self.sensor_depth_image,
            shape_index_image=self.sensor_shape_index_image,
        )

        depth_np = self.sensor_depth_image.numpy()[0]
        shape_index_np = self.sensor_shape_index_image.numpy()[0]
        per_camera_clouds = [
            _extract_point_cloud_world_m(
                depth_np[camera_index],
                shape_index_np[camera_index],
                self.sensor_camera_rays_np[camera_index],
                camera,
            )
            for camera_index, camera in enumerate(self.sensor_cameras)
        ]
        merged_cloud = _merge_point_clouds(per_camera_clouds)

        output_path = self.pointcloud_dir / f"pc_{self.capture_frame_index}.npy"
        np.save(output_path, merged_cloud)
        self.capture_frame_index += 1

    def _print_camera_debug(self):
        if not self.debug_camera_enabled or self.frame_index % self.debug_camera_interval != 0:
            return

        primary_sensor_camera = self.sensor_cameras[0]
        print(
            f"[frame {self.frame_index}] sensor_camera pos={primary_sensor_camera.pos_m} "
            f"pitch={primary_sensor_camera.pitch_deg:.1f} yaw={primary_sensor_camera.yaw_deg:.1f} "
            f"fov={primary_sensor_camera.fov_deg:.1f}"
        )

        if hasattr(self.viewer, "camera"):
            camera = self.viewer.camera
            print(
                f"[frame {self.frame_index}] viewer_camera pos=({camera.pos.x:.3f}, {camera.pos.y:.3f}, {camera.pos.z:.3f}) "
                f"pitch={camera.pitch:.1f} yaw={camera.yaw:.1f} fov={camera.fov:.1f}"
            )

    def set_up_control(self):
        self.control = self.model.control()

        # we are controlling the velocity
        out_dim = 6
        in_dim = self.model.joint_dof_count

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        @wp.kernel
        def compute_body_out(body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)):
            # TODO verify transform twist
            mv = transform_twist(wp.static(self.endeffector_offset), body_qd[wp.static(self.endeffector_id)])
            for i in range(6):
                body_out[i] = mv[i]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)

        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)

        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                (-50.0, -50.0, -10.0),
                wp.quat_identity(),
            ),
            floating=False,
            scale=100,  # URDF is in meters, scale to cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        clamp_close_activation_val = 0.1
        clamp_open_activation_val = 0.8

        # self.robot_key_poses = np.array(
        #     [
        #         # translation_duration, gripper transform (3D position [cm], 4D quaternion), gripper activation
        #         # top left
        #         [2.5, 31.0, -60.0, 23.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         [2, 31.0, -60.0, 23.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [2, 26.0, -60.0, 26.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [2, 12.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [3, -6.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [1, -6.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         # bottom left
        #         [2, 15.0, -33.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         [3, 15.0, -33.0, 21.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         [3, 15.0, -33.0, 21.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [2, 15.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [3, -2.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [1, -2.0, -33.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         # top right
        #         [2, -28.0, -60.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         [2, -28.0, -60.0, 20.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         [2, -28.0, -60.0, 20.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [2, -18.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [3, 5.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [1, 5.0, -60.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         # bottom right
        #         [3, -18.0, -30.0, 20.5, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         [3, -18.0, -30.0, 20.5, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [2, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [3, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [2, -3.0, -30.0, 31.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         # bottom
        #         [2, 0.0, -20.0, 30.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         [2, 0.0, -20.0, 19.5, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         [2, 0.0, -20.0, 19.5, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [2, 0.0, -20.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [1, 0.0, -30.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [1.5, 0.0, -30.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [1.5, 0.0, -40.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
        #         [1.5, 0.0, -40.0, 35.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #         [2, -28.0, -60.0, 28.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
        #     ],
        #     dtype=np.float32,
        # )
        self.robot_key_poses = np.array(
            [
                # translation_duration, gripper transform (3D position [cm], 4D quaternion), gripper activation
                # top left
                [3, 0, -50.0, 21.0, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, 0, -50.0, 21.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 0, -50.0, 26.0, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
            ],
            dtype=np.float32,
        )
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform(
            [
                0.0,
                0.0,
                22.0,
            ],
            wp.quat_identity(),
        )

    def compute_body_jacobian(
        self,
        model: Model,
        joint_q: wp.array,
        joint_qd: wp.array,
        include_rotation: bool = False,
    ):
        """
        Compute the Jacobian of the end effector's velocity related to joint_q

        """

        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel, 1, inputs=[self.temp_state_for_jacobian.body_qd], outputs=[self.body_out]
            )

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def generate_control_joint_qd(
        self,
        state_in: State,
    ):
        # After the key poses sequence ends, hold position with zero velocity
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        self.target = self.targets[current_interval]

        include_rotation = True

        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )

        self.compute_body_jacobian(
            self.model,
            state_in.joint_q,
            state_in.joint_qd,
            include_rotation=include_rotation,
        )
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)

        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = state_in.joint_q.numpy()

        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]

        K_null = 1.0
        delta_q_null = K_null * (q_des - q)

        delta_q = J_inv @ delta_target + N @ delta_q_null

        # Apply gripper finger control (finger positions in cm)
        delta_q[-2] = self.target[-1] * 4.0 - q[-2]
        delta_q[-1] = self.target[-1] * 4.0 - q[-1]

        self.target_joint_qd.assign(delta_q)

    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.capture_point_cloud()
        self.frame_index += 1

        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            # robot sim
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                # set particle_count = 0 to disable particle simulation in robot solver
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                # Update the robot pose - this will modify state_0 and copy to state_1
                self.model.shape_contact_pair_count = 0

                self.state_0.joint_qd.assign(self.target_joint_qd)
                # Just update the forward kinematics to get body positions from joint coordinates
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                self.state_0.particle_f.zero_()

                # restore original settings
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # cloth sim
            self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        self._print_camera_debug()

        # Scale particle and body positions from cm to meters for visualization
        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        # Swap model shape data to meter-scale for rendering
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        # Render the table box manually at meter scale
        self.viewer.log_shapes(
            "/table",
            newton.GeoType.BOX,
            self.table_viz_scale,
            self.table_viz_xform,
            self.table_viz_color,
        )
        self.viewer.end_frame()

        # Restore simulation shape data
        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale

    def test_final(self):
        p_lower = wp.vec3(-36.0, -95.0, -5.0)
        p_upper = wp.vec3(36.0, 5.0, 56.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 200.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 70.0,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=500)
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
