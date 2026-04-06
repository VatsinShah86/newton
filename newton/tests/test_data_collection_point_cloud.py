# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.examples.data_collection.example_cloth_franka import (
    DEFORMABLE_TRIANGLE_MESH_SHAPE_ID,
    SensorCameraConfig,
    _extract_point_cloud_world_m,
    _merge_point_clouds,
    _next_run_dir,
)


class TestDataCollectionPointCloud(unittest.TestCase):
    def test_next_run_dir_starts_at_one_and_increments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            run_dir_1 = _next_run_dir(base_dir)
            self.assertEqual(run_dir_1.name, "run_1")
            self.assertTrue((run_dir_1 / "pc").is_dir())

            run_dir_2 = _next_run_dir(base_dir)
            self.assertEqual(run_dir_2.name, "run_2")
            self.assertTrue((run_dir_2 / "pc").is_dir())

    def test_merge_point_clouds_concatenates_two_cameras(self):
        camera_0 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        camera_1 = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)

        merged = _merge_point_clouds([camera_0, camera_1])

        np.testing.assert_array_equal(
            merged,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_extract_point_cloud_world_m_filters_non_deformable_hits_and_converts_units(self):
        depth_image_cm = np.array([[100.0, 75.0]], dtype=np.float32)
        shape_index_image = np.array(
            [[DEFORMABLE_TRIANGLE_MESH_SHAPE_ID, np.uint32(7)]],
            dtype=np.uint32,
        )
        camera_rays = np.zeros((1, 2, 2, 3), dtype=np.float32)
        camera_rays[..., 1, :] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        camera = SensorCameraConfig(
            name="cam_0",
            pos_m=(0.0, 0.0, 0.0),
            pitch_deg=0.0,
            yaw_deg=0.0,
            fov_deg=45.0,
            width_px=2,
            height_px=1,
        )

        points_world_m = _extract_point_cloud_world_m(depth_image_cm, shape_index_image, camera_rays, camera)

        np.testing.assert_allclose(points_world_m, np.array([[1.0, 0.0, 0.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
