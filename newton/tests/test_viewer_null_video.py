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

import argparse
import unittest
from unittest.mock import Mock, patch


class TestNullViewerInit(unittest.TestCase):
    def test_init_auto_enables_output_mp4_for_null_viewer(self):
        import newton.examples

        parser = newton.examples.create_parser()
        args = argparse.Namespace(
            device=None,
            viewer="null",
            rerun_address=None,
            output_path="output.usd",
            num_frames=12,
            headless=False,
            test=False,
            quiet=False,
            benchmark=False,
            video_output_path=None,
            video_width=640,
            video_height=360,
            video_fps=24,
        )

        with patch.object(parser, "parse_args", return_value=args):
            with patch("newton.viewer.ViewerNull", return_value=Mock()) as viewer_cls:
                newton.examples.init(parser)

        viewer_cls.assert_called_once_with(
            num_frames=12,
            benchmark=False,
            benchmark_timeout=None,
            video_output_path="output.mp4",
            video_width=640,
            video_height=360,
            video_fps=24,
            strict_recording=False,
        )

    def test_init_respects_explicit_null_viewer_video_path(self):
        import newton.examples

        parser = newton.examples.create_parser()
        args = argparse.Namespace(
            device=None,
            viewer="null",
            rerun_address=None,
            output_path="output.usd",
            num_frames=8,
            headless=False,
            test=False,
            quiet=False,
            benchmark=False,
            video_output_path="captures/demo.mp4",
            video_width=800,
            video_height=600,
            video_fps=30,
        )

        with patch.object(parser, "parse_args", return_value=args):
            with patch("newton.viewer.ViewerNull", return_value=Mock()) as viewer_cls:
                newton.examples.init(parser)

        viewer_cls.assert_called_once_with(
            num_frames=8,
            benchmark=False,
            benchmark_timeout=None,
            video_output_path="captures/demo.mp4",
            video_width=800,
            video_height=600,
            video_fps=30,
            strict_recording=True,
        )

    def test_init_skips_auto_recording_for_tests(self):
        import newton.examples

        parser = newton.examples.create_parser()
        args = argparse.Namespace(
            device=None,
            viewer="null",
            rerun_address=None,
            output_path="output.usd",
            num_frames=8,
            headless=False,
            test=True,
            quiet=False,
            benchmark=False,
            video_output_path=None,
            video_width=800,
            video_height=600,
            video_fps=30,
        )

        with patch.object(parser, "parse_args", return_value=args):
            with patch("newton.viewer.ViewerNull", return_value=Mock()) as viewer_cls:
                newton.examples.init(parser)

        self.assertIsNone(viewer_cls.call_args.kwargs["video_output_path"])


class TestViewerNullRecorderDelegation(unittest.TestCase):
    def test_viewer_null_forwards_frame_lifecycle_to_video_recorder(self):
        from newton._src.viewer.viewer_null import ViewerNull

        recorder = Mock()
        with patch("newton._src.viewer.viewer_null._NullVideoRecorder", return_value=recorder):
            viewer = ViewerNull(num_frames=2, video_output_path="demo.mp4")

        viewer.begin_frame(0.25)
        viewer.end_frame()
        viewer.close()

        recorder.begin_frame.assert_called_once_with(0.25)
        recorder.end_frame.assert_called_once_with()
        recorder.close.assert_called_once_with()

    def test_viewer_null_sets_recorder_model_before_world_offsets(self):
        from newton._src.viewer.viewer_null import ViewerNull
        import warp as wp

        recorder = Mock()
        with patch("newton._src.viewer.viewer_null._NullVideoRecorder", return_value=recorder):
            viewer = ViewerNull(num_frames=2, video_output_path="demo.mp4")

        model = Mock()
        model.device = wp.get_device("cpu")
        model.shape_sdf_index = None
        model.world_count = 1
        model.up_axis = 2

        with patch("newton._src.viewer.viewer.ViewerBase._populate_shapes"):
            with patch("newton._src.viewer.viewer.ViewerBase._auto_compute_world_offsets") as auto_offsets:
                def _trigger_offsets():
                    viewer.set_world_offsets((1.0, 2.0, 0.0))

                auto_offsets.side_effect = _trigger_offsets
                viewer.set_model(model)

        self.assertEqual(recorder.method_calls[0][0], "set_model")
        self.assertIn(("set_world_offsets", ((1.0, 2.0, 0.0),), {}), recorder.method_calls)


if __name__ == "__main__":
    unittest.main()
