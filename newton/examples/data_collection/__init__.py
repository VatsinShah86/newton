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

"""Data collection examples."""

from __future__ import annotations

import os


def get_examples() -> dict[str, str]:
    """Return data-collection example names mapped to full module paths."""
    example_map = {}
    examples_dir = os.path.realpath(os.path.dirname(__file__))
    for filename in sorted(os.listdir(examples_dir)):
        if filename.startswith("example_") and filename.endswith(".py"):
            example_name = filename[8:-3]
            example_map[example_name] = f"newton.examples.data_collection.{filename[:-3]}"
    return example_map


__all__ = ["get_examples"]
