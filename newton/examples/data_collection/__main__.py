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

"""Entry point for running data-collection examples via ``python -m``."""

from __future__ import annotations

import runpy
import sys

from . import get_examples


def main():
    """Run a data-collection example by short name."""
    examples = get_examples()

    if len(sys.argv) < 2:
        print("Usage: python -m newton.examples.data_collection <example_name>")
        print("\nAvailable examples:")
        for name in examples:
            print(f"  {name}")
        sys.exit(1)

    example_name = sys.argv[1]

    if example_name not in examples:
        print(f"Error: Unknown example '{example_name}'")
        print("\nAvailable examples:")
        for name in examples:
            print(f"  {name}")
        sys.exit(1)

    target_module = examples[example_name]
    sys.argv = [target_module, *sys.argv[2:]]
    runpy.run_module(target_module, run_name="__main__")


if __name__ == "__main__":
    main()
