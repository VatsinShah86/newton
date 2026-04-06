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

import unittest


class TestExamplesRegistry(unittest.TestCase):
    def test_top_level_registry_keeps_unique_short_names(self):
        import newton.examples

        examples = newton.examples.get_examples()

        self.assertEqual(
            examples["softbody_hanging"],
            "newton.examples.softbody.example_softbody_hanging",
        )

    def test_top_level_registry_exposes_qualified_names_for_duplicates(self):
        import newton.examples

        examples = newton.examples.get_examples()

        self.assertNotIn("softbody_franka", examples)
        self.assertEqual(
            examples["softbody.softbody_franka"],
            "newton.examples.softbody.example_softbody_franka",
        )
        self.assertEqual(
            examples["data_collection.softbody_franka"],
            "newton.examples.data_collection.example_softbody_franka",
        )

    def test_data_collection_subpackage_registry(self):
        import newton.examples.data_collection

        examples = newton.examples.data_collection.get_examples()

        self.assertEqual(
            examples["softbody_franka"],
            "newton.examples.data_collection.example_softbody_franka",
        )


if __name__ == "__main__":
    unittest.main()
