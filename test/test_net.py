# Copyright 2015 Leon Sixt
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
import tempfile

from unittest import TestCase

import numpy as np

from bernet import utils

from bernet.net import Network


class TestNetwork(TestCase):
    def test_get_data(self):
        with tempfile.NamedTemporaryFile("w+b") as f:
            random = np.random.sample((200, 200))
            np.savez(f, rand=random)
            f.seek(0)
            nn = Network(
                name="test_net",
                data_url="file://" + f.name,
                data_sha256=utils.sha256_file(f)
            )

            self.assert_(np.all(nn.data["rand"] == random))

    def test_check_sha256sum(self):
        with tempfile.NamedTemporaryFile("w+b") as f:
            random = np.random.sample((200, 200))
            np.savez(f, rand=random)
            f.seek(0)
            self.assertRaises(ValueError, Network,
                              name="test_net", data_url="file://" + f.name,
                              data_sha256="wrong")
