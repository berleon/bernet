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

import hashlib
import tempfile
from unittest import TestCase

import theano.tensor as T
from bernet.utils import *


class TestUtils(TestCase):
    def test_download(self):
        encoding = 'utf-8'
        with tempfile.NamedTemporaryFile("w+b") as server_file:
            server_file.write("Hello World!".encode(encoding))
            server_file.flush()
            with tempfile.NamedTemporaryFile("w+b") as f:
                download("file://" + server_file.name, f)
                f.seek(0)
                bytes = f.read()
                self.assertEqual(bytes.decode(encoding),  "Hello World!")

    def test_sha256_file(self):
        with tempfile.NamedTemporaryFile("w+b") as f:
            sha256sum = sha256_file(f)
            self.assertEqual(sha256sum,
                             hashlib.sha256("".encode("utf-8")).hexdigest())
            self.assertEqual(
                sha256sum,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e"
                "4649b934ca495991b7852b855")

    def test_shape_operators(self):
        self.assertEqual(bs((5,)), 1)
        self.assertEqual(chans((5,)), 1)
        self.assertEqual(h((5,)), 1)

        self.assertEqual(bs((5, 1, 4, 4)), 5)
        self.assertEqual(chans((5, 1, 4, 4)), 1)
        self.assertEqual(h((5, 1, 14, 4)), 14)
        self.assertEqual(w((5, 1, 4, 40)), 40)

    def test_symbolic_tensor_from_dims(self):
        t = symbolic_tensor_from_dims("test", 2)
        self.assertEqual(type(t), type(T.matrix()))

    def test_symbolic_tensor_from_shape(self):
        t = symbolic_tensor_from_shape("test", (20, 20))
        self.assertEqual(type(t), type(T.matrix()))

    def test_shared_like(self):
        array = theano.shared(np.ones((10, 10)), "array")
        t = shared_like(array, "t")
        self.assertTupleEqual(t.get_value().shape, array.get_value().shape)

    def test_print_confusion_matrix(self):
        matrix = np.asarray([
            [99, 0,  0,  0,  0, 50],
            [0, 96,  0,  0,  0,  0],
            [0,  0, 92,  0,  0,  0],
            [0,  0,  0, 85, 20,  0],
            [0,  2,  3,  7, 70,  0],
            [1,  2,  5,  8, 10, 50]])
        print_confusion_matrix(matrix / matrix.sum())
