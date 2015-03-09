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
import os
import shutil
import tempfile
from unittest import TestCase
from bernet.dataset import MNISTDataset, Dataset
from bernet.utils import size


class TestDataset(TestCase):
    def test_dataset(self):
        dataset = Dataset()
        self.assertRaises(NotImplementedError, dataset.train)
        self.assertRaises(NotImplementedError, dataset.test)
        self.assertRaises(NotImplementedError, dataset.validate)


class TestMNISTDataset(TestCase):
    def setUp(self):
        local = MNISTDataset._local_file
        if os.path.exists(local):
            _, self.tmp_abs_path = tempfile.mkstemp()
            shutil.move(local, self.tmp_abs_path)
            MNISTDataset._url = "file://" + self.tmp_abs_path

    def tearDown(self):
        if hasattr(self, 'tmp_abs_path') and os.path.exists(self.tmp_abs_path):
            os.remove(self.tmp_abs_path)

    def test_mnist_dataset(self):
        dataset = MNISTDataset()
        train_batch = next(dataset.train())
        self.assertEqual(size(train_batch.data().shape[1:2]),
                         size((28, 28)))

        self.assertTupleEqual(train_batch.labels().shape,
                              (train_batch.n_examples(),))

        test_batch = next(dataset.test())
        self.assertEqual(size(test_batch.data().shape[1:2]),
                         size((28, 28)))

        valid_batch = next(dataset.validate())
        self.assertEqual(size(valid_batch.data().shape[1:2]),
                         size((28, 28)))

    def test_minibatch(self):
        mnist = MNISTDataset()
        for train in mnist.train():
            first = True
            n = 32
            for start, end in train.minibatch_idx(n):
                self.assertEqual(start % n, 0)
                self.assertEqual(end % n, 0)
                self.assertLessEqual(end, train.n_examples())

    def test_handels_corruption(self):
        local_file = MNISTDataset._local_file
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        with open(local_file, 'w+') as f:
            f.write("bla")

        dataset = MNISTDataset()
        train_batch = next(dataset.train())
        self.assertEqual(size(train_batch.data().shape[1:2]),
                         size((28, 28)))
