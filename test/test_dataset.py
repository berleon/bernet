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
from PIL import Image

import numpy as np
import numpy.testing
from bernet.dataset import MNISTDataset, Dataset, GeneratedDataset, \
    LineDataset, ILSVRCDataset
from bernet.utils import size, save_images


class TestDataset(TestCase):
    def test_dataset(self):
        dataset = Dataset()
        self.assertRaises(NotImplementedError, dataset.train_epoch)
        self.assertRaises(NotImplementedError, dataset.test_epoch)
        self.assertRaises(NotImplementedError, dataset.validate_epoch)


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
        train_batch = next(dataset.train_epoch())
        self.assertEqual(size(train_batch.data().shape[1:2]),
                         size((28, 28)))

        self.assertTupleEqual(train_batch.labels().shape,
                              (train_batch.n_examples(),))

        test_batch = next(dataset.test_epoch())
        self.assertEqual(size(test_batch.data().shape[1:2]),
                         size((28, 28)))

        valid_batch = next(dataset.validate_epoch())
        self.assertEqual(size(valid_batch.data().shape[1:2]),
                         size((28, 28)))

    def test_minibatch(self):
        mnist = MNISTDataset()
        for train in mnist.train_epoch():
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
        train_batch = next(dataset.train_epoch())
        self.assertEqual(size(train_batch.data().shape[1:2]),
                         size((28, 28)))


class TestILSVRCDataset(TestCase):
    def test_dataset(self):
        _dir = os.path.dirname(__file__)
        ilsvrc2012_dir = _dir + "/../data/ILSVRC2012"
        use_random_data = False
        if not os.path.exists(ilsvrc2012_dir):
            use_random_data = True
            ilsvrc2012_dir = tempfile.mkdtemp()
            os.makedirs(ilsvrc2012_dir + '/validate')
            n_images = 64
            images = np.uint8(255*np.random.sample((n_images, 3, 277, 277)))
            save_images(images,
                        ['{}/validate/{}.jpeg'.format(ilsvrc2012_dir, i)
                         for i in range(n_images)])
            ground_truth = ilsvrc2012_dir + '/validate/ground_truth.txt'
            with open(ground_truth, mode='w') as gt:
                gt.write('\n'.join(
                    [str(int(10*i)) for i in np.random.sample((n_images, ))]))

        dataset = ILSVRCDataset(epoch_size=2, data_dir=ilsvrc2012_dir)
        epoche = next(dataset.validate_epoch())
        first_img_data = np.uint8(epoche.data()[0, :] * 255).\
            swapaxes(0, 2).swapaxes(0, 1)
        first_img = Image.fromarray(first_img_data)
        self.assertTupleEqual(epoche.data().shape,
                              (dataset.epoch_size, 3, 227, 227))
        self.assertTupleEqual(epoche.labels().shape, (dataset.epoch_size,))

        if use_random_data:
            shutil.rmtree(ilsvrc2012_dir)


class TestGeneratedDataset(TestCase):
    def test_generated_dataset(self):
        m = 5
        c = 3

        def line(x):
            return m*x + c
        dataset = GeneratedDataset(lambda x: x, line, (10, 10))
        train = next(dataset.train_epoch())
        self.assertTrue(np.all(line(train.data()) == train.labels()))


class TestLineDataset(TestCase):
    def test_line_dataset(self):
        m = 5
        c = 3

        def line(x):
            return m*x + c
        dataset = LineDataset((100, 1), m=m, c=c)
        train = next(dataset.train_epoch())
        numpy.testing.assert_array_almost_equal(
            line(train.data()).reshape((-1)),
            train.labels())
