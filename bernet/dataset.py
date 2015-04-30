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

import gzip
import os
import pickle
from multiprocessing import Pipe, Process

import numpy as np
from bernet.utils import chunks, load_images, download


class Epoche(object):
    def __init__(self, data, labels=None):
        self._data = data
        self._labels = labels

    def n_examples(self) -> int:
        return self._data.shape[0]

    def data(self) -> 'ndarray':
        return self._data

    def labels(self):
        return self._labels

    def minibatch_idx(self, batch_size=64):
        return zip(range(0, self.n_examples() - batch_size, batch_size),
                   range(batch_size, self.n_examples(), batch_size))


class Dataset(object):
    def train_epoch(self) -> Epoche:
        raise NotImplementedError("")

    def test_epoch(self) -> Epoche:
        raise NotImplementedError("")

    def validate_epoch(self) -> Epoche:
        raise NotImplementedError("")

    def data_dims(self) -> int:
        raise NotImplementedError("")

    def labels_dims(self) -> int:
        raise NotImplementedError("")


class MNISTDataset(Dataset):
    _url = "https://github.com/mnielsen/neural-networks-and-deep-learning/" \
           "blob/master/data/mnist.pkl.gz?raw=true"
    _local_file = os.path.expanduser("~/.bernet/mnist/mnist.pkl.gz")

    def __init__(self):
        def load_mnist():
            with gzip.open(self._local_file, 'rb') as f:
                self._train_set, self._valid_set, self._test_set = \
                    pickle.load(f, encoding='latin1')

        if not os.path.exists(self._local_file):
            self._download_mnist()

        try:
            load_mnist()
        except Exception as e:
            print("Exception {} occured. Delete File and retry to download"
                  .format(e))
            os.remove(self._local_file)
            self._download_mnist()
            load_mnist()

    def _download_mnist(self):
        os.makedirs(os.path.dirname(self._local_file), exist_ok=True)
        with open(self._local_file, "wb+") as f:
            download(self._url, f)

    def labels_dims(self):
        return 1

    def data_dims(self):
        return 2

    def train_epoch(self) -> Epoche:
        yield Epoche(self._train_set[0], self._train_set[1])

    def validate_epoch(self) -> Epoche:
        yield Epoche(self._valid_set[0], self._valid_set[1])

    def test_epoch(self) -> Epoche:
        yield Epoche(self._test_set[0], self._test_set[1])


def _ilsvrc_preprocessing(conn, data_dir, task, epoch_size):
    conn.recv()
    dir = os.path.join(data_dir, task)
    files = sorted(os.listdir(dir))
    all_labels = np.uint32(np.genfromtxt(dir + '/ground_truth.txt'))
    chunks_files = chunks(files, epoch_size)
    chunks_labels = chunks(all_labels, epoch_size)
    for i, (img_names, labels) in enumerate(zip(chunks_files, chunks_labels)):
        img_paths = list(map(lambda n: os.path.join(dir, n), img_names))
        data = load_images(img_paths, (227, 227))
        conn.send((data, labels))


class ILSVRCEpochGenerator():
    def __init__(self, data_dir, epoch_size, task):
        self.parent_conn, preprocessing_conn = Pipe()
        p = Process(target=_ilsvrc_preprocessing, daemon=True,
                    args=(preprocessing_conn, data_dir, task, epoch_size))
        p.start()
        self.parent_conn.send(True)

    def __iter__(self):
        return self

    def __next__(self):
        self.parent_conn.send(True)
        data, labels = self.parent_conn.recv()
        return Epoche(data, labels)


class ILSVRCDataset(Dataset):
    def __init__(self, data_dir=None, epoch_size=128):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__),
                                    "../data/ILSVRC2011/")
        self.data_dir = data_dir
        self.epoch_size = epoch_size

    def labels_dims(self) -> int:
        return 1

    def train_epoch(self) -> Epoche:
        return self._generate_epoch('train')

    def test_epoch(self) -> Epoche:
        return self._generate_epoch('test')

    def validate_epoch(self) -> Epoche:
        return self._generate_epoch('validate')

    def data_dims(self) -> int:
        pass

    def _generate_epoch(self, task):
        return ILSVRCEpochGenerator(self.data_dir, self.epoch_size, task)


class GeneratedDataset(Dataset):
    def __init__(self, data_func, label_func, shape, seed=None):
        self.data_func = data_func
        self.label_func = label_func
        self.shape = shape
        self.seed = seed

    def labels_dims(self):
        return 1

    def data_dims(self):
        return len(self.shape)

    def _random(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.sample(self.shape)

    def _generate_batch(self):
        rand = self._random()
        data = self.data_func(rand)
        return Epoche(data, labels=self.label_func(data))

    def train_epoch(self) -> Epoche:
        yield self._generate_batch()

    def test_epoch(self) -> Epoche:
        yield self._generate_batch()

    def validate_epoch(self) -> Epoche:
        yield self._generate_batch()


class LineDataset(GeneratedDataset):
    def __init__(self, shape, m=5, c=3, seed=None):
        super().__init__(lambda x: x,
                         lambda x: np.reshape(m*x + c, (-1,)),
                         shape,
                         seed=seed)
