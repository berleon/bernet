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
import bernet.utils


class Batch(object):
    def __init__(self, data, labels=None):
        self._data = data
        self._labels = labels

    def n_examples(self):
        return self._data.shape[0]

    def data(self):
        return self._data

    def labels(self):
        return self._labels

    def minibatch_idx(self, n=128):
        for i in range(0, self.n_examples()//n, n):
            yield (i, i+n)


class Dataset(object):
    def train(self) -> Batch:
        raise NotImplementedError("")

    def test(self) -> Batch:
        raise NotImplementedError("")

    def validate(self) -> Batch:
        raise NotImplementedError("")


class MNISTDataset(Dataset):
    _url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
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
            bernet.utils.download(self._url, f)

    def train(self) -> Batch:
        yield Batch(self._train_set[0], self._train_set[1])

    def validate(self) -> Batch:
        yield Batch(self._valid_set[0], self._valid_set[1])

    def test(self) -> Batch:
        yield Batch(self._test_set[0], self._test_set[1])
