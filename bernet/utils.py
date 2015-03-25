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
import operator
import urllib.request
from functools import reduce

import numpy as np
import theano
import theano.tensor as T


def download(url: str, file) -> bool:
    # TODO: make the download bar nicer (e.g MBytes, a real bar, ...)
    u = urllib.request.urlopen(url)
    meta = u.info()
    file_size = int(meta["Content-Length"])
    print("Downloading: %s Bytes: %s" % (url, file_size))

    file_size_dl = 0
    block_sz = 2**16
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        file.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl,
                                       file_size_dl * 100. / file_size)
        print(status, end='\r')

    file.flush()


def sha256_file(file, block_size: int=65536) -> str:
    sha = hashlib.sha256()

    file.seek(0)
    buf = file.read(block_size)
    while len(buf) > 0:
        sha.update(buf)
        buf = file.read(block_size)

    return sha.hexdigest()


def bs(shape):
    """:return the batch size of `shape`."""
    if len(shape) >= 4:
        return shape[-4]
    else:
        return 1


def chans(shape):
    if len(shape) >= 3:
        return shape[-3]
    else:
        return 1


def h(shape):
    """:return the width of `shape`."""
    if len(shape) >= 2:
        return shape[-2]
    else:
        return 1


def w(shape):
    """:return the width of `shape`."""
    return shape[-1]


def size(shape):
    """:return the total number of elements.
    E.g. `size((2, 20, 10))` would be `2*20*10 = 400`"""
    return reduce(operator.mul, shape, 1)


def shared_tensor_from_dims(name, dims):
    shape = (1, ) * dims
    return theano.shared(np.zeros(shape), name=name)


def symbolic_tensor_from_dims(name, dims):
    tpe = T.TensorType(dtype=theano.config.floatX,
                       broadcastable=(False,)*dims)
    return tpe(name)


def symbolic_tensor_from_shape(name, shp):
    floatX = theano.config.floatX
    tpe = T.TensorType(dtype=floatX, broadcastable=(False,)*len(shp))
    return tpe(name)


def shared_like(shared_tensor, name, init=0):
    return theano.shared(np.zeros_like(shared_tensor.get_value()) + init,
                         name="{}_{}".format(shared_tensor.name, name))


def confusion_matrix(pred_labels, true_labels):
    n_cls = np.unique(pred_labels).size
    n_examples = true_labels.size
    return np.bincount(n_cls * true_labels + pred_labels,
                       minlength=n_cls*n_cls).reshape(n_cls, n_cls)/n_examples


def print_confusion_matrix(matrix):
    from termcolor import colored
    n = matrix.shape[0]
    for i in range(n):
        def with_color(i, j):
            v = matrix[i, j]
            total = np.sum(matrix[:, i])
            formated = "{:.3g}".format(matrix[i, j]*100).center(5)
            if i == j:
                percent_right = v / total
                if percent_right >= 0.99:
                    return colored(formated, 'green', attrs=['bold'])
                elif percent_right >= 0.95:
                    return colored(formated, 'green')
                elif percent_right >= 0.9:
                    return colored(formated, 'yellow')
                elif percent_right >= 0.80:
                    return colored(formated, 'red')
                else:
                    return colored(formated, 'red', attrs=['bold'])
            else:
                percent_false = v / total
                if percent_false <= 0.01:
                    return colored(formated, 'grey', attrs=['bold'])
                elif percent_false <= 0.03:
                    return formated
                elif percent_false <= 0.06:
                    return colored(formated, 'yellow')
                elif percent_false < 0.10:
                    return colored(formated, 'red')
                else:
                    return colored(formated, 'red', attrs=['bold'])

        print(" ".join([with_color(i, j) for j in range(n)]))
