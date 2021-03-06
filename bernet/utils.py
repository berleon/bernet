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
from PIL import Image, ImageDraw, ImageFont
from math import sqrt, ceil

import numpy as np
import theano

import theano.tensor as T


def download(url: str, file) -> bool:
    # TODO: make the download bar nicer (e.g MBytes, a real bar, ...)
    u = urllib.request.urlopen(url)
    meta = u.info()
    file_size = int(meta["Content-Length"])
    print("Downloading url {} to {}. Total bytes: {}"
          .format(url, file.name, file_size))

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


def sha256_of_file(file, block_size: int=65536) -> str:
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


def tile_image(arr, tile_spacing=(1, 1), name=""):
    def rectify(n):
        n_w = ceil(sqrt(n))
        n_h = ceil(n / n_w)
        return (n_h, n_w)

    if arr.ndim > 3:
        arr = arr.reshape(-1, arr.shape[1], arr.shape[2])

    assert arr.ndim == 3
    n = arr.shape[0]
    arr_h = arr.shape[1]
    arr_w = arr.shape[2]
    n_h, n_w = rectify(n)
    t_h, t_w = tile_spacing
    img_shape = (n_h*arr_h + (n_h-1)*t_h,
                 n_w*arr_w + (n_w-1)*t_w,)
    img_arr = np.ones(img_shape) * 0.5
    print(img_shape)
    for i in range(n_w):
        for j in range(n_h):
            s_w = slice(i*(arr_w+t_w), (i+1)*(arr_w+t_w) - t_w)
            s_h = slice(j*(arr_h+t_h), (j+1)*(arr_h+t_h) - t_h)
            channel = i*n_h + j
            if channel < n:
                img_arr[s_h, s_w] = arr[channel]

    img = Image.fromarray(img_arr).convert('RGB')
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = None

    d = ImageDraw.Draw(img)
    d.text((5, 5), name, font=font, fill=(0, 255, 0))
    return img


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


class fast_compile(object):
    def __enter__(self):
        self.saved_mode = theano.config.mode
        theano.config.mode = 'FAST_COMPILE'

    def __exit__(self, exc_type, exc_val, exc_tb):
        theano.config.mode = self.saved_mode


def prod(factors):
    return reduce(operator.mul, factors, 1)


def chunks(list, n):
    """ Yield successive n-sized chunks from list l. """
    for i in range(0, len(list), n):
        yield list[i:i+n]


def to_image_magic(x):
    """Reshapes x with shape (..., channels, height, weight) to Image Magic
    shape (..., height, weight, channel)."""
    assert x.ndim >= 3
    n = x.ndim - 3
    return x.swapaxes(n+1, n+2).swapaxes(n, n+2)


def from_image_magic(x):
    """Reshapes x with a Image Magic shape of (..., height, weight, channel) to
    a shape of (..., channel, height, weight)."""
    assert x.ndim >= 3
    n = x.ndim - 3
    return x.swapaxes(n, n+2).swapaxes(n+1, n+2)


def save_images(image_np_arrays, image_paths):
    assert image_np_arrays.shape[0], len(image_paths)
    image_np_arrays = to_image_magic(image_np_arrays)
    for i in range(image_np_arrays.shape[0]):
        img = Image.fromarray(image_np_arrays[i])
        img.save(image_paths[i])


def load_images(image_paths, size):
    shape = (len(image_paths), 3, size[0], size[1])
    arr = np.empty(shape, dtype=theano.config.floatX)
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('RGB')
        if img.size != size:
            width, height = img.size
            left, right, upper, lower = 0, width, 0, height
            if width > height:
                diff = (width - height)
                left = diff // 2
                right = diff // 2 + height
            else:
                diff = (height - width)
                upper = diff // 2
                lower = diff // 2 + width

            croped = img.crop((left, upper, right, lower))
            img = croped.resize(size, Image.NEAREST)
        normalized = np.array(img, dtype=theano.config.floatX) / 255
        arr[i, :] = normalized.swapaxes(0, 2).swapaxes(1, 2)
    return arr
