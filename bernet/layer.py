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
import random

import numpy as np
import re
import theano
from theano.ifelse import ifelse
import theano.tensor as T
from theano.tensor.nnet import conv2d
import theano.tensor.signal.downsample
from yaml import ScalarNode, SequenceNode, MappingNode

from bernet.config import REQUIRED, OPTIONAL, TAGS, REPEAT, ConfigObject, \
    ConfigField, ENUM, config_error, EITHER, ConfigError

from bernet.utils import chans, bs, w, h, fast_compile, prod

# # # # # # # # # # # # # # # # # - Utilities - # # # # # # # # # # # # # # # #


class Filler(ConfigObject):
    @staticmethod
    def _apply_sparsity(sparsity, param):
        if sparsity is not None:
            param *= np.random.binomial(n=1, p=1.-sparsity, size=param.shape)

    def fill(self, shape):
        raise NotImplementedError("Please use a subclass of Filler")


class ConstFiller(Filler):
    const_value = REQUIRED(float)
    sparsity = OPTIONAL(float, default=0.)

    def fill(self, shape):
        param = np.empty(shape)
        param.fill(self.const_value)
        self._apply_sparsity(self.sparsity, param)
        return param


class UniformFiller(Filler):
    low = REQUIRED(float)
    high = REQUIRED(float)
    sparsity = OPTIONAL(float, default=0.)

    def fill(self, shape):
        param = np.random.uniform(low=self.low, high=self.high, size=shape)
        self._apply_sparsity(self.sparsity, param)
        return param


class GaussianFiller(Filler):
    mean = OPTIONAL(float, default=0.)
    std = OPTIONAL(float, default=1.)
    sparsity = OPTIONAL(float, default=0.)

    def fill(self, shape):
        param = self.std*np.random.standard_normal(size=shape) + self.mean
        self._apply_sparsity(self.sparsity, param)
        return param


class Shape(REPEAT):
    # the total maximum allowed dimensions are
    # (batch size, channels, height, width)
    MAX_DIMS = 4

    def __init__(self, dims: int=None, max_dims: int=MAX_DIMS,
                 doc="", type_sig=""):
        super().__init__(int, doc=doc, type_sig=type_sig)
        self.dims = dims
        self.max_dims = max_dims
        if self.max_dims > self.MAX_DIMS:
            raise ValueError("The maximum number of dimension is {:}."
                             .format(self.MAX_DIMS))

    def construct(self, loader, node):
        listlike = super().construct(loader, node)

        if self.dims is None:
            shape = (1,) * (self.max_dims-len(listlike)) + tuple(listlike)
        else:
            shape = tuple(listlike)
        self.assert_valid(shape)
        return shape

    def represent(self, dumper, listlike):
        node = super().represent(dumper, listlike)
        node.flow_style = True
        return node

    def assert_valid(self, listlike, node=None):
        if listlike is None:
            raise config_error("No Shape given.", node=node)

        dims = len(listlike)

        if self.dims is not None and self.dims != dims:
            raise config_error(
                "A dimension of '{:}' is required. Got '{:}' with a dimension "
                "of '{:}'".format(self.dims, listlike, dims))

        if dims > self.max_dims:
            raise config_error(
                "Shape '{:}' has a dimension of '{:}'. The maximum allowed "
                "dimension is {:}."
                .format(listlike, len(listlike), self.max_dims))

        if len(listlike) == 0:
            raise config_error("Shape needs at least one element.")
        for elem in listlike:
            if elem <= 0:
                raise config_error("Shape must be strictly positive.")


class Parameter(ConfigObject):
    name = REQUIRED(str)
    shape = OPTIONAL(REPEAT(int))
    filler = OPTIONAL(Filler, default=GaussianFiller(mean=0., std=1.))
    learning_rate = OPTIONAL(float, default=1.)
    weight_decay = OPTIONAL(float, default=1.)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tensor = None
        self._shared = None

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        self._tensor = tensor
        self._shared = theano.shared(self.tensor, name=self.name)

    @property
    def shared(self):
        return self._shared


class NotConnectedException(Exception):
    pass


def layer_type(cls):
    """Returns the type of the layer class.
     For Example layer_type("ConvLayer") will return "Conv"
     """
    return cls.__name__.rstrip("Layer")


class Layer(ConfigObject):
    name = REQUIRED(str, doc="The name of the layer. The names within a "
                             "Network must be unique.")
    source = OPTIONAL(str, doc="Use the output of the `source` layer as input "
                               "of this layer.`TanhLayer(name=\"tanh#1\", "
                               "source=\"conv#1\")` uses the output of "
                               "`conv#1` as its input.")

    def __init__(self, **kwargs):
        """
            :param name: layer name
            :param options: LayerOption
            :param parameters: list of [..] TODO
            """
        super().__init__(**kwargs)

    def output(self, input: 'Theano Expression'):
        reshaped_input = self._reshape(input)
        return self._output(reshaped_input)

    def output_shape(self, input_shape: tuple):
        batch_size = bs(input_shape)
        out = self.output(T.zeros((1,) + input_shape[1:]))
        with fast_compile():
            return (batch_size,) + tuple(out.shape.eval())[1:]

    def _output(self, input):
        raise NotImplementedError("Please use a subclass of Layer")

    def _reshape(self, input: 'Theano Expression'):
        expected = self._expected_shape()

        if expected is not None:
            return ifelse(T.eq(expected, input.shape),
                          input,
                          input.reshape(expected))

        max_dims = self._reshape_dims()
        if input.ndim > max_dims:
            sym_shp = input.shape
            if max_dims == 2:
                shp = (sym_shp[0], -1)
            if max_dims == 3:
                shp = (sym_shp[0], sym_shp[1], -1)

            return input.reshape(shp, ndim=max_dims)

        return input

    def _reshape_dims(self):
        return 4

    def _expected_shape(self):
        return None


class ParameterLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = []

    def parameter_shape(self, param: 'Parameter|str'):
        if type(param) == str:
            param = self._param_by_name(param)

        return self._parameter_shape(param)

    def _param_by_name(self, param_name):
        fitting_params = [p for p in self.parameters if p.name == param_name]
        assert len(fitting_params) == 1
        return fitting_params[0]

    def _parameter_shape(self, param: Parameter):
        raise NotImplementedError("Please use a subclass of Layer")

    def fill_parameter(self, param: 'Parameter|str'):
        if type(param) == str:
            param = self._param_by_name(param)

        param.tensor = param.filler.fill(self.parameter_shape(param))

    def fill_parameters(self):
        for p in self.parameters:
            self.fill_parameter(p)


class ConvLayer(ParameterLayer):
    weight = REQUIRED(Parameter)
    bias = REQUIRED(Parameter)

    num_feature_maps = REQUIRED(int)

    # kernel width
    kernel_w = REQUIRED(int)
    # kernel height
    kernel_h = REQUIRED(int)

    stride_v = OPTIONAL(int, default=1,
                        doc="Number of pixels the filter moves in vertical "
                            "direction.")
    stride_h = OPTIONAL(int, default=1,
                        doc="Number of pixels the filter moves in horizontal "
                            "direction.")

    group = OPTIONAL(int, default=1)

    border_mode = OPTIONAL(ENUM("valid", "full", "same"), default="valid")

    input_shape = REQUIRED(Shape(dims=4), doc="""Shape of the input tensor""")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = [self.bias, self.weight]

    def _parameter_shape(self, param):
        if param.shape is not None:
            return param.shape

        if param == self.weight:
            print("Shape of {} is {}".format(param.name, self.filter_shape()))
            return self.filter_shape()
        if param == self.bias:
            return self.bias_shape()

    def filter_shape(self):
        return (self.num_feature_maps,
                chans(self.input_shape) // self.group,
                self.kernel_h,
                self.kernel_w)

    def bias_shape(self):
        shp = self.filter_shape()
        return shp[0],

    def _reshape(self, input: 'Theano Expression'):
        return T.reshape(input, self.input_shape)

    def _output(self, input):
        assert self.weight.tensor is not None
        in_chan = chans(self.input_shape) // self.group
        f = self.num_feature_maps // self.group
        filter_shape = (f, ) + self.filter_shape()[1:]
        input_shape = self.input_shape[:1] + (in_chan, ) + self.input_shape[2:]
        conv_outs = []
        if self.border_mode == 'same':
            assert self.stride_h == 1 and self.stride_v == 1
            border_mode = 'full'
        else:
            border_mode = self.border_mode
        for i in range(self.group):
            conv_out = T.nnet.conv2d(
                input=input[:, in_chan*i:in_chan*(i+1), :],
                image_shape=input_shape,
                filters=self.weight.shared[f*i:f*(i+1), :],
                filter_shape=filter_shape,
                subsample=(self.stride_h, self.stride_v),
                border_mode=border_mode,
            )
            if self.border_mode == 'same':
                conv_out = self._fix_same_border_mode(conv_out)
            conv_outs.append(conv_out)

        concated = T.concatenate(conv_outs, axis=1)
        if self.bias is None:
            return concated
        else:
            return concated + self.bias.shared.dimshuffle('x', 0, 'x', 'x')

    def output_shape(self, in_shp: tuple):
        if self.border_mode == 'same':
            return (in_shp[0],) + (self.num_feature_maps,) + in_shp[2:]

        batch_size = bs(in_shp)
        channels = self.num_feature_maps  # * chans(input_shape)
        miss_h = self.kernel_h - 1
        miss_w = self.kernel_w - 1
        # maximum possible step, if stride_X = 1.
        if self.border_mode == "valid":
            max_steps_h = h(in_shp) - miss_h
            max_steps_w = w(in_shp) - miss_w
        elif self.border_mode == "full":
            max_steps_h = h(in_shp) + miss_h
            max_steps_w = w(in_shp) + miss_w
        height = max_steps_h // self.stride_h
        width = max_steps_w // self.stride_v
        if max_steps_h % self.stride_h != 0:
            height += 1
        if max_steps_w % self.stride_v != 0:
            width += 1
        return batch_size, channels, height, width

    def _fix_same_border_mode(self, conv_out):
        hb = (conv_out.shape[-2] - h(self.input_shape)) // 2
        he = hb + h(self.input_shape)

        wb = (conv_out.shape[-1] - w(self.input_shape)) // 2
        we = wb + w(self.input_shape)

        return conv_out[:, :, hb:he, wb:we]


class InnerProductLayer(ParameterLayer):
    n_units = REQUIRED(int)
    weight = OPTIONAL(Parameter)
    bias = OPTIONAL(EITHER(Parameter, bool))
    input_shape = REQUIRED(Shape(dims=2), doc="Shape of the input tensor.")

    def __init__(self, **kwargs):
        if "weight" not in kwargs:
            kwargs["weight"] = Parameter(name=kwargs['name'] + '_weight')
        if "bias" in kwargs and type(kwargs['bias']) == bool:
            if kwargs['bias']:
                kwargs['bias'] = Parameter(name=kwargs['name'] + '_bias')
            else:
                del kwargs['bias']

        super().__init__(**kwargs)
        self.parameters = [self.weight]
        if self.bias is not None:
            self.parameters.append(self.bias)

    def _parameter_shape(self, param: Parameter):
        in_size = prod(self.input_shape[1:])
        if param == self.weight:
            return self.n_units, in_size
        elif param == self.bias:
            return self.n_units,

    def _reshape_dims(self):
        return 2

    def _output(self, input):
        ip = T.dot(input, self.weight.shared.T)
        if self.bias is not None:
            return ip + self.bias.shared
        else:
            return ip

    def output_shape(self, input_shape: tuple):
        return input_shape[0], self.n_units


class PoolingLayer(Layer):
    poolsize = REQUIRED(Shape(max_dims=2))
    stride = OPTIONAL(Shape(max_dims=2), default=(1, 1))
    ignore_border = OPTIONAL(bool, default=False)

    def _output(self, input):
        return theano.tensor.signal.downsample.max_pool_2d(
            input=input,
            ds=self.poolsize,
            st=self.stride,
            ignore_border=self.ignore_border
        )


# ------------------------- Normalization Layers ------------------------------


class LRNLayer(Layer):
    """
    Local Response Normalization (LRN)

    See "ImageNet Classification with Deep Convolutional Neural Networks"
    Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton
    NIPS 2012
    """
    n = OPTIONAL(int, default=5)
    alpha = OPTIONAL(float, default=0.001)
    beta = OPTIONAL(float, default=0.75)
    k = OPTIONAL(float, default=3.)

    def _output(self, input):
        sq = T.sqr(input)
        nb_chans = input.shape[1]
        start = self.n // 2
        end = start + nb_chans
        sq = sq.reshape((sq.shape[0], 1, sq.shape[1], -1))
        filter_shape = (1, 1, self.n, 1)
        filter = T.ones(filter_shape)
        conved = conv2d(input=sq, filters=filter, border_mode='full')
        conved = conved.reshape((input.shape[0], -1, input.shape[2],
                                 input.shape[3]))
        scaled = self.k + (self.alpha / self.n) * conved[:, start:end, :, :]
        scale = scaled ** self.beta
        return input / scale

    def output_shape(self, input_shape: tuple):
        return input_shape

# ------------------------- Activation Layers ---------------------------------


class ActivationLayer(Layer):
    def output_shape(self, input_shape: tuple):
        return input_shape


class SigmoidLayer(ActivationLayer):
    def _output(self, input):
        return T.nnet.sigmoid(input)


class ReLULayer(ActivationLayer):
    def _output(self, input):
        return T.clip(input, 0, np.infty)


class TanHLayer(ActivationLayer):
    def _output(self, input):
        return T.tanh(input)


class SoftmaxLayer(ActivationLayer):
    def _reshape_dims(self):
        return 2

    def _output(self, input):
        return T.nnet.softmax(input)

# --------------------------- Utility Layer -----------------------------------


class RGB2BGRLayer(Layer):
    def _output(self, input):
        r, g, b = (input[:, i, :, :] for i in range(3))
        bgr = [c.dimshuffle(0, 'x', 1, 2) for c in [b, g, r]]
        return T.concatenate(bgr, axis=1)

    def output_shape(self, input_shape: tuple):
        assert input_shape[1] == 3
        assert len(input_shape) == 4
        return input_shape


class CropLayer(Layer):
    width = REQUIRED(int)
    height = REQUIRED(int)
    input_shape = REQUIRED(Shape())

    def _output(self, input):
        max_crop_w = w(self.input_shape) - self.width
        max_crop_h = h(self.input_shape) - self.height
        crop_w = random.randint(0, max_crop_w)
        crop_h = random.randint(0, max_crop_h)
        return input[:, :, crop_h:self.height+crop_h, crop_w:self.width+crop_w]

    def output_shape(self, input_shape: tuple):
        assert input_shape == self.input_shape
        return input_shape[:2] + (self.height, self.width)


class SubtractMeanLayer(Layer):
    mean = OPTIONAL(float)
    mean_file = OPTIONAL(str)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.mean is None and self.mean_file is None:
            raise config_error("Either mean or mean_file must be given.")
        if self.mean_file and self.mean:
            raise config_error("Both mean or mean_file are given.")

        if self.mean_file:
            print(self.mean_file)
            self.mean_from_file = np.load(self.mean_file)

    def output(self, input: 'Theano Expression'):
        if self.mean:
            return input - self.mean
        else:
            return input - self.mean_from_file

# ----------------------------- Connection ------------------------------------


class CONNECTIONS(ConfigField):
    """
    A simple Parser to allow shortcuts for connections.

    Example:
    .. testsetup:: *

        from bernet.layer import ConnectionsParser

    .. testcode::

        from bernet.layer import ConnectionsParser
        conn_parse = ConnectionsParser()
        print(conn_parse.construct("conv#1 -> relu#1")[0])

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

        Connection(from_name=conv#1, from_port=None, \
to_name=relu#1, to_port=None)

    """

    # `in_port` and `out_port` are optional
    # [in_port]layer_name[out_port]
    _layer_regex = re.compile('^\s*(\[{0}\])?{0}(\[{0}\])?\s*$'
                              .format('([^\]\s]+)'))

    def __init__(self, doc="", type_sig=""):
        super().__init__(doc=doc, type_sig=type_sig)

    def _parse_layer(self, string):
        match = self._layer_regex.match(string)
        if match is None:
            raise config_error(
                "Could not parse Connection `{:}`".format(string))
        grps = match.groups()
        return grps[1], grps[2], grps[4]

    def _parse_connection(self, start_token, end_token):
        _from = self._parse_layer(start_token)
        to = self._parse_layer(end_token)
        return Connection(from_name=_from[1],
                          from_port=_from[2],
                          to_name=to[1],
                          to_port=to[0])

    def _parse_connections(self, string):
        tokens = string.split("->")
        for start, end in zip(tokens[:-1], tokens[1:]):
            yield self._parse_connection(start, end)

    def construct(self, loader, node):
        if isinstance(node, ScalarNode):
            return list(self._parse_connections(node.value))
        elif isinstance(node, SequenceNode):
            ret_list = []
            for conn in node.value:
                if isinstance(conn, ScalarNode):
                    ret_list.extend(list(self._parse_connections(conn.value)))
                elif isinstance(conn, MappingNode):
                    ret_list.append(Connection.from_yaml(loader, conn))
                else:
                    raise config_error(
                        "Cannot parse connection from value `{:}`"
                        .format(conn.value), conn)
            return ret_list
        else:
            raise config_error(
                "Expected str or list, but got value `{:}`"
                .format(node.value), node)

    def assert_valid(self, value, node=None):
        REPEAT(Connection).assert_valid(value)

    def represent(self, dumper, value) -> 'node':
        return REPEAT(Connection).represent(dumper, value)

    def type_signature(self):
        return "list of :class:`.Connection`"

ANY_LAYER = TAGS({
    "Conv": ConvLayer,
    "InnerProduct": InnerProductLayer,
    "TanH": TanHLayer,
    "Softmax": SoftmaxLayer,
    "Sigmoid": SigmoidLayer,
    "ReLU": ReLULayer,
    "Pooling": PoolingLayer,
    "LRN": LRNLayer
})


def format_ports(ports):
    if len(ports) == 0:
        return "`None`"
    else:
        return ",".join(["`{:}`".format(p) for p in ports])


class Connection(ConfigObject):
    from_name = REQUIRED(str)
    from_port = OPTIONAL(str, default=None)
    to_name = REQUIRED(str)
    to_port = OPTIONAL(str, default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.from_layer = None
        self.to_layer = None

    def is_part(self, layer):
        """Returns `True` if `layer` is part of this connection. """
        return layer.name == self.from_name or layer.name == self.to_name

    def add_layer(self, layer):
        if self.from_name == layer.name:
            self.from_layer = layer
        if self.to_name == layer.name:
            self.to_layer = layer

    def _format_name_port(self, name, port):
        port_str = ''
        if port is not None:
            port_str = "[{}]".format(port)
        return "{}{}".format(name, port_str)

    def from_str(self):
        return self._format_name_port(self.from_name, self.from_port)

    def to_str(self):
        return self._format_name_port(self.to_name, self.to_port)

    @classmethod
    def create_from_layers(cls, from_layer: Layer, to_layer: Layer):
        con = Connection(from_name=from_layer.name, to_name=to_layer.name)
        con.from_layer = from_layer
        con.to_layer = to_layer
        return con
