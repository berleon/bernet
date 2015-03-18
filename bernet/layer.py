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

import numpy as np
import re
import theano
from theano.ifelse import ifelse
import theano.tensor as T
from theano.tensor.nnet import conv2d
import theano.tensor.signal.downsample

from bernet.config import REQUIRED, OPTIONAL, EITHER, REPEAT, ConfigObject, \
    ConfigField

from bernet.utils import chans, bs, w, h

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

    def _construct(self, value, ctx):
        constr_list = super()._construct(value, ctx)
        if constr_list is None:
            ctx.error("No Shape given.")

        dims = len(constr_list)

        if self.dims is not None and self.dims != dims:
            ctx.error("A dimension of '{:}' is required. "
                      "Got '{:}' with a dimension of '{:}'"
                      .format(self.dims, constr_list, dims))

        if dims > self.max_dims:
            ctx.error("Shape '{:}' has a dimension of '{:}'. The maximum"
                      " allowed dimension is {:}."
                      .format(constr_list, len(constr_list), self.max_dims))

        if len(constr_list) == 0:
            ctx.error("Shape needs at least one element.")
        for elem in constr_list:
            if elem <= 0:
                ctx.error("Shape must be strictly positive.")

        return (1,) * (self.max_dims-len(constr_list)) + tuple(constr_list)


class Parameter(ConfigObject):
    name = REQUIRED(str)
    shape = REPEAT(int)
    filler = OPTIONAL(Filler, default=GaussianFiller(mean=0., std=1.))
    learning_rate = OPTIONAL(float, default=1.)
    weight_decay = OPTIONAL(float, default=1.)
    tensor = OPTIONAL(np.ndarray)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_property(self, name, value, ctx=None):
        super()._set_property(name, value, ctx=ctx)
        if name == "tensor" and self.tensor is not None:
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
    name = REQUIRED(str)
    type = EITHER("Conv", "InnerProduct", "TanH", "Softmax", "Sigmoid", "ReLU",
                  "DummyData", "Pooling", "Concat", "LRN")

    def __init__(self, **kwargs):
        """
            :param name: layer name
            :param options: LayerOption
            :param parameters: list of [..] TODO
            """
        kwargs["type"] = kwargs.get("type",
                                    self.__class__.__name__.rstrip("Layer"))
        super().__init__(**kwargs)
        self._connected = False

    def copy(self):
        raise NotImplementedError

    def output(self, input: 'Theano Expression'):
        """
        :param input: dict of {"<input_port>": symbolic tensor variable}
        """
        reshaped_input = self._reshape(input)
        return self._output(reshaped_input)

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

    @classmethod
    def construct_subclass(cls, value, ctx):
        if type(value) == dict:
            assert type(value['type']) == str
            allowed_types = Layer.__config_fields__["type"].fix_values
            if value['type'] in allowed_types:
                class_name = value['type'] + "Layer"
                return globals()[class_name](**value)
            else:
                ctx.error("type `{:}` is not in allowed types `{:}`"
                          .format(value['type'], allowed_types))
        elif issubclass(type(value), Layer):
            return value
        else:
            ctx.error("Cannot construct a subclass of Layer from value `{:}`"
                      .format(value))


def has_multiple_inputs(layer):
    return issubclass(type(layer), MultiInLayer)


def has_multiple_outputs(layer):
    return issubclass(type(layer), MultiOutLayer)


class MultiInLayer(Layer):
    def input_ports(self):
        raise NotImplementedError()

    def _expected_shape(self):
        return None

    def _reshape(self, inputs):
        reshaped = {}
        for port, input in inputs.items():
            expected = self._expected_shape()

            if expected is not None:
                reshaped[port] = ifelse(T.eq(expected, input.shape),
                                        input,
                                        input.reshape(expected))

            max_dims = self._reshape_dims()
            if input.ndim > max_dims:
                sym_shp = input.shape
                if max_dims == 2:
                    shp = (sym_shp[0], -1)
                if max_dims == 3:
                    shp = (sym_shp[0], sym_shp[1], -1)

                reshaped[port] = input.reshape(shp, ndim=max_dims)
            if port not in reshaped:
                reshaped[port] = input
        return reshaped

    def output(self, inputs: '{str: symbolic tensor}'):
        """
        :param input: dict of {"<input_port>": symbolic tensor variable}
        """
        for l in self.input_ports():
            if l not in inputs:
                raise KeyError("Expected a symbolic tensor variable for input"
                               " port `{!s}`.".format(l))

        reshaped_inputs = self._reshape(inputs)

        return self._output(reshaped_inputs)

    def _output(self, inputs: '{str: symbolic tensor}'):
        raise NotImplementedError()


class MultiOutLayer(Layer):
    def out_ports(self):
        raise NotImplementedError()

    def output(self, inputs: '{str: symbolic tensor}'):
        """
        :param input: dict of {"<input_port>": symbolic tensor variable}
        """
        for l in self.input_ports():
            if l not in inputs:
                raise KeyError("Expected a symbolic tensor variable for input"
                               " port `{!s}`.".format(l))
        reshaped_inputs = self._reshape(inputs)
        outs = self._output(reshaped_inputs)
        assert self.out_ports() in outs
        return outs

    def _output(self, inputs):
        raise NotImplementedError()


class WithParameterLayer(Layer):
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

    def loss(self):
        return T.as_tensor_variable(0)


class ConvLayer(WithParameterLayer):
    weight = REQUIRED(Parameter)
    bias = REQUIRED(Parameter)

    num_feature_maps = REQUIRED(int)

    # kernel width
    kernel_w = REQUIRED(int)
    # kernel height
    kernel_h = REQUIRED(int)

    # number of pixels the filter moves in vertical direction
    stride_v = OPTIONAL(int, default=1)
    # number of pixels the filter moves in horizontal direction
    stride_h = OPTIONAL(int, default=1)

    # `valid` or `full`. Default is `valid`. see scipy.signal.convolve2d
    border_mode = OPTIONAL(EITHER("valid", "full"), default="valid")

    input_shape = REQUIRED(Shape(dims=4), doc="""Shape of the input tensor""")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = [self.bias, self.weight]

    def _parameter_shape(self, param):
        if param == self.weight:
            return self.filter_shape()
        if param == self.bias:
            return self.bias_shape()

    def filter_shape(self):
        return (self.num_feature_maps,
                chans(self.input_shape),
                self.kernel_h,
                self.kernel_w)

    def bias_shape(self):
        shp = self.filter_shape()
        return shp[0],

    def _output(self, input):
        assert self.weight.tensor is not None
        conv_out = T.nnet.conv2d(
            input=input,
            image_shape=self.input_shape,
            filters=self.weight.shared,
            filter_shape=self.filter_shape(),
            subsample=(self.stride_h, self.stride_v),
            border_mode=self.border_mode,
        )
        return conv_out + self.bias.shared.dimshuffle('x', 0, 'x', 'x')


def _to_2d_shape(shp):
    return bs(shp), chans(shp)*h(shp)*w(shp)


def _to_Nd_shape(shp, n):
    return shp[0:n-1] + (-1, )


class InnerProductLayer(WithParameterLayer):
    n_units = REQUIRED(int)
    weight = REQUIRED(Parameter)
    bias = REQUIRED(Parameter)
    input_shape = REQUIRED(Shape(), doc="Shape of the input tensor.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = [self.weight, self.bias]

    def _parameter_shape(self, param: Parameter):
        shp = self.input_shape
        in_size = chans(shp)*h(shp)*w(shp)
        if param == self.weight:
            return in_size, self.n_units
        elif param == self.bias:
            return self.n_units,

    def _reshape_dims(self):
        return 2

    def _output(self, input):
        return T.dot(input, self.weight.shared) + self.bias.shared


class PoolingLayer(Layer):
    poolsize = REQUIRED(Shape(max_dims=2))
    ignore_border = OPTIONAL(bool, default=False)

    def _output(self, input):
        return theano.tensor.signal.downsample.max_pool_2d(
            input=input,
            ds=self.poolsize,
            ignore_border=self.ignore_border
        )


# ---------------------------- Source Layers ----------------------------------


class DataSourceLayer(Layer):
    shape = REQUIRED(Shape())

    def _output_shapes(self):
        return {"out": self.shape}

    def input_ports(self):
        return ()


class DummyDataLayer(DataSourceLayer):
    filler = OPTIONAL(Filler, default=GaussianFiller())

    def _output(self, input):
        return T.as_tensor_variable(self.filler.fill(self.shape), self.name)

# --------------------------- Util Layers -------------------------------------


class ConcatLayer(MultiInLayer):
    in_ports = REPEAT(str)
    axis = REQUIRED(int)

    def input_ports(self):
        return tuple(self.in_ports)

    def _output(self, inputs):
        tensors = [inputs[p] for p in self.in_ports]
        return T.concatenate(tensors, axis=self.axis)


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
        scaled = self.k + self.alpha * conved[:, start:end, :, :]
        scale = scaled ** self.beta
        return input / scale


# ------------------------- Activation Layers ---------------------------------


class ActivationLayer(Layer):
    pass


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


# ----------------------------- Connection ------------------------------------


class ConnectionsParser(ConfigField):
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
        self.ctx = None

    def _parse_layer(self, string):
        match = self._layer_regex.match(string)
        if match is None:
            self.ctx.error("Could not parse Connection `{:}`".format(string))
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

    def _construct(self, value, ctx):
        self.ctx = ctx
        if type(value) == str:
            return list(self._parse_connections(value))
        elif type(value) == list:
            ret_list = []
            for conn in value:
                if type(conn) == Connection:
                    ret_list.append(conn)
                elif type(conn) == dict:
                    ret_list.append(Connection(**conn))
                elif type(conn) == str:
                    return list(self._parse_connections(conn))
                else:
                    ctx.error("Cannot parse connection from value `{:}`"
                              .format(value))
            return ret_list
        else:
            ctx.error("Expected str or list, but got type `{:}`".format(value))

    def _type(self):
        return "list of :class:`.Connection`"


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
