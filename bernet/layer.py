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
import theano.tensor as T

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


class Parameter(ConfigObject):
    name = REQUIRED(str)
    shape = REPEAT(int)
    filler = OPTIONAL(Filler, default=GaussianFiller(mean=0., std=1.))
    type = EITHER("weight", "bias")
    learning_rate = OPTIONAL(float, default=1.)
    weight_decay = OPTIONAL(float, default=1.)
    tensor = OPTIONAL(np.ndarray)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tensor = theano.shared(kwargs.get("tensor", None), name=self.name)


class NotConnectedException(Exception):
    pass


class Layer(ConfigObject):
    name = REQUIRED(str)
    type = EITHER("ConvLayer", "FCLayer", "TanhLayer", "SoftmaxLayer",
                  "ReLULayer")

    parameters = REPEAT(Parameter)

    def __init__(self, **kwargs):
        """
            :param name: layer name
            :param options: LayerOption
            :param parameters: list of [..] TODO
            """
        super().__init__(**kwargs)
        self._connected = False
        self.input_shapes = {}

    def input_ports(self):
        return "in",

    def output_ports(self):
        return "out",

    def connected(self):
        return self._connected

    def set_input_shapes(self, input_shapes: "{str: tuple}"):
        if not self._are_all_inputs_ports_satisfied_by(input_shapes.keys()):
            not_found = self._not_found_in_input_ports(input_shapes.keys())
            not_satisfied = self._not_satisfied_input_ports(
                input_shapes.keys())
            if len(not_satisfied) >= 1:
                raise ValueError("Labels {!s} are not satisfied by {!s}".
                                 format(not_satisfied, input_shapes.keys()))
            if len(not_found) >= 1:
                raise ValueError("Recieved input shapes for ports {!s}, but "
                                 "got no such ports"
                                 .format(not_found))

        self.input_shapes = input_shapes
        self._update_connection_status()

    def clear_connections(self):
        self._assert_connected()
        self._connected = False
        self.input_shapes = {}

    def copy(self):
        # TODO: checkout copying
        raise NotImplementedError

    def copy_disconnected(self):
        raise NotImplementedError

    def parameter_shape(self, param: 'Parameter|str'):
        self._assert_connected()
        if type(param) == str:
            fitting_params = [p for p in self.parameters if p.name == param]
            assert len(fitting_params) == 1
            param = fitting_params[0]

        return self._parameter_shape(param)

    def _parameter_shape(self, param: Parameter):
        raise NotImplementedError("Please use a subclass of Layer")

    def fill_parameters(self):
        for p in self.parameters:
            p.tensor = T.as_tensor_variable(
                p.filler.fill(self.parameter_shape(p)),
                p.name)

    def set_parameter(self, name, nparray):
        self.parameters[name].tensor = nparray

    def loss(self):
        self._assert_connected()
        return 0

    def outputs(self, inputs: '{str: symbolic tensor}'):
        """
        :param input: dict of {"<input_port>": symbolic tensor variable}
        """
        self._assert_connected()
        for l in self.input_ports():
            if l not in inputs:
                raise KeyError("Expected a symbolic tensor variable for input"
                               " port `{!s}`.".format(l))
        out = self._outputs(inputs)
        if type(out) is not dict:
            assert len(self.output_ports()) == 1
            return {self.output_ports()[0]: out}
        else:
            return out

    def _outputs(self, inputs):
        raise NotImplementedError("Please use a subclass of Layer")

    def output_shapes(self):
        self._assert_connected()
        return self._output_shapes()

    def _output_shapes(self):
        raise NotImplementedError("Please use a subclass of Layer")

    def output_shape(self, port=None):
        if port is None:
            assert len(self.output_ports()) == 1,\
                "None is only acceptable if there is " \
                "exactly one output channel."
            port = self.output_ports()[0]
        return self.output_shapes()[port]

    def _not_satisfied_input_ports(self, given_ports):
        return list(filter(lambda l: l not in given_ports,
                           self.input_ports()))

    def _not_found_in_input_ports(self, given_ports):
        return list(filter(
            lambda g: g not in self.input_ports(),
            given_ports
        ))

    def _are_all_inputs_ports_satisfied_by(self, given_ports):
        return len(self._not_satisfied_input_ports(given_ports)) == 0 \
            and len(self._not_found_in_input_ports(given_ports)) == 0

    def _update_connection_status(self):
        self._connected = self._are_all_inputs_ports_satisfied_by(
            self.input_shapes.keys())

    def _assert_connected(self):
        if not self._connected:
            raise NotConnectedException(
                "{:} is not connected. The following channels are not"
                " connected: {:}"
                .format(type(self).__name__,
                        self._not_connected_input_ports()))

    def _not_connected_input_ports(self):
        return list(filter(lambda p: p not in self.input_shapes,
                           self.input_ports()))


class ConvolutionLayer(Layer):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for p in self.parameters:
            if p.type == "weight":
                self.weight = p
            elif p.type == "bias":
                self.bias = p

    def _parameter_shape(self, param):
        if param == self.weight:
            return self.filter_shape()
        if param == self.bias:
            return self.bias_shape()

    def filter_shape(self):
        return (self.num_feature_maps,
                chans(self.input_shapes["in"]),
                self.kernel_h,
                self.kernel_w)

    def bias_shape(self):
        shp = self.filter_shape()
        return shp[0],

    def _output_shapes(self):
        input_shape = self.input_shapes["in"]
        batch_size = bs(input_shape)
        channels = self.num_feature_maps  # * chans(input_shape)
        miss_h = self.kernel_h - 1
        miss_w = self.kernel_w - 1

        # maximum possible step, if stride_X = 1.
        if self.border_mode == "valid":
            max_steps_h = h(input_shape) - miss_h
            max_steps_w = w(input_shape) - miss_w
        elif self.border_mode == "full":
            max_steps_h = h(input_shape) + miss_h
            max_steps_w = w(input_shape) + miss_w

        height = max_steps_h // self.stride_h
        width = max_steps_w // self.stride_v

        if max_steps_h % self.stride_h != 0:
            height += 1
        if max_steps_w % self.stride_v != 0:
            width += 1

        return {"out": (batch_size, channels, height, width)}

    def _outputs(self, inputs):
        assert self.weight.tensor is not None

        conv_out = T.nnet.conv2d(
            input=inputs["in"],
            image_shape=self.input_shapes["in"],
            filters=self.weight.tensor,
            filter_shape=self.filter_shape(),
            subsample=(self.stride_h, self.stride_v),
            border_mode=self.border_mode,
        )

        return conv_out + self.bias.tensor.dimshuffle('x', 0, 'x', 'x')


class Shape(REPEAT):
    def __init__(self, **kwargs):
        super().__init__(int)

    def _construct(self, value, ctx):
        constr_list = super()._construct(value, ctx)
        if len(constr_list) > 4:
            ctx.error("Shape '{:}' has a dimension of '{:}'. The maximum"
                      " allowed dimension is 4."
                      .format(constr_list, len(constr_list)))

        if len(constr_list) == 0:
            ctx.error("Shape needs at least one element.")
        for elem in constr_list:
            if elem <= 0:
                ctx.error("Shape must be strictly positive.")

        return (1,) * (4-len(constr_list)) + tuple(constr_list)
