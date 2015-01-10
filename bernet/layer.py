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
        self.tensor = kwargs.get("tensor", None)


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

    def parameter_shape(self, param: 'Parameter|str', input_shape=None):
        if type(param) == str:
            fitting_params = [p for p in self.parameters if p.name == param]
            assert len(fitting_params) == 1
            param = fitting_params[0]

        return self._parameter_shape(param, input_shape=input_shape)

    def _parameter_shape(self, param: Parameter, input_shape=None):
        raise NotImplementedError("Please use a subclass of Layer")

    def fill_parameters(self, input_shape=None):
        for p in self.parameters:
            p.tensor = p.filler.fill(
                self.parameter_shape(p, input_shape=input_shape))

    def set_parameter(self, name, nparray):
        self.parameters[name].tensor = nparray

    def loss(self):
        return 0

    def outputs(self, inputs, input_shapes=None):
        """
        :param input: dict of "name": symbolic input variable
        """
        for l in self.con_in_labels():
            assert l in inputs, "no input for label {!r}".format(l)

        out = self._outputs(inputs, input_shapes=input_shapes)
        if type(out) is not dict:
            assert len(self.con_out_labels()) == 1
            return {self.con_out_labels()[0]: out}
        else:
            return out

    def _outputs(self, inputs, input_shapes=None):
        raise NotImplementedError("Please use a subclass of Layer")

    def output_shape(self, con_out_label=None, input_shape=None):
        raise NotImplementedError("Please use a subclass of Layer")

    def con_in_labels(self):
        return ["in"]

    def con_out_labels(self):
        return ["out"]


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

    def _parameter_shape(self, param, input_shape=None):
        assert input_shape is not None
        if param == self.weight:
            return self.filter_shape(input_shape)
        if param == self.bias:
            return self.bias_shape(input_shape)

    def filter_shape(self, input_shape):
        return (self.num_feature_maps,
                chans(input_shape),
                self.kernel_h,
                self.kernel_w)

    def bias_shape(self, input_shape):
        shp = self.filter_shape(input_shape)
        return shp[0], shp[1], 1, 1

    def output_shape(self, con_out_label=None, input_shape=None):
        assert input_shape is not None
        batch_size = bs(input_shape)
        channels = self.num_feature_maps
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

        return batch_size, channels, height, width

    def _outputs(self, inputs, input_shapes=None):
        assert "in" in input_shapes
        assert "in" in inputs
        conv_out = T.nnet.conv2d(
            input=inputs["in"],
            filters=self.weight.tensor,
            subsample=(self.stride_h, self.stride_v),
            image_shape=input_shapes["in"],
            border_mode=self.border_mode,
            filter_shape=self.filter_shape(input_shapes["in"])
        )

        return conv_out + self.bias.tensor


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
