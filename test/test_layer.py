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

from unittest import TestCase
from numpy.testing import assert_array_equal

import theano

theano.config.mode = "FAST_COMPILE"

from bernet.layer import *
from bernet.config import ConfigException, InitContext


class StandardLayerTest(object):
    def test_properties_types(self):
        tensor_type = type(T.matrix())
        self.assertEqual(type(self.layer.input), tensor_type)
        self.assertEqual(len(self.layer.shape), 4)
        self.assertEqual(type(self.layer.params), list)
        self.assertEqual(type(self.layer.monitors), dict)


class TestLayer(TestCase):
    def setUp(self):
        self.layer = Layer(type="FCLayer", name="test_layer")

    def test_base_function(self):
        self.assertEqual(self.layer.loss(), 0)

        self.assertListEqual(self.layer.con_in_labels(), ["in"])
        self.assertListEqual(self.layer.con_out_labels(), ["out"])

        self.assertRaises(AssertionError, self.layer.outputs, {})

        self.assertNotImplemented(self.layer.outputs, {"in": []})
        self.assertNotImplemented(self.layer.output_shape)

    def assertNotImplemented(self, callable, *args, **kwargs):
        self.assertRaisesRegex(
            NotImplementedError,
            "Please use a subclass of Layer",
            callable,
            *args,
            **kwargs)


class TestFiller(TestCase):
    def test_gaussian(self):
        filler = GaussianFiller(mean=0., std=1.)
        arr = filler.fill((200, 200))

    def test_apply_sparsity_half(self):
        n = 100000
        sparsity = 0.5
        arr = np.ones(n)
        Filler._apply_sparsity(sparsity, arr)
        self.assertAlmostEqual(np.sum(arr == 0.) / n, sparsity, delta=0.05)

    def test_apply_sparsity_zero(self):
        n = 100000
        sparsity = 0.0
        arr = np.ones(n)
        Filler._apply_sparsity(sparsity, arr)
        self.assertEqual(np.sum(arr == 0.) / n, sparsity)

    def test_const(self):
        filler = ConstFiller(const_value=1.)
        arr = filler.fill((20, 20))
        assert_array_equal(arr, np.ones((20, 20)))

        sparsity = 0.5
        filler_sparsity = ConstFiller(const_value=1., sparsity=sparsity)
        self.assertEqual(filler_sparsity.sparsity, sparsity)
        shape = (200, 200)
        spare_arr = filler_sparsity.fill(shape)
        n_zeros = np.sum(spare_arr == 0.)
        n_total = shape[0] * shape[1]
        self.assertAlmostEqual(n_zeros / n_total, sparsity, delta=0.05)

    def test_uniform(self):
        filler = UniformFiller(low=-1., high=1.)
        arr = filler.fill((200, 200))
        self.assertGreater(arr.min(), -1.)
        self.assertLess(arr.max(), 1.)


class TestShape(TestCase):
    def test_construct(self):
        s = Shape()
        self.assert_(s.valid([1, 4]))
        self.assert_(s.valid([4, 4]))
        self.assert_(s.valid([2, 3, 4, 4]))
        self.assert_(not s.valid([5, 2, 3, 4, 4]))
        self.assert_(not s.valid([0, 0]))
        self.assert_(not s.valid([]))
        self.assertRaisesRegex(
            ConfigException,
            re.escape("Shape '[5, 2, 3, 4, 4]' has a dimension of '5'. "
                      "The maximum allowed dimension is 4."),
            s.construct,
            [5, 2, 3, 4, 4],
            ctx=InitContext(raise_exceptions=True))


class TestConvolutionLayer(TestCase):
    def setUp(self):
        self.gauss = GaussianFiller()
        fm = 0  # num_feature_maps
        kh = 1  # kernel_h
        kw = 2  # kernel_w
        sh = 3  # stride_h
        sv = 4  # stride_v
        bm = 5  # border_mode

        conv_nets_properties = [
            # Explanation of abbreviations:
            # fm, kh, kw,  sh, sv, bm
            (1,   5,  5,   1,  1, "valid"),
            (4,    5,  5,   1,  1, "valid"),
            (5,    4,  4,   2,  2, "valid"),
            (4,   15, 15,   3,  3, "valid"),
            (3,    7,  5,   4,  8, "valid"),
            (3,    7,  5,   4,  8, "full"),
            (2,    5,  5,   1,  1, "full"),
            (2,    5,  3,   3,  2, "full"),
        ]
        self.conv_layers = []
        for p in conv_nets_properties:
            name = "conv#test_fm{:}_kw{:}_kh{:}_sh{:}_sv{:}_bm_{:}"\
                .format(*p)
            self.conv_layers.append(
                ConvolutionLayer(
                    name=name,
                    type="ConvLayer",
                    num_feature_maps=p[fm],
                    kernel_h=p[kh],
                    kernel_w=p[kw],
                    stride_h=p[sh],
                    stride_v=p[sv],
                    border_mode=p[bm],
                    parameters=[
                        Parameter(name=name + "#weight", type="weight"),
                        Parameter(name=name + "#bias", type="bias")
                    ]
                )
            )
        self.simple_conv_layer = self.conv_layers[0]

    def test_init(self):
        conv = ConvolutionLayer(
            name="conv#test_init",
            type="ConvLayer",
            num_feature_maps=20,
            kernel_w=5,
            kernel_h=5
        )
        self.assertEqual(conv.name, "conv#test_init")
        self.assertEqual(conv.type, "ConvLayer")
        self.assertEqual(conv.num_feature_maps, 20)
        self.assertEqual(conv.kernel_w, 5)
        self.assertEqual(conv.kernel_h, 5)

    def test_parameter_shape(self):
        conv = ConvolutionLayer(
            name="conv#test",
            type="ConvLayer",
            num_feature_maps=20,
            kernel_w=5,
            kernel_h=5,
            parameters=[
                Parameter(name="conv#test#weight", type="weight")
            ]
        )
        # bs, c, h, w
        self.assertEqual(
            conv.parameter_shape(conv.weight, (32, 3, 200, 200)),
            (20, 3, 5, 5)
        )
        conv.num_feature_maps = 32
        self.assertEqual(
            conv.parameter_shape(conv.weight, (32, 3, 200, 200)),
            (32, 3, 5, 5)
        )

    def test_output_shape_manuel(self):
        conv = self.simple_conv_layer
        input_shape = (32, 3, 128, 128)
        self.assertEqual(
            conv.output_shape(input_shape=input_shape),
            (32, conv.num_feature_maps,
             128-conv.kernel_h+1, 128-conv.kernel_w+1))

    def test_output_shape_auto(self):
        input_shapes = [(3, 3, 16, 16),
                        (1, 3, 64, 64),
                        (1, 1, 128, 128),
                        (1, 1, 51, 51)]
        for input_shape in input_shapes:
            for conv in self.conv_layers:
                filter_shape = conv.filter_shape(input_shape)
                self.check_conv2d_shapes(
                    conv,
                    input_shape,
                    conv.output_shape(input_shape=input_shape),
                    filters=self.gauss.fill(filter_shape),
                    filter_shape=filter_shape,
                    border_mode=conv.border_mode,
                    subsample=(conv.stride_h, conv.stride_v)
                )

    def check_conv2d_shapes(self, conv_layer, input_shape,
                            expected_output_shape, **kwargs):
        x = theano.shared(GaussianFiller().fill(input_shape))
        conv_x = theano.tensor.nnet.conv2d(x, **kwargs)
        f = theano.function([], conv_x)

        self.assertEqual(f().shape, expected_output_shape,
                         "input_shape was: {!s},\n"
                         "ConvolutionLayer.name: {!r}"
                         .format(input_shape,
                                 conv_layer.name))

    def test_outputs(self):
        conv = self.simple_conv_layer
        input_shape = (4, 3, 16, 16)
        conv.fill_parameters(input_shape=input_shape)
        x = theano.shared(self.gauss.fill(input_shape))
        conv_out = conv.outputs({"in": x}, input_shapes={"in": input_shape})
