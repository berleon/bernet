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


def layer_life_cycle(test: TestCase, l: Layer, input_shapes=None):
    gauss = GaussianFiller()
    # default labels are "in" and "out"
    test.assertEqual(type(l.input_ports()), tuple)
    test.assertEqual(type(l.output_ports()), tuple)

    # default state is to be not connected.
    test.assertFalse(l._connected)
    # All this functions are only callable, if the layer is connected.
    test.assertRaises(NotConnectedException, l.loss)
    test.assertRaises(NotConnectedException, l.parameter_shape, "foo")
    test.assertRaises(NotConnectedException, l.output_shapes)
    test.assertRaises(NotConnectedException, l.output_shape, port="name")
    test.assertRaises(NotConnectedException, l.clear_connections)
    test.assertRaises(NotConnectedException, l.outputs, {})

    # setup the connection
    if input_shapes is None:
        input_shapes = {label: (1, 3, 16, 16) for label in l.input_ports()}

    l.set_input_shapes(input_shapes)
    # Now we should be connected
    test.assertTrue(l._connected)
    test.assertEqual(l.loss(), 0)
    test.assertRaises(KeyError, l.outputs, {})
    l.fill_parameters()

    inputs = {label: theano.shared(gauss.fill(shp))
              for label, shp in input_shapes.items()}

    sym_outputs = l.outputs(inputs)
    sorted_outs = sorted(sym_outputs.items(), key=lambda pair: pair[0])
    f = theano.function([], list(map(lambda p: p[1], sorted_outs)))

    real_outputs = f()
    for i, out in enumerate(real_outputs):
        port = sorted_outs[i][0]
        test.assertEqual(out.shape, l.output_shape(port), input_shapes)

    for p in l.parameters:
        test.assertEqual(type(l.parameter_shape(p)), tuple)

    for port in l.output_ports():
        test.assertEqual(type(l.output_shape(port=port)), tuple)

    test.assertEqual(type(l.output_shapes()), dict)

    test.assertTrue(l.connected())
    l.clear_connections()
    test.assertFalse(l.connected())


class TestLayer(TestCase):
    def setUp(self):
        #                        just some random type to satisfy EITHER
        self.layer = Layer(type="FCLayer",
                           name="test_layer")

    def test_base_function(self):
        l = self.layer
        # default labels are "in" and "out"
        self.assertTupleEqual(l.input_ports(), ("in",))
        self.assertTupleEqual(l.output_ports(), ("out",))

        # default state is to be not connected.
        self.assertFalse(l._connected)
        # All this functions are only callable, if the layer is connected.
        self.assertRaises(NotConnectedException, l.loss)
        self.assertRaises(NotConnectedException, l.parameter_shape, "foo")
        self.assertRaises(NotConnectedException, l.output_shape)
        self.assertRaises(NotConnectedException, l.clear_connections)
        self.assertRaises(NotConnectedException, l.outputs, {})

        # setup the connection
        l.set_input_shapes({"in": (2, 3, 16, 16)})
        # Now we should be connected
        self.assertTrue(l._connected)

        self.assertEqual(l.loss(), 0)
        self.assertRaises(KeyError, l.outputs, {})
        self.assertNotImplemented(l.outputs, {"in": []})
        self.assertNotImplemented(l.output_shape, "out")

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

    def test_max_dims(self):
        s = Shape(max_dims=2)
        self.assert_(s.valid([1, 4]))
        self.assert_(s.valid([4]))
        self.assert_(not s.valid([2, 3, 4]))
        self.assert_(not s.valid([2, 3, 4, 4]))
        self.assert_(not s.valid([5, 2, 3, 4, 4]))
        self.assert_(not s.valid([0, 0]))
        self.assert_(not s.valid([]))

    def test_dims(self):
        s = Shape(dims=2)
        self.assert_(s.valid([1, 4]))
        self.assert_(s.valid([9, 44]))
        self.assert_(not s.valid([4]))
        self.assert_(not s.valid([2, 3, 4]))
        self.assert_(not s.valid([2, 3, 4, 4]))
        self.assert_(not s.valid([5, 2, 3, 4, 4]))
        self.assert_(not s.valid([0, 0]))
        self.assert_(not s.valid([]))


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
        conv.set_input_shapes({"in": (32, 3, 200, 200)})
        # bs, c, h, w
        self.assertEqual(
            conv.parameter_shape(conv.weight),
            (20, 3, 5, 5)
        )
        conv.num_feature_maps = 32
        self.assertEqual(
            conv.parameter_shape(conv.weight),
            (32, 3, 5, 5)
        )

    def test_output_shape_manuel(self):
        conv = self.simple_conv_layer
        input_shape = (32, 3, 128, 128)
        conv.set_input_shapes({"in": input_shape})
        self.assertEqual(
            conv.output_shape("out"),
            (32, conv.num_feature_maps,
             128-conv.kernel_h+1, 128-conv.kernel_w+1))
        conv.clear_connections()

    def test_output_shape_auto(self):
        input_shapes = [(3, 1, 16, 16),
                        (1, 3, 32, 32),
                        (1, 2, 128, 128),
                        (1, 1, 21, 21)]
        for input_shape in input_shapes:
            for conv in self.conv_layers:
                conv.set_input_shapes({"in": input_shape})
                conv.fill_parameters()
                filter_shape = conv.filter_shape()
                self.assertEqual(bs(filter_shape), conv.num_feature_maps)
                self.assertEqual(chans(filter_shape), chans(input_shape))
                x = theano.shared(GaussianFiller().fill(input_shape))
                conv_out = conv.outputs({"in": x})
                f = theano.function([], conv_out["out"])
                expected_output_shape = conv.output_shape("out")

                self.assertEqual(f().shape,
                                 expected_output_shape,
                                 "input_shape was: {!s},\n"
                                 "ConvolutionLayer.name: {!r}"
                                 .format(input_shape, conv.name))
                conv.clear_connections()

    def test_outputs(self):
        conv = self.simple_conv_layer
        input_shape = (4, 3, 16, 16)
        conv.set_input_shapes({"in": input_shape})
        conv.fill_parameters()
        x = theano.shared(self.gauss.fill(input_shape))
        conv_out = conv.outputs({"in": x})

    def test_life_cycle(self):
        for l in self.conv_layers:
            layer_life_cycle(self, l, input_shapes={"in": (1, 3, 32, 32)})


def create_layer(layer_class, **kwargs):
        return layer_class(name=layer_class.__name__ + "_test",
                           type=layer_class.__name__, **kwargs)


class TestActivationLayers(TestCase):
    def setUp(self):
        self.layer_classes = [
            SigmoidLayer,
            TanHLayer,
            ReLULayer,
        ]

    def test_simple_computation(self):
        dummy = DummyDataLayer(name="dummy", type="DummyDataLayer",
                               shape=(1, 3, 16, 16))

        for layer_class in self.layer_classes:
            layer = create_layer(layer_class)

            layer.set_input_shapes({"in": dummy.output_shape("out")})
            dummy_out = dummy.outputs({})["out"]
            out = layer.outputs({"in": dummy_out})["out"]
            f = theano.function([], [dummy_out, out])
            dummpy_out_real, out_real = f()
            self.assertEqual(dummpy_out_real.shape, out_real.shape)

    def test_life_cycle(self):
        for layer_cls in self.layer_classes:
            layer_life_cycle(self, create_layer(layer_cls))


class TestPoolingLayer(TestCase):
    def test_life_cycle(self):
        layer_options = [
            ((2, 2), True),
            ((5, 5), False),
            ((4, 2), True),
            ((2, 3), False)
        ]
        input_shapes = [(1, 1, 5, 5), (3, 1, 16, 16),
                        (1, 1, 32, 32), (1, 1, 91, 31)]
        for input_shape in input_shapes:
            for poolsize, ignore_border in layer_options:
                layer = create_layer(PoolingLayer, poolsize=poolsize,
                                     ignore_border=ignore_border)
                layer_life_cycle(self, layer, input_shapes={"in": input_shape})
