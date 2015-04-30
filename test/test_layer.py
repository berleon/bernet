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
from numpy.testing import assert_array_equal, assert_almost_equal

import theano
import yaml
from yaml.constructor import ConstructorError

from bernet.layer import *
from bernet.config import ConfigError
from test.test_config import ConfigFieldTestCase

theano.config.mode = "FAST_COMPILE"


class StandardLayerTest(object):
    def test_properties_types(self):
        tensor_type = type(T.matrix())
        self.assertEqual(type(self.layer.input), tensor_type)
        self.assertEqual(len(self.layer.shape), 4)
        self.assertEqual(type(self.layer.params), list)
        self.assertEqual(type(self.layer.monitors), dict)


class TestLayer(TestCase):
    def setUp(self):
        self.layer = Layer(name="test_layer")

    def test_base_function(self):
        l = self.layer
        self.assertNotImplemented(l.output, T.matrix("foo"))

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
        shape = (200, 200)
        arr = filler.fill(shape)
        self.assertTupleEqual(arr.shape, shape)

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


class TestShape(ConfigFieldTestCase):
    def test_construct(self):
        s = Shape()
        s.assert_valid([1, 4])
        s.assert_valid([4, 4])
        s.assert_valid([2, 3, 4, 4])

        self.assertNotValid(s, [5, 2, 3, 4, 4])
        self.assertNotValid(s, [0, 0])
        self.assertNotValid(s, [])
        self.assertRaisesRegex(
            ConfigError,
            re.escape("Shape '(5, 2, 3, 4, 4)' has a dimension of '5'. "
                      "The maximum allowed dimension is 4."),
            s.construct,
            self.loader,
            yaml.compose("[5, 2, 3, 4, 4]"))
        self.assertRaisesRegexp(
            ConfigError,
            re.escape("No Shape given."),
            s.assert_valid,
            None)

    def test_max_dims(self):
        s = Shape(max_dims=2)
        s.assert_valid([1, 4])
        s.assert_valid([4])
        self.assertNotValid(s, [2, 3, 4])
        self.assertNotValid(s, [2, 3, 4, 4])
        self.assertNotValid(s, [5, 2, 3, 4, 4])
        self.assertNotValid(s, [0, 0])
        self.assertNotValid(s, [])

    def test_dims(self):
        s = Shape(dims=2)
        s.assert_valid([1, 4])
        s.assert_valid([9, 44])
        self.assertNotValid(s, [4])
        self.assertNotValid(s, [2, 3, 4])
        self.assertNotValid(s, [2, 3, 4, 4])
        self.assertNotValid(s, [5, 2, 3, 4, 4])
        self.assertNotValid(s, [0, 0])
        self.assertNotValid(s, [])


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
                ConvLayer(
                    name=name,
                    num_feature_maps=p[fm],
                    kernel_h=p[kh],
                    kernel_w=p[kw],
                    stride_h=p[sh],
                    stride_v=p[sv],
                    border_mode=p[bm],
                    weight=Parameter(name=name + "#weight"),
                    bias=Parameter(name=name + "#bias"),
                    input_shape=(1, 3, 16, 16)
                )
            )
        self.simple_conv_layer = self.conv_layers[0]

    def test_init(self):
        conv = ConvLayer(
            name="conv#test_init",
            num_feature_maps=20,
            kernel_w=5,
            kernel_h=5,
            weight=Parameter(name="weight"),
            bias=Parameter(name="weight"),
            input_shape=(1, 3, 50, 50)
        )
        self.assertEqual(conv.name, "conv#test_init")
        self.assertEqual(conv.num_feature_maps, 20)
        self.assertEqual(conv.kernel_w, 5)
        self.assertEqual(conv.kernel_h, 5)

    def test_parameter_shape(self):
        conv = ConvLayer(
            name="conv#test",
            num_feature_maps=20,
            kernel_w=5,
            kernel_h=5,
            weight=Parameter(name="conv#test#weight"),
            bias=Parameter(name="conv#test#bias"),
            input_shape=(32, 3, 200, 200))

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

    def test_outputs(self):
        conv = self.simple_conv_layer
        input_shape = conv.input_shape
        conv.fill_parameters()
        x = theano.shared(self.gauss.fill(input_shape))
        conv_out = conv.output(x).eval()
        self.assertTupleEqual(conv_out.shape, (1, 1, 12, 12))
        self.assertTupleEqual(conv.output_shape(input_shape), (1, 1, 12, 12))

    def test_same_border_mode(self):
        conv = ConvLayer(
            name="conv#test",
            num_feature_maps=20,
            kernel_w=5,
            kernel_h=5,
            weight=Parameter(name="conv#test#weight"),
            bias=Parameter(name="conv#test#bias"),
            border_mode='same',
            input_shape=(1, 3, 20, 20))

        input_shape = conv.input_shape
        conv.fill_parameters()
        x = theano.shared(self.gauss.fill(input_shape))
        conv_out = conv.output(x).eval()
        self.assertTupleEqual(conv_out.shape, (1, 20, 20, 20))
        self.assertTupleEqual(conv.output_shape(input_shape), (1, 20, 20, 20))


def create_layer(layer_class, **kwargs):
        return layer_class(name=layer_class.__name__ + "_test", **kwargs)


class TestActivationLayers(TestCase):
    def setUp(self):
        self.layer_classes = [
            (SigmoidLayer, lambda x: 1./(1.+np.exp(-x))),
            (TanHLayer, lambda x: np.tanh(x)),
            (ReLULayer, lambda x: np.maximum(np.zeros_like(x), x))
        ]

    def test_simple_computation(self):
        shape = (1, 3, 16, 16)
        for layer_class, func in self.layer_classes:
            layer = create_layer(layer_class)
            dummy_data = np.random.sample(shape)
            out = layer.output(theano.shared(dummy_data)).eval()
            assert_almost_equal(out, func(dummy_data))
            self.assertTupleEqual(out.shape, layer.output_shape(shape))
            self.assertTupleEqual(layer.output_shape(shape), shape)


class TestSoftmaxLayer(TestCase):
    def test_softmax_layer(self):
        softmax = SoftmaxLayer(name="softmax")
        x = T.matrix('x')
        fn = theano.function([x], softmax.output(x))
        out = fn(np.random.random((200, 200)))
        np.testing.assert_almost_equal(out.sum(axis=1), np.ones(200,))


class TestPoolingLayer(TestCase):
    def test_pooling_layer(self):
        input_shapes = [(1, 1, 5, 5), (3, 1, 16, 16),
                        (1, 1, 32, 32), (1, 1, 91, 31)]

        layer = create_layer(PoolingLayer, poolsize=(2, 2),
                             ignore_border=True)
        for input_shape in input_shapes:
            input = theano.shared(np.random.sample(input_shape))
            out = layer.output(input)


class TestInnerProductLayer(TestCase):
    def test_inner_product_layer(self):
        layer = InnerProductLayer(name="innerprod",
                                  n_units=20,
                                  weight=Parameter(name="weight"),
                                  bias=Parameter(name="bias"),
                                  input_shape=(1, 3*28*28))
        layer.fill_parameters()
        np_input = np.random.sample((1, 3, 28, 28))
        reshaped = np.reshape(np_input, (1, -1))
        np_output = np.dot(reshaped, layer.weight.tensor.T) + layer.bias.tensor
        input = theano.shared(np_input)
        np.testing.assert_almost_equal(
            layer.output(input).eval(),
            np_output)


class TestLRNLayer(TestCase):
    def test_lrn_layer(self):
        n = 5
        # big value for alpha so mistakes involving alpha show up strongly
        alpha = 1.5
        beta = 0.75
        k = 3.

        def ground_truth_lrn(np_arr):
            # The Theano implementation of LRN requires some rather complex
            # reshaping and convolution. Here the LRN formular is written with
            # simple numpy code.
            half = n // 2
            nb_chans = chans(np_arr.shape)
            sqr = np_arr ** 2
            scale = np.zeros_like(np_arr)
            for ch in range(nb_chans):
                start = max(ch-half, 0)
                stop = min(ch+half+1, nb_chans)
                for i in range(start, stop):
                    scale[:, ch, :, :] += sqr[:, i, :, :]
            scale = k + (alpha/n)*scale
            return np_arr / scale**beta

        test_np = np.random.uniform(size=(1, 16, 32, 32))
        test_shared = theano.shared(test_np, 'test_shared')
        lrn = LRNLayer(name="lrn", n=n, alpha=alpha, k=k)
        out = lrn.output(test_shared)
        np.testing.assert_almost_equal(out.eval(), ground_truth_lrn(test_np))

    def test_output_shape(self):
        lrn = LRNLayer(name="lrn")
        input_shape = (1, 3, 16, 16)
        self.assertEqual(lrn.output_shape(input_shape), input_shape)


class TestCONNECTIONS(ConfigFieldTestCase):
    def test_parse_layer(self):
        con_parse = CONNECTIONS()
        self.assertRaises(ConfigError, con_parse._parse_layer, "")
        self.assertRaises(ConfigError, con_parse._parse_layer, "ba dfa df")
        self.assertRaises(ConfigError, con_parse._parse_layer,
                          "[in] layer [out]")
        self.assertRaises(ConfigError, con_parse._parse_layer,
                          "bla [in]layer[out]")

        self.assertTupleEqual(con_parse._parse_layer("conv"),
                              (None, "conv", None))

        self.assertTupleEqual(con_parse._parse_layer("[in]layer_name[out]"),
                              ("in", "layer_name", "out"))

        self.assertTupleEqual(con_parse._parse_layer("layer_name[foo]"),
                              (None, "layer_name", "foo"))

        self.assertTupleEqual(con_parse._parse_layer("[in]layer_name[out]"),
                              ("in", "layer_name", "out"))

        # layer names with special signs are permitted
        self.assertTupleEqual(con_parse._parse_layer("layer_name_123#$"),
                              (None, "layer_name_123#$", None))

    def test_parse_connection(self):
        con_parse = CONNECTIONS()

        self.assertEqual(
            con_parse._parse_connection("from_layer", "to_layer"),
            Connection(from_name="from_layer", to_name="to_layer"))

        self.assertEqual(con_parse._parse_connection("from_layer[out1]",
                                                     "[in]to_layer"),
                         Connection(
                             from_name="from_layer",
                             from_port="out1",
                             to_name="to_layer",
                             to_port="in"))

    def test_construct(self):
        con_parse = CONNECTIONS()
        node = ScalarNode(None, "layer1 -> layer2 -> layer3")
        self.assertListEqual(
            con_parse.construct(self.loader, node),
            [Connection(from_name="layer1", to_name="layer2"),
             Connection(from_name="layer2", to_name="layer3")]
        )

        node = ScalarNode(None,
                          "layer1[45] -> [in]layer2[out2] -> layer3[bla]")
        self.assertListEqual(
            con_parse.construct(self.loader, node),
            [
                Connection(from_name="layer1", from_port="45",
                           to_name="layer2", to_port="in"),
                Connection(from_name="layer2", from_port="out2",
                           to_name="layer3")
            ]
        )
        node = yaml.compose("[{from_name: layer1, to_name: layer2}, "
                            "{from_name: layer2, to_name: layer3}]")
        self.assertEqual(con_parse.construct(self.loader, node),
                         [Connection(from_name="layer1", to_name="layer2"),
                          Connection(from_name="layer2", to_name="layer3")])
