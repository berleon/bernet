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


class ExampleLifeCycle(object):
    METHODS_TYPES = [
        ('input_ports', tuple, [], {}),
        ('output_ports', tuple, [], {}),
        ('output_shapes', dict, [], {}),
        ('clear_connections', type(None), [], {})
    ]
    METHODS_ONLY_AVAILABLE_IF_CONNECTED = [
        ('output_shapes', [], {}),
        ('clear_connections', [], {}),
        ('outputs', [{}], {})
    ]

    def __init__(self, layer: Layer, test: TestCase, input_shapes: dict=None):
        self.layer = layer
        self.t = test
        if input_shapes is None:
            input_shapes = {label: (1, 3, 16, 16)
                            for label in self.layer.input_ports()}

        self.input_shapes = input_shapes

    def simulate(self):
        self.test_types()
        self.t.assert_(not self.layer.connected())
        self.methods_throw_exception_if_not_connected()

        self.set_up_connection()
        self.t.assert_(self.layer.connected())
        self.test_types()
        self.t.assertRaises(KeyError, self.layer.outputs, {})

        self.fill_parameters()
        fn = self.compile_theano_fn()
        outputs = self.compute_outputs(fn)
        self.compare_output_shapes(outputs)

        self.t.assert_(self.layer.connected())
        self.layer.clear_connections()
        self.t.assert_(not self.layer.connected())

    def methods_throw_exception_if_not_connected(self):
        for method, args, kwargs in self.METHODS_ONLY_AVAILABLE_IF_CONNECTED:
            self.t.assertRaises(NotConnectedException,
                                getattr(self.layer, method),
                                *args, **kwargs)

    def test_types(self):
        method_unavailbe = list(map(lambda m: m[0],
                                    self.METHODS_ONLY_AVAILABLE_IF_CONNECTED))

        for method, tpe, args, kwargs in self.METHODS_TYPES:
            if method not in method_unavailbe:
                callabe = getattr(self.layer, method)
                self.t.assertEqual(type(callabe(*args, **kwargs)), tpe)

        if self.layer.connected():
            for port, shape in self.layer.output_shapes().items():
                self.t.assertEqual(type(shape), tuple, msg=port)

        for port, shape in self.layer.input_shapes.items():
            self.t.assertEqual(type(shape), tuple, msg=port)

    def set_up_connection(self):
        self.layer.set_input_shapes(self.input_shapes)

    def fill_parameters(self):
        pass

    def compile_theano_fn(self):
        inputs = {label: theano.shared(GaussianFiller().fill(shp))
                  for label, shp in self.input_shapes.items()}

        sym_outputs = self.layer.outputs(inputs)
        # make sure the theano outputs have a deterministic order.
        # They are sorted by output_ports.
        sorted_outs = sorted(sym_outputs.items(), key=lambda pair: pair[0])
        return theano.function([], list(map(lambda p: p[1], sorted_outs)))

    def compute_outputs(self, callabe):
        raw_output = callabe()
        return dict(zip(sorted(self.layer.output_ports()), raw_output))

    def compare_output_shapes(self, outputs: dict):
        for port, out in outputs.items():
            self.t.assertEqual(out.shape, self.layer.output_shapes()[port],
                               self.input_shapes)


class WithParameterExampleLifeCycle(ExampleLifeCycle):
    METHODS_TYPES = ExampleLifeCycle.METHODS_TYPES + [
        ('fill_parameters', type(None), [], {}),
        ('parameter_shape', tuple, ['foo'], {}),
    ]

    METHODS_ONLY_AVAILABLE_IF_CONNECTED = \
        ExampleLifeCycle.METHODS_ONLY_AVAILABLE_IF_CONNECTED + [
            ('fill_parameters', [], {}),
            ('parameter_shape', ['foo'], {}),
            ('loss', [], {}),
        ]

    def fill_parameters(self):
        self.layer.fill_parameters()


class TestLayer(TestCase):
    def setUp(self):
        #                        just some random type to satisfy EITHER
        self.layer = Layer(type="InnerProduct",
                           name="test_layer")

    def test_base_function(self):
        l = self.layer
        # default labels are "in" and "out"
        self.assertTupleEqual(l.input_ports(), ("in",))
        self.assertTupleEqual(l.output_ports(), ("out",))

        # default state is to be not connected.
        self.assertFalse(l._connected)
        # All this functions are only callable, if the layer is connected.
        self.assertRaises(NotConnectedException, l.output_shapes)
        self.assertRaises(NotConnectedException, l.clear_connections)
        self.assertRaises(NotConnectedException, l.outputs, {})

        # setup the connection
        l.set_input_shapes({"in": (2, 3, 16, 16)})
        # Now we should be connected
        self.assertTrue(l._connected)

        self.assertRaises(KeyError, l.outputs, {})
        self.assertNotImplemented(l.outputs, {"in": []})
        self.assertNotImplemented(l.output_shapes)

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
                ConvLayer(
                    name=name,
                    type="Conv",
                    num_feature_maps=p[fm],
                    kernel_h=p[kh],
                    kernel_w=p[kw],
                    stride_h=p[sh],
                    stride_v=p[sv],
                    border_mode=p[bm],
                    weight=Parameter(name=name + "#weight"),
                    bias=Parameter(name=name + "#bias"),
                )
            )
        self.simple_conv_layer = self.conv_layers[0]

    def test_init(self):
        conv = ConvLayer(
            name="conv#test_init",
            type="Conv",
            num_feature_maps=20,
            kernel_w=5,
            kernel_h=5,
            weight=Parameter(name="weight"),
            bias=Parameter(name="weight")
        )
        self.assertEqual(conv.name, "conv#test_init")
        self.assertEqual(conv.type, "Conv")
        self.assertEqual(conv.num_feature_maps, 20)
        self.assertEqual(conv.kernel_w, 5)
        self.assertEqual(conv.kernel_h, 5)

    def test_parameter_shape(self):
        conv = ConvLayer(
            name="conv#test",
            type="Conv",
            num_feature_maps=20,
            kernel_w=5,
            kernel_h=5,
            weight=Parameter(name="conv#test#weight"),
            bias=Parameter(name="conv#test#bias")
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
            conv.output_shape(),
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
                expected_output_shape = conv.output_shape()

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
            life_cycle = WithParameterExampleLifeCycle(
                l, self,  input_shapes={"in": (1, 3, 32, 32)})
            life_cycle.simulate()


def create_layer(layer_class, **kwargs):
        return layer_class(name=layer_class.__name__ + "_test",
                           type=layer_type(layer_class), **kwargs)


class TestActivationLayers(TestCase):
    def setUp(self):
        self.layer_classes = [
            SigmoidLayer,
            TanHLayer,
            ReLULayer,
            SoftmaxLayer,
        ]

    def test_simple_computation(self):
        dummy = DummyDataLayer(name="dummy", type="DummyData",
                               shape=(1, 3, 16, 16))
        for layer_class in self.layer_classes:
            layer = create_layer(layer_class)

            layer.set_input_shape(dummy.output_shapes()["out"])
            dummy_out = dummy.outputs({})["out"]
            out = layer.output(dummy_out)
            f = theano.function([], [dummy_out, out])
            dummpy_out_real, out_real = f()
            self.assertEqual(dummpy_out_real.shape,
                             dummy.output_shapes()["out"])
            self.assertEqual(out_real.shape, layer.output_shape())

    def test_life_cycle(self):
        for layer_cls in self.layer_classes:
            life_cycle = ExampleLifeCycle(create_layer(layer_cls), self)
            life_cycle.simulate()


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

                ExampleLifeCycle(
                    layer, self,  input_shapes={"in": input_shape}
                ).simulate()


class TestInnerProductLayer(TestCase):
    def test_life_cycle(self):
        layer = InnerProductLayer(name="innerprod", type="InnerProduct",
                                  n_units=20,
                                  weight=Parameter(name="weight"),
                                  bias=Parameter(name="bias"))

        WithParameterExampleLifeCycle(layer, self).simulate()


class TestConcatLayer(TestCase):
    def test_concat_layer(self):
        join = ConcatLayer(name="join", in_ports=["foo", "bar"], axis=2)
        in_shape = (1, 3, 20, 20)
        join.set_input_shapes({"foo": in_shape, "bar": in_shape})

        self.assertTupleEqual(join.output_shapes()["out"],
                              (1, 3, 40, 20))

        rand_foo = np.random.sample(in_shape)
        rand_bar = np.random.sample(in_shape)
        outs = join.outputs({"foo": theano.shared(rand_foo),
                             "bar": theano.shared(rand_bar)})

        f = theano.function([], [outs["out"]])
        assert_array_equal(f()[0], np.concatenate([rand_foo, rand_bar], axis=2))


class TestConnectionParser(TestCase):
    def test_parse_layer(self):
        con_parse = ConnectionsParser()
        con_parse.ctx = InitContext(raise_exceptions=True)
        self.assertRaises(ConfigException, con_parse._parse_layer, "")
        self.assertRaises(ConfigException, con_parse._parse_layer, "ba dfa df")
        self.assertRaises(ConfigException, con_parse._parse_layer,
                          "[in] layer [out]")
        self.assertRaises(ConfigException, con_parse._parse_layer,
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
        con_parse = ConnectionsParser()
        con_parse.ctx = InitContext(raise_exceptions=True)

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
        con_parse = ConnectionsParser()
        con_parse.ctx = InitContext(raise_exceptions=True)
        self.assertListEqual(
            con_parse.construct("layer1 -> layer2 -> layer3"),
            [Connection(from_name="layer1", to_name="layer2"),
             Connection(from_name="layer2", to_name="layer3")]
        )

        self.assertListEqual(
            con_parse.construct(
                "layer1[45] -> [in]layer2[out2] -> layer3[bla]"),
            [
                Connection(from_name="layer1", from_port="45",
                           to_name="layer2", to_port="in"),
                Connection(from_name="layer2", from_port="out2",
                           to_name="layer3")
            ]
        )

        self.assertEqual(con_parse.construct(
            [Connection(from_name="layer1", to_name="layer2"),
             {'from_name': 'layer2', 'to_name': 'layer3'}]),
            [Connection(from_name="layer1", to_name="layer2"),
             Connection(from_name="layer2", to_name="layer3")])

        self.assertRaises(ConfigException, con_parse.construct({}))
        self.assertRaises(ConfigException, con_parse.construct(
            Connection(from_name="layer1", to_name="layer2")))
