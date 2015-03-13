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
import os
import tempfile

from unittest import TestCase
import unittest

import numpy as np
import theano
import theano.tensor as T
import re

import bernet
from bernet.net import Network
from bernet.layer import ConvLayer, SoftmaxLayer, ReLULayer, TanHLayer, \
    Connection, InnerProductLayer, ConcatLayer, Parameter
from bernet.config import ConfigException
from bernet.utils import size, sha256_file


theano.config.mode = "FAST_COMPILE"


class TestNetwork(TestCase):
    def setUp(self):
        self.relu = ReLULayer(name="relu")
        self.softmax = SoftmaxLayer(name="softmax")
        self.tanh = TanHLayer(name="tanh")
        self.con_relu_softmax = Connection(from_name="relu", to_name="softmax")
        self.con_softmax_tanh = Connection(from_name="softmax", to_name="tanh")
        self.simple_network = Network(
            name="test",
            layers=[self.relu, self.softmax, self.tanh],
            connections=[self.con_relu_softmax, self.con_softmax_tanh]
        )

        self.complex_layers = {
            "conv": ConvLayer(
                name="conv#1",
                weight=Parameter(name="conv#1#weight"),
                bias=Parameter(name="conv#1#bias"),
                kernel_w=4,
                kernel_h=4,
                num_feature_maps=5,
                input_shape=(1, 3, 20, 20),
            ),
            "relu#1": ReLULayer(name="relu#1"),
            "join": ConcatLayer(name="join#1",
                                in_ports=["from_relu", "from_net_in"],
                                axis=2),
            "innerprod": InnerProductLayer(
                name="innerprod#1",
                n_units=64,
                weight=Parameter(name="conv#1#weight"),
                bias=Parameter(name="conv#1#bias"),
                input_shape=(1, 5, 37, 17),
            ),
            "relu#2": ReLULayer(name="relu#2"),
            "softmax": SoftmaxLayer(name="softmax#1"),
        }
        self.complex_connections = [
            Connection(from_name="conv#1", to_name="relu#1"),
            Connection(from_name="relu#1", to_name="join#1",
                       to_port="from_relu"),
            Connection(from_name="join#1", to_name="innerprod#1"),
            Connection(from_name="innerprod#1", to_name="relu#2"),
            Connection(from_name="relu#2", to_name="softmax#1"),
        ]
        self.complex_net = Network(
            name="test_complex",
            layers=self.complex_layers.values(),
            connections=self.complex_connections
        )

    def test_get_data(self):
        with tempfile.NamedTemporaryFile("w+b") as f:
            random = np.random.sample((200, 200))
            np.savez(f, rand=random)
            f.seek(0)
            nn = Network(
                name="test_net",
                data_url="file://" + f.name,
                data_sha256=sha256_file(f)
            )

            self.assert_(np.all(nn.data["rand"] == random))

    def test_connections_to_from(self):
        net = self.simple_network
        self.assertListEqual(net._connections_to(self.relu), [])
        self.assertListEqual(net._connections_from(self.relu),
                             [self.con_relu_softmax])

        self.assertListEqual(net._connections_to(self.softmax),
                             [self.con_relu_softmax])

        self.assertListEqual(net._connections_from(self.softmax),
                             [self.con_softmax_tanh])

        self.assertListEqual(net._connections_to(self.tanh),
                             [self.con_softmax_tanh])

        self.assertListEqual(net._connections_from(self.tanh), [])

    def test_setup_connections(self):
        net = self.simple_network
        self.assertDictEqual(net.layer_free_in_ports, {"relu": ["in"]})
        self.assertDictEqual(net.layer_free_out_ports, {"tanh": ["out"]})

    def test_multiple_taken_input_port_throws_exception(self):
        self.assertRaisesRegexp(
            ConfigException,
            re.escape("Layer `softmax` has multiple connections for input port"
                      " `in`. The connections are from: "
                      "`relu[out]`, `tanh[out]`."),
            Network,
            name="test",
            layers=[self.relu, self.softmax, self.tanh],
            connections=[Connection(from_name="relu", to_name="softmax"),
                         Connection(from_name="tanh", to_name="softmax")]
        )

    def test_output_simple(self):
        net = self.simple_network
        input_shape = (20, 20)
        x = T.ones(input_shape)
        outs = net.layer_outputs({"relu": {"in": x}})

        f = theano.function([], [outs["tanh"]["out"]])
        out = f()
        self.assertEqual(out[0].shape, input_shape)

    def test_output_complex(self):
        net = self.complex_net
        input_shape = (1, 3, 20, 20)

        # totally wrong inputs
        self.assertRaises(
            ConfigException,
            net.layer_outputs,
            {"relu": {"in": input_shape}}
        )

        # join#1[from_net_in] is missing
        self.assertRaises(
            ConfigException,
            net.layer_outputs,
            {"conv#1": {"in": input_shape}}
        )

        conv_input_shp = (1, 3, 20, 20)
        join_input_shp = (1, 5, 20, 17)

        outs = net.layer_outputs({
            "conv#1": {"in": T.ones(input_shape)},
            "join#1": {"from_net_in": T.ones(join_input_shp)}
        })
        join_fn = theano.function([], [outs["join#1"]["out"]])
        self.assertEqual(join_fn()[0].shape, (1, 5, 37, 17))
        f = theano.function([], [outs["softmax#1"]["out"]])
        self.assertTupleEqual(f()[0].shape, (1, 64))

    def test_parameters_not_changing(self):
        net = self.complex_net
        for p1, p2 in zip(net.parameters_as_shared(),
                          net.parameters_as_shared()):
            self.assertEqual(p1, p2)

    def test_loading_parameters_from_file(self):
        params = {
            'ip1_weight': (1, 1, 20, 20),
            'ip1_bias': (1, 1,  1, 20),
            'conv1_weight': (1, 3, 15, 15),
            'conv1_bias': (1, 3, 1, 15),
        }
        # generate data and save it to npz_file
        params_data = {p: np.random.uniform(shape)
                       for p, shape in params.items()}
        with tempfile.NamedTemporaryFile("w+b") as f:
            np.savez_compressed(f, **params_data)
            f.flush()
            sha256sum = sha256_file(f)
            net = Network(
                name="net_test_loading",
                layers=[
                    InnerProductLayer(
                        name='ip1',
                        weight=Parameter(
                            name='ip1_weight',
                            shape=params['ip1_weight']
                        ),
                        bias=Parameter(
                            name='ip1_bias',
                            shape=params['ip1_bias']
                        ),
                        n_units=400,
                        input_shape=(1, 3, 20, 20)
                    ),
                    ConvLayer(
                        name='conv1',
                        num_feature_maps=3,
                        kernel_w=5,
                        kernel_h=5,
                        weight=Parameter(
                            name='conv1_weight',
                            shape=params['conv1_weight']
                        ),
                        bias=Parameter(
                            name='conv1_bias',
                            shape=params['conv1_bias']

                        ),
                        input_shape=(2, 2, 14, 14)
                    )
                ],
                data_url='file://' + f.name,
                data_sha256=sha256sum)

            for param_name, data in params_data.items():
                name_exists = False
                for p in net.parameters():
                    if p.name == param_name:
                        name_exists = True
                        self.assert_(np.all(p.tensor == data))
                self.assert_(name_exists)

    @unittest.skip
    def test_forward(self):
        net = self.simple_network
        input_shape = (20, 20)
        outs = net.forward({"relu": {"in": np.random.sample(input_shape)}})
        self.assertEqual(size(outs["tanh"]["out"].shape), size(input_shape))

    def test_check_sha256sum(self):
        with tempfile.NamedTemporaryFile("w+b") as f:
            random = np.random.sample((200, 200))
            np.savez(f, rand=random)
            f.seek(0)
            self.assertRaises(ValueError, Network,
                              name="test_net", data_url="file://" + f.name,
                              data_sha256="wrong")

    def test_load_json(self):
        _dir = os.path.dirname(os.path.realpath(__file__))

        with open(_dir + "/../example/shallow-net.json") as f:
            net = Network.load_json(f)
            self.assertDictEqual(
                net.free_in_ports(),
                {'ip#1': ['in']}
            )
            self.assertDictEqual(
                net.free_out_ports(),
                {'softmax#1': ['out']}
            )

            out = net.forward(
                {'ip#1': {'in': np.random.sample((1, 1, 28, 28))}})

            self.assertTupleEqual(out["softmax#1"]["out"].shape, (1, 10))
