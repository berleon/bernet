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
import tempfile

from unittest import TestCase

import numpy as np
import theano
import theano.tensor as T
import re

import bernet
from bernet.net import Network
from bernet.layer import ConvLayer, SoftmaxLayer, ReLULayer, TanHLayer, \
    Connection, InnerProductLayer, ConcatLayer, Parameter
from bernet.config import ConfigException
from bernet.utils import size


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
                num_feature_maps=5
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
                data_sha256=bernet.utils.sha256_file(f)
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
        net.set_input_shapes({"relu": {"in": input_shape}})
        outs = net.layer_outputs({"relu": {"in": T.ones(input_shape)}})

        self.assertEqual(size(net.get_layer("tanh").output_shape()),
                         size(input_shape))

    def test_output_complex(self):
        net = self.complex_net
        input_shape = (1, 3, 20, 20)

        # totally wrong inputs
        self.assertRaises(
            ConfigException,
            net.set_input_shapes,
            {"relu": {"in": input_shape}}
        )

        # join#1[from_net_in] is missing
        self.assertRaises(
            ConfigException,
            net.set_input_shapes,
            {"conv#1": {"in": input_shape}}
        )

        conv_input_shp = (1, 3, 20, 20)
        join_input_shp = (1, 5, 20, 17)

        net.set_input_shapes({
            "conv#1": {"in": conv_input_shp},
            "join#1": {"from_net_in": join_input_shp}
        })

        outs = net.layer_outputs({
            "conv#1": {"in": T.ones(input_shape)},
            "join#1": {"from_net_in": T.ones(join_input_shp)}
        })
        f = theano.function([], [outs["softmax#1"]["out"]])
        self.assertEqual(f()[0].shape,
                         net.get_layer("softmax#1").output_shape())

    def test_forward(self):
        net = self.simple_network
        input_shape = (20, 20)
        net.set_input_shapes({"relu": {"in": input_shape}})

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
