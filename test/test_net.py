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

from bernet.net import FeedForwardNet
from bernet.layer import ConvLayer, SoftmaxLayer, TanHLayer, \
    InnerProductLayer, Parameter
from bernet.config import load
from bernet.utils import size, sha256_of_file


theano.config.mode = "FAST_COMPILE"


class TestFeedForwardNet(TestCase):
    def setUp(self):
        self.one_layer_net = FeedForwardNet(
            name="test", input_shape=(1, 3, 16, 16),
            layers=[TanHLayer(name="tanh")])

        self.two_layer_net = FeedForwardNet(
            name="two_layer_net", input_shape=(1, 3, 16, 16),
            layers=[TanHLayer(name="tanh"),
                    SoftmaxLayer(name="softmax", source="tanh")])

        self.innerprod_net = FeedForwardNet(
            name="innerprod_net", input_shape=(1, 3, 16, 16),
            layers=[
                InnerProductLayer(name="ip#1", n_units=256,
                                  input_shape=(1, 3*16*16),
                                  ),
                TanHLayer(name="tanh#1", source="ip#1"),
                InnerProductLayer(name="ip#2", n_units=10, source="tanh#1",
                                  input_shape=(1, 256)),
                SoftmaxLayer(name="softmax#1", source="ip#2"),
                ]
        )

    def parameters_not_changing(self, net):
        for p1, p2 in zip(net.parameters_as_shared(),
                          net.parameters_as_shared()):
            self.assertEqual(p1, p2)

    @unittest.skip
    def test_forward(self):
        net = self.simple_network
        input_shape = (20, 20)
        outs = net.forward(np.random.sample(input_shape))
        self.assertEqual(size(outs.shape), size(input_shape))

    def test_check_sha256sum(self):
        with tempfile.NamedTemporaryFile("w+b") as f:
            random = np.random.sample((200, 200))
            np.savez(f, rand=random)
            f.seek(0)
            self.assertRaises(ValueError, FeedForwardNet,
                              name="test_net", data_url="file://" + f.name,
                              input_shape=(1, 1),
                              data_sha256="wrong",
                              layers=[], connections=[])

    def test_constructor(self):
        self.assertEqual(type(self.one_layer_net), FeedForwardNet)

        self.assertEqual(type(self.two_layer_net), FeedForwardNet)
        self.assertEqual(self.two_layer_net.input_layer.name, "tanh")

    def test_connections(self):
        net = self.two_layer_net
        self.assert_(net.is_connected(net.input_layer, "softmax"))

        net = self.innerprod_net
        self.assertEqual(net.input_layer.name, "ip#1")

        self.assert_(not net.is_connected("ip#1", "softmax#1"))
        self.assert_(not net.is_connected("softmax", "ip#1"))
        self.assert_(net.is_connected("ip#1", "tanh#1"))
        self.assert_(net.is_connected("tanh#1", "ip#2"))
        self.assert_(net.is_connected("ip#2", "softmax#1"))
        self.assert_(not net.is_connected("softmax#1", "ip#2"))

    def test_shape_info(self):
        info = self.innerprod_net.shape_info()
        self.assertDictEqual(info['ip#1'], {
            'input_shape': (1, 3, 16, 16),
            'output_shape': (1, 256),
            'params': {
                'ip#1_weight': (256, 768)
            }
        })

        self.assertDictEqual(info['tanh#1'], {
            'input_shape': (1, 256),
            'output_shape': (1, 256)
        })

        self.assertDictEqual(info['ip#2'], {
            'input_shape': (1, 256),
            'output_shape': (1, 10),
            'params': {
                'ip#2_weight': (10, 256)
            }
        })

    def test_load_shallow_net(self):
        _dir = os.path.dirname(os.path.realpath(__file__))

        with open(_dir + "/../models/shallow-net.yaml") as f:
            net = load(FeedForwardNet, f)
            self.assertEqual(net.input_layer.name, 'ip#1')
            self.assertEqual(net.output_layer.name, 'softmax#1')
            out = net.forward(np.random.sample((64, 1, 28, 28)))
            self.assertTupleEqual(out.shape, (64, 10))

    def test_loading_parameters_from_file(self):
        params = {
            'ip1_weight': (20, 20),
            'ip1_bias': (20, ),
            'conv1_weight': (3, 2, 5, 5),
            'conv1_bias': (3,),
            }
        # generate data and save it to a npz file
        params_data = {p: np.random.sample(shape)
                       for p, shape in params.items()}
        with tempfile.NamedTemporaryFile("w+b") as f:
            np.savez_compressed(f, **params_data)
            f.flush()
            sha256sum = sha256_of_file(f)
            net = FeedForwardNet(
                name="net_test_loading",
                input_shape=(1, 3, 1, 20),
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
                        input_shape=(1, 3*20*20)
                    ),
                    ConvLayer(
                        name='conv1',
                        source='ip1',
                        num_feature_maps=3,
                        kernel_w=5,
                        kernel_h=5,
                        input_shape=(1, 2, 10, 20),
                        weight=Parameter(
                            name='conv1_weight',
                            shape=params['conv1_weight']
                        ),
                        bias=Parameter(
                            name='conv1_bias',
                            shape=params['conv1_bias']
                        ),
                    )
                ],
                data_url='file://' + f.name,
                data_sha256=sha256sum)
            for param_name, data in params_data.items():
                param = net.get_parameter(param_name)
                self.assert_(np.all(param.tensor == data))
