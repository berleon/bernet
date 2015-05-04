# Copyright 2014 Leon Sixt
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
from collections import OrderedDict
import os

import numpy as np
import theano
import theano.tensor as T

from bernet import utils
from bernet.config import REQUIRED, OPTIONAL, ConfigObject, REPEAT, \
    ConfigError, ENUM
from bernet.layer import ParameterLayer, ANY_LAYER, Shape, Connection
from bernet.loss import NegativeLogLikelihood
from bernet.utils import symbolic_tensor_from_shape


class FeedForwardNet(ConfigObject):
    """
    Network connects multiple :class:`.Layer` together.

    The Network is a graph where the nodes are :class:`.Layer` and the edges
    are connections :class:`.Connection`.

    .. graphviz::

        digraph foo {
           "TODO:" -> "Network" -> "to" -> "dot";
        }

    Create a Network
    ----------------

    The default way to create a Network is to load it form a network json file.
    .. code-block:: python

        net = Network.from_json("my_network.json")

    """

    name = REQUIRED(str, doc="name of the Network")

    description = OPTIONAL(str, doc="a short description of the Network")

    data_url = OPTIONAL(str, doc="A URL of a numpy array file. "
                                 "Use this to load saved layer parameters.")

    data_sha256 = OPTIONAL(str, doc="The sha256 sum of the file given "
                                    "by `data_url`")

    input_shape = REQUIRED(Shape())

    batch_size = OPTIONAL(int, doc="Size of a minibatch", default=32)

    layers = REPEAT(ANY_LAYER(), doc="A list of :class:`.Layer`")

    loss = OPTIONAL(ENUM('NLL', 'MSE'),
                    default=NegativeLogLikelihood(), doc="")

    MODELS_DIR = os.path.expanduser("~/.bernet/")

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.data = {}
        if self.data_url is not None and self.data_sha256 is None:
            raise ConfigError("Field data_url requires data_sha256 to be set")
        if self.data_url is not None:
            file_name = os.path.join(self.MODELS_DIR,
                                     kwargs['name'] + "_parameters.npz")
            data_url = kwargs['data_url']
            data_sha256 = kwargs['data_sha256']
            self.data = self._get_data(file_name, data_url, data_sha256)
        self._setup_connections()
        self._setup_parameters()

    def _get_data(self, file_path, url, sha256_expected):
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        with open(file_path, "w+b") as f:
            utils.download(url, f)
            sha256_got = utils.sha256_of_file(f)
            if sha256_expected != sha256_got:
                raise ValueError("The given sha256sum {:} of is not equal to"
                                 " {:} from the url {:}"
                                 .format(sha256_expected, sha256_got, url))
            f.seek(0)
            npzfile = np.load(f)
            npz_dict = {n: np.asarray(npzfile[n], dtype=theano.config.floatX)
                        for n in npzfile.files}
            parameter_names = [p.name for p in self.parameters()]
            for name in npz_dict.keys():
                assert name in parameter_names, \
                    "Got parameter `{}` in file `{}`, but there is no " \
                    "parameter in the network with this name.\n" \
                    "Parameter names in file:    [{}]\n" \
                    "Parameter names in network: [{}]" \
                    .format(name, self.data_url,
                            ", ".join(sorted(npz_dict.keys())),
                            ", ".join(sorted(parameter_names)))
            return npz_dict

    def get_layer(self, name):
        """Return the layer with layer.name == `name`. If no such layer
        exists, None is returned."""
        for l in self.layers:
            if l.name == name:
                return l
        raise KeyError("Layer with name {} does not exists.".format(name))

    def parameters(self):
        params = []
        for layer in self.layers:
            if issubclass(type(layer), ParameterLayer):
                params.extend(layer.parameters)
        return params

    def parameters_as_shared(self):
        return [p.shared for p in self.parameters()]

    def shape_info(self):
        input_shape = self.input_shape
        info = {}
        for layer in self.layer_iter():
            output_shape = layer.output_shape(input_shape)

            info[layer.name] = {
                'output_shape': output_shape,
                'input_shape': input_shape,
            }
            if issubclass(type(layer), ParameterLayer):
                info[layer.name]['params'] = {}
                for param in layer.parameters:
                    info[layer.name]['params'][param.name] = \
                        layer.parameter_shape(param)

            input_shape = output_shape
        return info

    def get_parameter(self, name):
        for p in self.parameters():
            if p.name == name:
                return p
        raise KeyError("No parameter with name {}.".format(name))

    def _setup_input_layer(self):
        layers_without_a_source = [l for l in self.layers if l.source is None]
        assert len(layers_without_a_source) == 1, \
            "In a feed forward network there can only be one input layer, " \
            "But found multiple: {:}".format(", ".join(
                [l.name for l in layers_without_a_source]))

        self.input_layer = layers_without_a_source[0]

    def _setup_output_layer(self):
        used_as_source = {l.source for l in self.layers}
        names = {l.name for l in self.layers}
        not_used_as_source = names - used_as_source
        assert len(not_used_as_source) == 1, \
            "Multiple layers {} are output layers, but a feed forward " \
            "network only has a single output layer. ".format(
                ', '.join(not_used_as_source)
            )

        self.output_layer = self.get_layer(not_used_as_source.pop())

    def _setup_connections(self):
        self._setup_input_layer()
        self._setup_output_layer()

        self.connections = []
        self._connection_from_layer = {}
        self._connection_to_layer = {}
        layer_dict = {l.name: l for l in self.layers}
        for l in set(self.layers) - {self.input_layer}:
            from_layer = layer_dict[l.source]
            con = Connection.create_from_layers(from_layer, l)
            self._connection_from_layer[from_layer.name] = con
            self._connection_to_layer[l.name] = con
            self.connections.append(con)

    def layer_iter(self):
        layer = self.input_layer
        while True:
            yield layer
            if layer.name in self._connection_from_layer:
                next_layer = self._connection_from_layer[layer.name].to_layer
                layer = next_layer
            else:
                return

    def _setup_parameters(self):
        input_shape = self.input_shape
        for layer in self.layer_iter():
            self._setup_parameters_for_layer(layer)
            input_shape = layer.output_shape(input_shape)

    def _setup_parameters_for_layer(self, layer):
        if issubclass(type(layer), ParameterLayer):
            for param in layer.parameters:
                if self.data.get(param.name) is not None:
                    param.tensor = self.data[param.name]
                elif param.tensor is None:
                    layer.fill_parameter(param)

    def layer_outputs(self, input):
        next_input = input
        outputs = OrderedDict()
        for layer in self.layer_iter():
            output = layer.output(next_input)
            outputs[layer.name] = output
            next_input = output

        return outputs

    def forward(self, input):
        if not hasattr(self, '_forward_func'):
            x = symbolic_tensor_from_shape('x', self.input_shape)
            y = self.output(x)
            self._forward_func = theano.function([x], [y])

        return self._forward_func(input)[0]

    def minibatch_func(self, shared_input, updates=None):
        x = symbolic_tensor_from_shape('x', self.input_shape)
        begin = T.iscalar('minibatch_begin')
        end = T.iscalar('minibatch_end')
        y = self.output(x)
        func = theano.function([begin, end], [y],
                               givens={x: shared_input[begin:end, :]},
                               updates=updates)
        return lambda b, e: func(b, e)[0]

    def output(self, input: 'symbolic tensor'):
        return self.layer_outputs(input)[self.output_layer.name]

    def get_accuracy(self, net_out, labels):
        pred_labels = T.argmax(net_out, axis=1)
        return T.sum(T.eq(pred_labels, labels)) / labels.shape[0]

    def get_loss(self, out, labels):
        return self.loss.loss(out, labels)

    def is_connected(self, from_layer, to_layer):
        def str2layer(layer):
            if type(layer) is str:
                return self.get_layer(layer)
            else:
                return layer

        from_layer = str2layer(from_layer)
        to_layer = str2layer(to_layer)
        for c in self.connections:
            if c.from_layer == from_layer and c.to_layer == to_layer:
                return True
        return False
