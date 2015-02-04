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
from collections import defaultdict

import numpy as np
import theano

from bernet import utils
from bernet.config import REQUIRED, OPTIONAL, ConfigObject, REPEAT, SUBCLASS_OF
from bernet.layer import Layer, Connection, WithParameterLayer, format_ports
from bernet.utils import tensor_from_shape


class Network(ConfigObject):
    name = REQUIRED(str)
    description = OPTIONAL(str)
    # input_shape = REQUIRED(int)
    batch_size = OPTIONAL(int, default=32)
    data_url = OPTIONAL(str)
    data_sha256 = OPTIONAL(str)
    layers = REPEAT(SUBCLASS_OF(Layer))
    connections = REPEAT(Connection)

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.ctx = self._get_ctx(kwargs)
        self.data = {}
        if self.data_url is not None and self.data_sha256 is None:
            self.ctx.error("Field data_url required data_sha256 to be set")
            return

        if self.data_url is not None:
            file_name = kwargs['name'] + "_parameters.npz"
            data_url = kwargs['data_url']
            data_sha256 = kwargs['data_sha256']
            self.data = self._get_data(file_name, data_url, data_sha256)

        self.layer_free_in_ports = defaultdict(list)
        self.layer_free_out_ports = defaultdict(list)

        self._input_shapes = {}
        self._valid_input_shape = False

        self._theano_order_outs = []
        self._theano_order_ins = []

        self._setup_connections()

    def _get_data(self, file_name, url, sha256_expected):
        with open(file_name, "w+b") as f:
            utils.download(url, f)
            sha256_got = utils.sha256_file(f)
            if sha256_expected != sha256_got:
                raise ValueError("The given sha256sum {:} of is not equal to"
                                 " {:} from the url {:}"
                                 .format(sha256_expected, sha256_got, url))
            f.seek(0)
            npzfile = np.load(f)
            return {n: npzfile[n] for n in npzfile.files}

    def set_input_shapes(self, input_shapes: '{<layer>: {<port>: shape}}'):
        self._valid_input_shape = False
        self._check_satisfies_free_input_ports(input_shapes)
        self._valid_input_shape = True

        self._input_shapes = input_shapes
        self._compile()

    def _build_symbolic_outputs(self):
        self._theano_order_ins = []
        theano_ins = []
        free_in_ports_tensors = defaultdict(dict)

        for layer_name, in_ports in self.layer_free_in_ports.items():
            for in_port in in_ports:
                shape = self._input_shapes[layer_name][in_port]
                name = "{:}[{:}]".format(layer_name, in_port)
                tensor = tensor_from_shape(name, shape)
                theano_ins.append(tensor)
                free_in_ports_tensors[layer_name][in_port] = tensor
                self._theano_order_ins.append((layer_name, in_port))

        return self.layer_outputs(free_in_ports_tensors), theano_ins

    def _compile(self):
        self._assert_valid_input_shape()

        layer_outs, theano_ins = self._build_symbolic_outputs()
        self._theano_order_outs = []

        theano_outs = []
        for layer_name, out_ports in self.layer_free_out_ports.items():
            for out_port in out_ports:
                theano_outs.append(layer_outs[layer_name][out_port])
                self._theano_order_outs.append((layer_name, out_port))

        self._func = theano.function(theano_ins, theano_outs)

    def _from_theano_outs(self, theano_outs: list) -> \
            '{<layer_name>: {<port_name>: output}}':
        """Converts the output list of theano.function to an output dictionary
        `{<layer_name>: {<port_name>: output}}. """
        dict_out = defaultdict(dict)
        for (layer_name, port_name), out in \
                zip(self._theano_order_outs, theano_outs):
            dict_out[layer_name][port_name] = out

        return dict_out

    def _to_theano_ins(self, theano_ins):
        """Converts an input dictionary with structure
        `{<layer_name>: {<port_name>: output}} to a list for `self._func`. """
        return [theano_ins[layer][port]
                for layer, port in self._theano_order_ins]

    def forward(self, inputs: '{<layer>: {<port>: object}}'):
        self._assert_valid_input_shape()
        assert self._func is not None
        return self._from_theano_outs(self._func(self._to_theano_ins(inputs)))

    def layer_outputs(self, inputs: '{<layer>: {<port>: object}}'):
        outputs = {}
        inputs = defaultdict(dict, **inputs)
        inputs_shapes = defaultdict(dict, **self._input_shapes)

        dic_layers_ports = {l: set(l.input_ports()) -
                            set(inputs[l.name].keys())
                            for l in self.layers}

        self._check_satisfies_free_input_ports(inputs)

        while dic_layers_ports:
            layers_with_satisfied_inputs = \
                [layer for layer, ports in dic_layers_ports.items()
                 if not ports]

            for layer in layers_with_satisfied_inputs:
                inputs_for_layer = inputs.get(layer.name)
                for p in layer.input_ports():
                    assert p in inputs_for_layer

                layer.set_input_shapes(inputs_shapes[layer.name])
                self._setup_parameters(layer)
                outputs[layer.name] = layer.outputs(inputs_for_layer)
                for con in self._connections_from(layer):
                    inputs_shapes[con.to_name][con.to_port] = \
                        layer.output_shapes()[con.from_port]
                    inputs[con.to_name][con.to_port] = \
                        outputs[con.from_name][con.from_port]
                    # port is now satisfied
                    dic_layers_ports[con.to_layer].remove(con.to_port)
                del dic_layers_ports[layer]

        return outputs

    def _check_satisfies_free_input_ports(
            self, inputs: '{<layer>: {<port>: object}}'):
        for layer_name, free_in_ports in self.layer_free_in_ports.items():
            if layer_name not in inputs:
                self.ctx.error("No ports given for layer `{:}` "
                               "with free ports `{:}`"
                               .format(layer_name, free_in_ports))

            given_in_ports = list(inputs[layer_name].keys())
            if given_in_ports != free_in_ports:
                self.ctx.error(
                    "At Layer `{:}`: Ports given by input {:} do not match "
                    "the free ports {:}. "
                    .format(layer_name, format_ports(given_in_ports),
                            format_ports(free_in_ports)))

    def _assert_valid_input_shape(self) -> bool:
        assert self._valid_input_shape

    def _setup_parameters(self, layer):
        if issubclass(type(layer), WithParameterLayer):
            for param in layer.parameters:
                if self.data.get(param.name) is not None:
                    param.tensor = self.data[param.name]
                else:
                    layer.fill_parameter(param)

    def _connections_from(self, layer):
        return [c for c in self.connections if c.from_name == layer.name]

    def _connections_to(self, layer):
        return [c for c in self.connections if c.to_name == layer.name]

    def _setup_connections(self):
        self._add_layers_to_connections()
        for layer in self.layers:
            self._check_connections(layer)
            free_in = self._free_in_ports(layer)
            if len(free_in) > 0:
                self.layer_free_in_ports[layer.name].extend(free_in)

            free_out = self._free_out_ports(layer)
            if len(free_out) > 0:
                self.layer_free_out_ports[layer.name].extend(free_out)

    def _check_connections(self, layer):
        connected_to_port = [c.to_port for c in self._connections_to(layer)]

        for layer_in_port in layer.input_ports():
            if layer_in_port in connected_to_port:
                if connected_to_port.count(layer_in_port) > 1:
                    connections = [c for c in self._connections_to(layer)
                                   if c.to_port == layer_in_port]
                    _from = ["`{:}[{:}]`".format(c.from_name, c.from_port)
                             for c in connections]
                    self.ctx.error("Layer `{:}` has multiple connections for "
                                   "input port `{:}`. "
                                   "The connections are from: {:}."
                                   .format(layer.name, layer_in_port,
                                           ", ".join(_from)))

    def _add_layers_to_connections(self):
        for layer in self.layers:
            for con in self.connections:
                if con.is_part(layer):
                    con.add_layer(layer)

    def _free_in_ports(self, layer):
        connected_in_ports = [c.to_port for c in self._connections_to(layer)]
        return [p for p in layer.input_ports() if p not in connected_in_ports]

    def _free_out_ports(self, layer):
        connected_out_ports = [c.from_port
                               for c in self._connections_from(layer)]
        return [p for p in layer.output_ports()
                if p not in connected_out_ports]

    def get_layer(self, name):
        """Return the layer with name `name`. If no layer with that name
         exists, None is returned."""
        for l in self.layers:
            if l.name == name:
                return l
        return None
