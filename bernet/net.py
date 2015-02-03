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

from bernet import utils
from bernet.config import REQUIRED, OPTIONAL, ConfigObject, REPEAT, SUBCLASS_OF
from bernet.layer import Layer, Connection


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
        ctx = self._get_ctx(kwargs)
        if self.data_url is not None and self.data_sha256 is None:
            ctx.error("Field data_url required data_sha256 to be set")
            return

        if self.data_url is not None:
            file_name = kwargs['name'] + "_parameters.npz"
            data_url = kwargs['data_url']
            data_sha256 = kwargs['data_sha256']
            self.data = self._get_data(file_name, data_url, data_sha256)

        self.input_layers_labels = defaultdict(list)
        self.output_layers_labels = defaultdict(list)

        self._setup_connections()
        self._setup_parameters()

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

    def outputs(self, inputs: '{<layer>: {<port>: object}}'):
        pass

    def _setup_parameters(self):
        pass

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
                self.input_layers_labels[layer.name].extend(free_in)

            free_out = self._free_out_ports(layer)
            if len(free_out) > 0:
                self.output_layers_labels[layer.name].extend(free_out)

    def _check_connections(self, layer):
        connected_to_port = [c.to_port for c in self._connections_to(layer)]

        for in_port in layer.input_ports():
            if in_port in connected_to_port:
                assert connected_to_port.count(in_port) == 1

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
