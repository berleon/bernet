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

import numpy as np

from bernet import utils
from bernet.config import REQUIRED, OPTIONAL, ConfigObject


class Network(ConfigObject):
    name = REQUIRED(str)
    description = OPTIONAL(str)
    # input_shape = REQUIRED(int)
    batch_size = OPTIONAL(int, default=32)
    data_url = OPTIONAL(str)
    data_sha256 = OPTIONAL(str)

    def __init__(self,  **kwargs):
        super(ConfigObject, self).__init__()
        ctx = self._get_ctx(kwargs)
        if self.data_url is not None and self.data_sha256 is None:
            ctx.error("Field data_url required data_sha256 to be set")
            return

        if self.data_url is not None:
            file_name = kwargs['name'] + "_parameters.npz"
            data_url = kwargs['data_url']
            data_sha256 = kwargs['data_sha256']
            self.data = self._get_data(file_name, data_url, data_sha256)

    def _get_data(self, file_name, url, sha256_expected):
        with open(file_name, "w+b") as f:
            utils.download(url, f)
            sha256_got = utils.sha256_file(f)
            if sha256_expected != sha256_got:
                raise ValueError("The given sha256sum {:} of is not equal"
                                 " {:} from the url {:}"
                                 .format(sha256_expected, sha256_got, url))
            f.seek(0)
            npzfile = np.load(f)
            return {n: npzfile[n] for n in npzfile.files}

