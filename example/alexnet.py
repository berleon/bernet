#! /usr/bin/env python
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

import numpy as np
from scipy.misc import imread

from bernet.net import FeedForwardNet
from bernet.config import load


_dir = os.path.dirname(os.path.realpath(__file__))

with open(_dir + "/../models/alexnet.yaml") as f:
    net = load(FeedForwardNet, f)
    cat = (imread(_dir + "/cat.jpg") / 255.)
    cat = cat.swapaxes(0, 2).reshape((1, 3, 224, 224))
    out = net.forward(cat)
    print("Predicted class is {}".format(np.argmax(out)))

