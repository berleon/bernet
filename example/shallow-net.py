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

from bernet.net import SimpleNetwork
from bernet.optimization import SupervisedTrainer
from bernet.dataset import MNISTDataset

_dir = os.path.dirname(os.path.realpath(__file__))

mnist = MNISTDataset()
with open(_dir + "/shallow-net.json") as f:
    net = SimpleNetwork.load_json(f)
    trainer = SupervisedTrainer()
    trainer.train(net, mnist)
    print(net.confusion_matrix(next(mnist.train())))
    print(net.confusion_matrix(next(mnist.test())))

