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

import theano.tensor as T


class Loss(object):
    def loss(self, net_out, labels):
        raise NotImplementedError()


class MSE(Loss):
    def loss(self, net_out, labels):
        return T.mean((net_out - labels)**2)


class NegativeLogLikelihood(Loss):
    def loss(self, net_out, labels):
        return -T.mean(T.log(net_out[T.arange(labels.shape[0]),
                                     T.cast(labels, 'int32')]))
