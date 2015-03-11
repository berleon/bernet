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
import math
from unittest import TestCase
import itertools

import sys
import theano
import theano.tensor as T

from bernet.config import OPTIONAL, ConfigObject
from bernet.dataset import Dataset
from bernet.net import SimpleNetwork
from bernet.utils import shared_like, shared_tensor_from_dims


class Optimizator(ConfigObject):
    def _grads(self, cost, params):
        assert cost is not None
        for param in params:
            yield T.grad(cost, param)

    def updates(self, cost, params):
        grads = list(self._grads(cost, params))
        return list(self._updates(cost, params, grads))

    def _updates(self, cost, params, grads):
        raise NotImplementedError()


class Rprop(Optimizator):
    max_step = OPTIONAL(float, default=2)
    min_step = OPTIONAL(float, default=math.exp(-6))
    step_increase = OPTIONAL(float, default=1.2)
    step_decrease = OPTIONAL(float, default=0.5)

    def _updates(self, cost, params, grads):
        for param, grad in zip(params, grads):
            grad_tm1 = shared_like(param, 'grad')
            step_tm1 = shared_like(param, 'step', init=1.)
            test = grad * grad_tm1
            same = T.gt(test, 0)
            diff = T.lt(test, 0)
            step = \
                T.minimum(
                    self.max_step,
                    T.maximum(
                        self.min_step,
                        step_tm1*(T.eq(test, 0) +
                                  same * self.step_increase +
                                  diff * self.step_decrease)))

            # grad = grad - diff * grad
            yield param, param - T.sgn(grad) * step
            yield grad_tm1, grad
            yield step_tm1, step


class TrainState(object):
    def __init__(self, network):
        self.network = network
        self.best_error = sys.float_info.max
        self.best_iteration = -10000
        self.best_parameters = []


class SupervisedTrainer(ConfigObject, TestCase):
    optimizator = OPTIONAL(Rprop, default=Rprop())
    patience = OPTIONAL(int, default=20)
    validate_every = OPTIONAL(int, default=10)
    min_improvement = OPTIONAL(float, default=0.01)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_shared_tensors_for(self, dataset, network_name, batch_name):
        data = shared_tensor_from_dims(
            "{}_{}_data".format(network_name, batch_name),
            dataset.data_dims())

        labels = shared_tensor_from_dims(
            "{}_{}_labels".format(network_name, batch_name),
            dataset.labels_dims())
        return data, labels

    def enough_patience(self, iteration, errors, train_state):
        marker = ''
        if train_state.best_error - errors > self.min_improvement:
            train_state.best_error = errors
            train_state.best_iteration = iteration
            train_state.best_parameters = \
                [p.get_value().copy()
                 for p in train_state.network.parameters_as_shared()]
            marker = '* '
        print("{} {}".format(marker, errors))
        return iteration - train_state.best_iteration < \
            self.patience * self.validate_every

    def train(self, network: SimpleNetwork, dataset: Dataset):
        validate_step = self.validate_generator(network, dataset)
        train_step = self.train_generator(network, dataset)
        train_state = TrainState(network)
        for i in itertools.count():
            train_loss = next(train_step)
            if i % self.validate_every == 0 and i != 0:
                validate_error = next(validate_step)
                if not self.enough_patience(i, validate_error, train_state):
                    return

    def validate_generator(self, network, dataset):
        validate_data, validate_labels = \
            self._get_shared_tensors_for(dataset, network.name, "validate")
        validation_fn = self._validate_fn(network, validate_data,
                                          validate_labels)
        while True:
            validate_batch = next(dataset.validate())
            validate_data.set_value(validate_batch.data())
            validate_labels.set_value(validate_batch.labels())
            for b, e, in validate_batch.minibatch_idx():
                yield validation_fn(b, e)[0]

    def _validate_fn(self, network: SimpleNetwork, data, labels):
        x = T.matrix('x')
        y = T.vector('y')
        idx_begin = T.lscalar('idx_begin')
        idx_end = T.lscalar('idx_end')
        out = network.net_output(x)
        loss = network.get_loss_with_output(out, y)
        self.validate_params = network.parameters_as_shared()
        fn = theano.function(
            [idx_begin, idx_end],
            [loss],
            givens={
                x: data[idx_begin:idx_end, :],
                y: labels[idx_begin:idx_end]
            }
        )
        return fn

    def train_generator(self, network, dataset):
        train_data, train_labels = \
            self._get_shared_tensors_for(dataset, network.name, "train")

        train_fn = self._train_fn(network, train_data, train_labels)
        while True:
            train_batch = next(dataset.train())
            train_data.set_value(train_batch.data())
            train_labels.set_value(train_batch.labels())
            for b, e, in train_batch.minibatch_idx():
                yield train_fn(b, e)

    def _train_fn(self, network: SimpleNetwork, data, labels):
        x = T.matrix('x')
        y = T.vector('y')
        idx_begin = T.lscalar('idx_begin')
        idx_end = T.lscalar('idx_end')
        out = network.net_output(x)
        loss = network.get_loss_with_output(out, y)
        error = network.get_error_with_output(out, y)
        self.train_params = network.parameters_as_shared()
        fn = theano.function(
            [idx_begin, idx_end],
            [loss, error],
            givens={
                x: data[idx_begin:idx_end, :],
                y: labels[idx_begin:idx_end]
            },
            updates=self.optimizator.updates(loss, self.train_params)
        )
        return fn
