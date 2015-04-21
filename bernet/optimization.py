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

import sys
import numpy as np
import theano
import theano.tensor as T
import time

from bernet.config import OPTIONAL, ConfigObject
from bernet.dataset import Dataset
from bernet.net import FeedForwardNet
from bernet.utils import shared_like, shared_tensor_from_dims, \
    symbolic_tensor_from_dims


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

            yield param, param - T.sgn(grad) * step
            yield grad_tm1, grad
            yield step_tm1, step


class TrainState(object):
    def __init__(self, network):
        self.network = network
        self.best_error = sys.float_info.max
        self.best_iteration = -10000
        self.best_parameters = []


class SupervisedTrainer(ConfigObject):
    optimizator = OPTIONAL(Rprop, default=Rprop())
    patience = OPTIONAL(int, default=5)
    validate_every = OPTIONAL(int, default=10)
    min_improvement = OPTIONAL(float, default=0.01)
    max_epochs = OPTIONAL(int)

    def train(self, network: FeedForwardNet, dataset: Dataset):
        trainer_state = SupervisedTrainerState(network, dataset, self)
        trainer_state.train()


class SupervisedTrainerState(object):
    def __init__(self, network: FeedForwardNet, dataset, train_opt):
        self.network = network
        self.dataset = dataset
        self.train_opt = train_opt

        self.best_loss = sys.float_info.max
        self.best_iteration = -10000
        self.best_parameters = []

        self._validate_data, self._validate_labels = \
            self._shared_tensors(dataset, network.name, "validate")
        self._train_data, self._train_labels = \
            self._shared_tensors(dataset, network.name, "train")

        self._train_fn = self._create_train_func()
        self._valid_fn = self._create_validate_func()

    def _shared_tensors(self, dataset, network_name, batch_name):
        data = shared_tensor_from_dims(
            "{}_{}_data".format(network_name, batch_name),
            dataset.data_dims())
        labels = shared_tensor_from_dims(
            "{}_{}_labels".format(network_name, batch_name),
            dataset.labels_dims())
        return data, labels

    def enough_patience(self, iteration, epoch_loss):
        max_epochs = self.train_opt.max_epochs
        if max_epochs is not None and iteration > max_epochs:
            return False
        if self.is_new_best(epoch_loss):
            self.best_loss = epoch_loss
            self.best_iteration = iteration
            self.best_parameters = \
                [p.get_value().copy()
                 for p in self.network.parameters_as_shared()]
        return iteration - self.best_iteration < \
            self.train_opt.patience

    def is_new_best(self, loss):
        return self.best_loss - loss > self.train_opt.min_improvement

    def train(self):
        epoche_iter = self.epoche_iter(self.network, self.dataset)
        for i, epoche_info in enumerate(epoche_iter, start=1):
            self._print_epoche_info(i, epoche_info)
            if not self.enough_patience(i, epoche_info['avg_valid_loss']):
                return

    def _print_epoche_info(self, epoche_number, epoche_info):
        def spaces(l, n=12):
            return ("{:.5f}".format(l)).center(n)

        best_mark = ""
        if epoche_info['is_best_iteration']:
            best_mark = "  *"

        if epoche_number % 10 == 0 or epoche_number == 1:
            offset = "".ljust(len(str(epoche_number)))
            print()
            print(offset +
                  "              train loss  | valid loss | valid accuracy")

        valid_accuracy_percent = 100*epoche_info['avg_valid_accuracy']
        valid_accuracy_str = "{:.1f}%".format(valid_accuracy_percent).ljust(4)

        print("#{} in {}s |{}|{}| {}{}"
              .format(epoche_number,
                      "{:.3f}".format(epoche_info['duration']).center(6),
                      spaces(epoche_info['avg_train_loss']),
                      spaces(epoche_info['avg_valid_loss']),
                      valid_accuracy_str,
                      best_mark))

    def epoche_iter(self, network: FeedForwardNet, dataset: Dataset) -> dict:
        while True:
            start = time.time()
            train_losses = list(self.run_train_epoch())
            valid_losses = []
            valid_accuracies = []
            for loss, accuracy in self.run_valid_epoch():
                valid_losses.append(loss)
                valid_accuracies.append(accuracy)
            avg_valid_loss = np.mean(valid_losses)
            yield dict(
                is_best_iteration=self.is_new_best(avg_valid_loss),
                duration=time.time()-start,
                avg_train_loss=np.mean(train_losses),
                avg_valid_loss=avg_valid_loss,
                avg_valid_accuracy=np.mean(valid_accuracies)
            )

    def run_valid_epoch(self):
        validate_batch = next(self.dataset.validate_epoch())
        self._validate_data.set_value(validate_batch.data())
        self._validate_labels.set_value(validate_batch.labels())
        for b, e, in validate_batch.minibatch_idx():
            yield self._valid_fn(b, e)

    def run_train_epoch(self):
        train_batch = next(self.dataset.train_epoch())
        self._train_data.set_value(train_batch.data())
        self._train_labels.set_value(train_batch.labels())
        for b, e, in train_batch.minibatch_idx():
            yield self._train_fn(b, e)

    def _create_validate_func(self):
        x = symbolic_tensor_from_dims('x', self.dataset.data_dims())
        labels = symbolic_tensor_from_dims('labels',
                                           self.dataset.labels_dims())
        idx_begin = T.lscalar('idx_begin')
        idx_end = T.lscalar('idx_end')
        batch_slice = slice(idx_begin, idx_end)
        out = self.network.output(x)
        loss = self.network.get_loss(out, labels)
        accuracy = self.network.get_accuracy(out, labels)
        self.validate_params = self.network.parameters_as_shared()
        fn = theano.function(
            [idx_begin, idx_end],
            [loss, accuracy],
            givens={
                x: self._validate_data[batch_slice],
                labels: self._validate_labels[batch_slice]
            }
        )
        return fn

    def _create_train_func(self):
        x = T.matrix('x')
        y = T.vector('y')
        idx_begin = T.lscalar('idx_begin')
        idx_end = T.lscalar('idx_end')
        out = self.network.output(x)
        loss = self.network.get_loss(out, y)
        self.train_params = self.network.parameters_as_shared()
        fn = theano.function(
            [idx_begin, idx_end],
            loss,
            givens={
                x: self._train_data[idx_begin:idx_end, :],
                y: self._train_labels[idx_begin:idx_end]
            },
            updates=self.train_opt.optimizator.updates(loss, self.train_params)
        )
        return fn
