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
import random
from unittest import TestCase
from unittest.mock import MagicMock, patch

import theano
import theano.tensor as T

from bernet.dataset import LineDataset
from bernet.optimization import Rprop, SupervisedTrainer


def simple_optimization_problem(model, line_m, line_c):
    def line(x):
        return line_m*x + line_c

    m = theano.shared(-0.1, 'c')
    c = theano.shared(0.1, 'm')
    x = T.scalar('x')
    expected = T.scalar('expected')
    y = x*m + c  # simple equation of a line
    cost = T.abs_(y - expected)

    fn = theano.function([x, expected],
                         [cost],
                         updates=model.updates(cost, [m, c]))
    # make sure every test gets the same random numbers
    random.seed(12345)
    for i in range(4000):
        x = random.random() * 1000 - 500
        fn(x, line(x))

    return m.get_value(), c.get_value()


class RpropTest(TestCase):
    def test_rprop(self):
        expected_m = 5
        expected_c = 3
        m, c = simple_optimization_problem(Rprop(), expected_m, expected_c)
        self.assertAlmostEqual(m, expected_m, places=1)
        self.assertAlmostEqual(c, expected_c, places=1)


class TestSupervisedTrainer(TestCase):
    def test_supervised_trainer(self):
        trainer = SupervisedTrainer()
        m = theano.shared(0.1, 'm')
        c = theano.shared(0.1, 'c')
        expected_m = 5
        expected_c = 3

        network = MagicMock()
        network.parameters_as_shared.return_value = [m, c]
        network.output = lambda x: T.reshape(m*x + c, (-1,))
        network.get_loss = lambda o, y: T.sum(T.sqr(o - y)**2)
        network.get_accuracy = lambda o, y: T.sum(T.eq(o, y))

        dataset = LineDataset(shape=(3000, 1), m=expected_m,
                              c=expected_c, seed=1234)
        trainer.train(network, dataset)
        self.assertAlmostEqual(m.get_value(), expected_m, places=1)
        self.assertAlmostEqual(c.get_value(), expected_c, places=1)
