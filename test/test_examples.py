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

import glob
import os

from unittest import TestCase
import subprocess
from os.path import join, abspath, dirname
import sys


class TestExamples(TestCase):
    def test_examples(self):
        dir = dirname(__file__)
        examples = abspath(join(dir, "../example/*.py"))
        for py_file in glob.glob(examples):
            print(join(sys.path))
            env = os.environ
            env["PYTHONPATH"] = ":".join(sys.path)
            subprocess.check_call(py_file, shell=True, env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
