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

from unittest import TestCase
import simplejson as json
from bernet.json.json_config import *


class TestREQUIRED(TestCase):
    def test_required(self):
        req1 = REQUIRED(int)
        self.assert_(req1.valid(12))
        self.assertFalse(req1.valid(None))
        self.assertFalse(req1.valid(1.1))

    def test_required_nested(self):
        req2 = REQUIRED(REPEAT(int))
        self.assert_(req2.valid([]))
        self.assert_(req2.valid([2]))
        self.assert_(req2.valid([2]*10))
        self.assertFalse(req2.valid([1.1]))

    def test_required_optional_fails(self):
        self.assertRaises(ValueError, REQUIRED, OPTIONAL(int))

    def test_no_validable_or_type_fails(self):
        self.assertRaises(ValueError, REQUIRED, 15)


class TestOPTIONAL(TestCase):
    def test_optional(self):
        opt = OPTIONAL(int)
        self.assert_(opt.valid(None))
        self.assert_(opt.valid(0))
        self.assertFalse(opt.valid(2.2))

    def test_optional_nested(self):
        opt = OPTIONAL(REPEAT(int))
        self.assert_(opt.valid([]))
        self.assert_(opt.valid(None))
        self.assert_(opt.valid([2]*10))
        self.assertFalse(opt.valid([1.1]))


class TestREPEAT(TestCase):
    def test_repeat(self):
        rep = REPEAT(float)
        self.assert_(rep.valid([]))
        self.assert_(rep.valid([2.2]))
        # float not list of float
        self.assertFalse(rep.valid(2.2))
        # list of ints not list of floats
        self.assertFalse(rep.valid([2, 2]))


class TestEITHER(TestCase):
    def test_either(self):
        eth = EITHER("left", "right")
        self.assert_(eth.valid("right"))
        self.assert_(eth.valid("left"))
        self.assertFalse(eth.valid("middle"))


class Person(JsonObject):
    name = REQUIRED(str)
    sex = EITHER("female", "male", "x")
    age = OPTIONAL(int)


class Company(JsonObject):
    name = REQUIRED(str)
    employes = REPEAT(Person)


class TestJsonObject(TestCase):
    def test_attributes(self):
        p = Person(name="Max", sex="male", age=20)
        self.assertEqual(p.name, "Max")
        self.assertEqual(p.sex, "male")
        self.assertEqual(p.age, 20)

        self.assertRaises(JsonException, Person, name=20, sex='female', age=20)
        self.assertRaises(JsonException, Person, name='Max', sex='', age=20)
        susi = Person(name='Susi', sex='female')
        self.assertNotEqual(susi, p)

    def test_nested(self):
        max = Person(name="Max", sex="male", age=20)
        susi = Person(name='Susi', sex='female')
        c1 = Company(name="Planetron Inc.", employes=[max, susi])
        c2 = Company(name="Planetron Lichtenstein Inc.", employes=None)

        self.assertNotEqual(susi, c1)
        self.assertRaises(JsonException, Company, name=20, employes=[])


class TestJsonEncoding(TestCase):
    def test_simple_encoding(self):
        max = Person(name="Max", sex="male", age=20)

        max_json = Person.loads(max.dumps())
        self.assertEqual(max, max_json)

    def test_complex_encoding(self):
        max = Person(name="Max", sex="male", age=20)
        hans = Person(name="Hans", sex="male", age=26)

        c = Company(name="Planetron Inc.", employes=[max, hans])
        through_json = Company.loads(c.dumps())
        self.assertEqual(c, through_json)

