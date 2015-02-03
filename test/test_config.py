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
import re
import unittest

from bernet.config import *
from bernet.config import _TypeConstructableWrapper


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

    def test_optional_default(self):
        opt = OPTIONAL(float, default=1.)
        self.assertEqual(opt.construct(0.), 0.)
        self.assertEqual(opt.construct(None), 1.)

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

    def test_error_handling_list(self):
        self.assertRaises(ConfigException, Company,
                          name="Labnet Inc.", employes="nobody")
        max = Person(name="Max", sex="male", age=20)
        self.assertRaisesRegex(
            ConfigException,
            "Expected a listlike type, but got type Person.\n"
            "Traceback:\n"
            "    Field:    employes,\n"
            "    Object:   Company",
            Company,
            name="Labnet Inc.", employes=max)

        self.assertRaisesRegex(
            ConfigException,
            "Expected type `Person`, but got value `susi` with type `str`.\n"
            "Traceback:\n"
            "    List:     at element [1],\n"
            "    Field:    employes,\n"
            "    Object:   Company",
            Company,
            name="Labnet Inc.", employes=[max, "susi"])


class TestEITHER(TestCase):
    def test_one_arg_fails(self):
        self.assertRaises(ValueError, EITHER, "wrong")

    def test_either_fixed_values(self):
        eth = EITHER("left", "right")
        self.assert_(eth.valid("right"))
        self.assert_(eth.valid("left"))
        self.assertFalse(eth.valid("middle"))

    def test_either_int_str(self):
        eth = EITHER(str, int)
        self.assert_(eth.valid("right"))
        self.assert_(eth.valid("left"))

        self.assert_(eth.valid(14))
        self.assert_(eth.valid(-1000))

        self.assertFalse(eth.valid(1.4))
        self.assertFalse(eth.valid(object))

    def test_multiple_true(self):
        eth2 = EITHER(Person, dict)
        self.assertRaises(ConfigException, eth2.construct,
                          {"name": "max", "sex": "male"},
                          InitContext(raise_exceptions=True))

    def test_same_fixed_values(self):
        self.assertRaises(ValueError, EITHER, "right", "right")


class TestSUBCLASS_OF(TestCase):
    class Parent(object):
        def __eq__(self, other):
            return type(other) == type(self)

    class Son(Parent):
        pass

    class Doughter(Parent):
        pass

    class Stepson(Son):
        pass

    def test_subclass_of_exception(self):
        sub = SUBCLASS_OF(Exception)
        self.assert_(sub.valid(ValueError("bla")))
        self.assert_(sub.valid(SyntaxError("bla")))
        self.assert_(sub.valid(Exception()))
        self.assert_(not sub.valid(None))

    def test_subclass_repeated(self):
        t = TestSUBCLASS_OF
        rep_sub = REPEAT(SUBCLASS_OF(t.Parent))
        self.assert_(rep_sub.valid([t.Son(), t.Doughter()]))
        self.assert_(rep_sub.valid((t.Son(),)))
        self.assert_(not rep_sub.valid([None]))
        self.assert_(not rep_sub.valid((t.Doughter, None)))

        self.assertListEqual(rep_sub.construct([t.Doughter(), t.Son()]),
                             [t.Doughter(), t.Son()])

    def test_subclass_error(self):
        sub = SUBCLASS_OF(Exception)
        ctx = InitContext(raise_exceptions=True)
        self.assertRaisesRegex(
            ConfigException,
            re.compile('Got value `\w+` of type `\w+`\.'
                       ' Expected a subclass of `\w+`'),
            sub.construct,
            "wrong", ctx
        )


class Person(ConfigObject):
    name = REQUIRED(str)
    sex = EITHER("female", "male", "x")
    age = OPTIONAL(int)


class Company(ConfigObject):
    name = REQUIRED(str)
    employes = REPEAT(Person)


class Car(ConfigObject):
    n_wheels = OPTIONAL(int, default=4)


class TestConfigObject(TestCase):
    def test_attributes(self):
        p = Person(name="Max", sex="male", age=20)
        self.assertEqual(p.name, "Max")
        self.assertEqual(p.sex, "male")
        self.assertEqual(p.age, 20)

        self.assertRaises(ConfigException,  Person,
                          name=20, sex='female', age=20)
        self.assertRaises(ConfigException, Person, name='Max', sex='', age=20)
        susi = Person(name='Susi', sex='female')
        self.assertNotEqual(susi, p)

    def test_attributes_assignment(self):
        p = Person(name="Max", sex="male", age=20)
        self.assertEqual(p.sex, "male")
        p.sex = "female"
        self.assertEqual(p.sex, "female")
        with self.assertRaises(ConfigException):
            p.sex = "wrong"

    def test_nested(self):
        max = Person(name="Max", sex="male", age=20)
        susi = Person(name='Susi', sex='female')
        c1 = Company(name="Planetron Inc.", employes=[max, susi])
        c2 = Company(name="Planetron Lichtenstein Inc.", employes=None)
        c3 = Company(name="Ventoex Inc.",
                     employes=[
                         {"name": "Max", "sex": "male", "age": 20},
                         {"name": "Susi", "sex": "female"},
                     ])

        self.assertNotEqual(susi, c1)
        self.assertRaises(ConfigException, Company, name=20, employes=[])

    def test_error_handling_object(self):
        self.assertRaises(ConfigException, Person)
        self.assertRaisesRegex(
            ConfigException,
            re.escape(
                "Expected type `str`, but got value `None` with type "
                "`NoneType`.\n"
                "Traceback:\n"
                "    Field:    name,\n"
                "    Object:   Person"),
            Person,
            sex="male")

    def test_optional_default(self):
        car3 = Car(n_wheels=3)
        self.assertEqual(car3.n_wheels, 3)

        car = Car()
        self.assertEqual(car.n_wheels, 4)


class TestJsonEncoding(TestCase):
    def test_simple_encoding(self):
        max = Person(name="Max", sex="male", age=20)

        max_json = Person.from_json(max.to_json())
        self.assertEqual(max, max_json)

    def test_complex_encoding(self):
        max = Person(name="Max", sex="male", age=20)
        hans = Person(name="Hans", sex="male", age=26)

        c = Company(name="Planetron Inc.", employes=[max, hans])
        through_json = Company.from_json(c.to_json())
        self.assertEqual(c, through_json)

    def test_wrong_json_fails(self):
        json_str = """{"name": "Max", "sex": "foo"}"""
        Person.from_json(json_str)


class TestInitContext(TestCase):
    def test_stack(self):
        ctx = InitContext()
        with ctx.step_into("Object", "Person"):
            with ctx.step_into("Field", "name"):
                self.assertListEqual(ctx._stack, [("Object", "Person"),
                                                  ("Field", "name")])


class TestTypeConstrutableWrapper(TestCase):
    def test_is_constructable(self):
        wrapper = _TypeConstructableWrapper(int)
        self.assertEqual(wrapper.construct(20, InitContext()), 20)

        wrapper = _TypeConstructableWrapper(REQUIRED(int))
        self.assertEqual(wrapper.construct(20, InitContext()), 20)
