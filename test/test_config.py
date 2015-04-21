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

from bernet.config import *


class ConfigFieldTestCase(TestCase):
    def setUp(self):
        self.loader = Loader("")
        self.dumper = Dumper("")

    def assertNotValid(self, field, value):
        self.assertRaises(ConfigError, field.assert_valid, value)


class TestCostraint(TestCase):
    def test_abstract_methods_raise_not_implemented(self):
        c = Constraint()
        self.assertRaises(NotImplementedError, c.assert_valid, None, None)
        self.assertRaises(NotImplementedError, c.construct, None, None)
        self.assertRaises(NotImplementedError, c.represent, None, None)
        self.assertRaises(NotImplementedError, c.type_signature)


class TestConfigField(TestCase):
    def test_abstract_methods_raise_not_implemented(self):
        c = ConfigField()
        self.assertEqual(c.default(), None)
        self.assertRaises(NotImplementedError, c.assert_valid, None, None)
        self.assertRaises(NotImplementedError, c.construct, None, None)
        self.assertRaises(NotImplementedError, c.represent, None, None)
        self.assertRaises(NotImplementedError, c.type_signature)


class TestREQUIRED(ConfigFieldTestCase):
    def test_required(self):
        req = REQUIRED(int)
        req.assert_valid(12)
        self.assertNotValid(req, None)
        self.assertNotValid(req, 1.1)

        node = ScalarNode(None, '210')
        self.assertEqual(req.construct(self.loader, node), 210)

        node = ScalarNode(None, 'randomfoo')
        self.assertRaises(ValueError, req.construct, self.loader, node)

    def test_required_nested(self):
        req2 = REQUIRED(REPEAT(int))
        req2.assert_valid([])
        req2.assert_valid([2])
        req2.assert_valid([2]*10)
        self.assertNotValid(req2, [1.1])

    def test_required_optional_fails(self):
        self.assertRaises(ValueError, REQUIRED, OPTIONAL(int))

    def test_no_validable_or_type_fails(self):
        self.assertRaises(ValueError, REQUIRED, 15)


class TestOPTIONAL(ConfigFieldTestCase):
    def test_optional(self):
        opt = OPTIONAL(int)
        opt.assert_valid(None)
        opt.assert_valid(0)
        self.assertNotValid(opt, 2.2)

    def test_optional_default(self):
        opt = OPTIONAL(float, default=1.)
        node = ScalarNode(None, '0.1234')
        self.assertEqual(opt.construct(self.loader, node), 0.1234)
        node.value = 'None'
        self.assertEqual(opt.construct(self.loader, node), 1.)

    def test_optional_nested(self):
        opt = OPTIONAL(REPEAT(int))
        opt.assert_valid([])
        opt.assert_valid(None)
        opt.assert_valid([2]*10)
        self.assertNotValid(opt, [1.1])


class TestREPEAT(ConfigFieldTestCase):
    def test_repeat(self):
        rep = REPEAT(float)
        rep.assert_valid([])
        rep.assert_valid([2.2])

        self.assertNotValid(rep, None)
        # float is not a list of float
        self.assertNotValid(rep, 2.2)
        # list of ints is not a list of floats
        self.assertNotValid(rep, [2, 2])

        self.assertNotValid(rep, [(4.5,), 2.3])

    def test_repeat_construct(self):
        node = yaml.compose('[12, 54]')
        rep = REPEAT(int)
        self.assertListEqual(rep.construct(self.loader, node), [12, 54])


class TestENUM(ConfigFieldTestCase):
    def test_either_fixed_values(self):
        enum = ENUM("left", "right")
        enum.assert_valid("right")
        enum.assert_valid("left")
        self.assertNotValid(enum, "middle")

    def test_same_fixed_values(self):
        self.assertRaises(ValueError, ENUM, "right", "right")

    def test_one_arg_fails(self):
        self.assertRaises(ValueError, ENUM, "wrong")


class TestTAGS(ConfigFieldTestCase):
    def test_one_arg_fails(self):
        self.assertRaises(ValueError, TAGS, {"wrong": int})

    def test_non_string_key_fails(self):
        self.assertRaises(ValueError, TAGS, {None: str, 'bla': int})

    def test_tags_int_str(self):
        tags = TAGS({'str': str, 'int': int})
        tags.assert_valid("right")
        tags.assert_valid("left")

        tags.assert_valid(14)
        tags.assert_valid(-1000)

        self.assertNotValid(tags, 1.4)
        self.assertNotValid(tags, object)

    def test_person_dict(self):
        eth2 = TAGS({'Person': Person, 'dict': dict})

        person = eth2.construct(self.loader,
                                yaml.compose('!Person {name: max, sex: male}'))
        self.assertEqual(person.name, "max")

        self.assertRaises(ConfigError, eth2.construct,
                          self.loader,
                          yaml.compose('!WRONG_TAG {name: max, sex: male}'))

        dic = eth2.construct(self.loader,
                             yaml.compose('!dict {name: max, sex: male}'))
        self.assertEqual(dic['name'], "max")


class TestEITHER(ConfigFieldTestCase):
    def test_assert_valid(self):
        eth = EITHER(str, int)
        eth.assert_valid("hello world!")
        eth.assert_valid("")
        eth.assert_valid(0)
        eth.assert_valid(-2000)
        eth.assert_valid(100000000000)

        self.assertNotValid(eth, None)
        self.assertNotValid(eth, 1.)
        self.assertNotValid(eth, [3, 45])

    def test_construct(self):
        eth = EITHER(Person, bool)
        self.assertEqual(True,
                         eth.construct(self.loader, yaml.compose("yes")))

        self.assertEqual(
            Person(name='max', sex='male'),
            eth.construct(self.loader,
                          yaml.compose('!Person {name: max, sex: male}'))
        )

    def test_represent(self):
        eth = EITHER(Person, bool)
        self.assertEqual("true", eth.represent(self.dumper, True).value)
        leon = eth.represent(self.dumper, Person(name='leon', sex='male'))
        self.assertEqual("Person", leon.tag)


class Person(ConfigObject):
    name = REQUIRED(str, doc="<Person.name docstring>")
    sex = ENUM("female", "male", "x")
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

        self.assertRaises(ConfigError,  Person,
                          name=20, sex='female', age=20)
        self.assertRaises(ConfigError, Person, name='Max', sex='', age=20)
        susi = Person(name='Susi', sex='female')
        self.assertNotEqual(susi, p)

    def test_attributes_assignment(self):
        p = Person(name="Max", sex="male", age=20)
        self.assertEqual(p.sex, "male")
        p.sex = "female"
        self.assertEqual(p.sex, "female")
        with self.assertRaises(ConfigError):
            p.sex = "wrong"

    def test_yaml_tag_is_classname(self):
        self.assertEqual(Person.yaml_tag, 'Person')
        self.assertEqual(Company.yaml_tag, 'Company')

    def test_nested(self):
        max = Person(name="Max", sex="male", age=20)
        susi = Person(name='Susi', sex='female')
        c1 = Company(name="Planetron Inc.", employes=[max, susi])
        c2 = Company(name="Planetron Lichtenstein Inc.", employes=[])
        c3 = Company(name="Ventoex Inc.",
                     employes=[
                         Person(name="Max", sex="male", age=20),
                         Person(name="Susi", sex="female"),
                     ])

        self.assertNotEqual(susi, c1)
        self.assertRaises(ConfigError, Company, name=20, employes=[])

    def test_error_handling_object(self):
        self.assertRaises(ConfigError, Person)
        self.assertRaisesRegex(
            ConfigError,
            re.escape(
                "Expected type `str`, but got value `None` of type "
                "`NoneType`."),
            Person,
            sex="male")

    def test_optional_default(self):
        car3 = Car(n_wheels=3)
        self.assertEqual(car3.n_wheels, 3)

        car = Car()
        self.assertEqual(car.n_wheels, 4)

    def test_docstrings(self):
        Person_cls = ConfigObjectMetaclass(
            "TestClass", (ConfigObject, object),
            {"field": REQUIRED(str, "<TestClass.field docstring>")})
        docstrings = ConfigObjectMetaclass._docstrings(Person_cls)
        self.assertIn(
            ":param field: <TestClass.field docstring>\n"
            ":type field: :class:`.str`",
            docstrings)


class TestYamlEncoding(TestCase):
    def test_simple_load(self):
        max_yaml = """
            name: Max
            sex: male
            age: 20"""
        max = load(Person, max_yaml)
        self.assertEqual(max.name, "Max")
        self.assertEqual(max.sex, "male")
        self.assertEqual(max.age, 20)

    def test_simple_dump(self):
        john = Person(name="John Doe", sex="male", age=20)
        john.use_yaml_tag = False
        john_yaml = dump(john, default_flow_style=False)
        self.assertEqual(john_yaml, 'age: 20\nname: John Doe\nsex: male\n')

    def test_simple_encoding(self):
        max = Person(name="Max", sex="male", age=20)
        max_through_yaml = load(Person, dump(max))
        self.assertEqual(max, max_through_yaml)

    def test_complex_dump(self):
        max = Person(name="Max", sex="male", age=20)
        susi = Person(name='Susi', sex='female')
        c = Company(name="Planetron Inc.", employes=[max, susi])
        self.maxDiff = None
        # stable over time
        self.assertEqual(dump(c), dump(c))
        self.assertEqual(dump(c),
                         "employes:\n"
                         "- age: 20\n"
                         "  name: Max\n"
                         "  sex: male\n"
                         "- name: Susi\n"
                         "  sex: female\n"
                         "name: Planetron Inc.\n"
                         "")
