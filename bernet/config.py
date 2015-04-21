# Copyright 2014 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from _yaml import MappingNode

import numpy as np
import yaml
from yaml import YAMLObjectMetaclass, SequenceNode, ScalarNode, \
    Loader, Dumper
from yaml.constructor import ConstructorError

try:
    import simplejson as json
except ImportError:
    import json


class ConfigError(yaml.error.MarkedYAMLError):
    pass


def config_error(msg, node=None):
    if node is not None:
        mark = node.start_mark
    else:
        mark = None
    return ConfigError(problem=msg, problem_mark=mark)


class Constraint(object):
    def assert_valid(self, value, node=None):
        raise NotImplementedError()

    def construct(self, loader, node) -> 'value':
        raise NotImplementedError()

    def represent(self, dumper, value) -> 'node':
        raise NotImplementedError()

    def type_signature(self) -> 'str':
        raise NotImplementedError()

    @staticmethod
    def factory(constraint):
        if issubclass(type(constraint), ConfigField):
            return ConfigFieldConstraint(constraint)
        elif type(constraint) == ConfigObjectMetaclass \
                and issubclass(constraint, ConfigObject):
            return ConfigObjectConstraint(constraint)
        elif type(constraint) == type:
            return BaseTypeConstraint(constraint)
        else:
            raise ValueError("Expected a subclass of ConfigField or a type,"
                             " but got `{}`.".format(constraint))


class ConfigFieldConstraint(Constraint):
    def __init__(self, config_field):
        self.config_field = config_field

    def represent(self, dumper, value) -> 'node':
        return self.config_field.represent(dumper, value)

    def construct(self, loader, node) -> 'value':
        return self.config_field.construct(loader, node)

    def assert_valid(self, value, node=None):
        return self.config_field.assert_valid(value, node)

    def type_signature(self):
        return self.config_field.type_signature()


class ConfigObjectConstraint(Constraint):
    def __init__(self, config_object_cls):
        self.config_object_cls = config_object_cls

    def represent(self, dumper, value) -> 'node':
        return self.config_object_cls.to_yaml(dumper, value)

    def construct(self, loader, node) -> 'value':
        return self.config_object_cls.from_yaml(loader, node)

    def assert_valid(self, value, node=None):
        if not isinstance(value, self.config_object_cls):
            raise config_error(
                "Expected type `{}`, but got value `{}` of type `{}`."
                .format(self.config_object_cls.__name__, value,
                        type(value).__name__), node)

    def type_signature(self):
        return ":class:`.{}`".format(self.config_object_cls.__name__)


class BaseTypeConstraint(Constraint):
    BASE_CONSTRUCTORS = {
        int: lambda loader, node: loader.construct_yaml_int(node),
        str: lambda loader, node: loader.construct_yaml_str(node),
        float: lambda loader, node: loader.construct_yaml_float(node),
        complex: lambda loader, node: loader.construct_python_complex(node),
        dict: lambda loader, node: loader.construct_mapping(node),
        list: lambda loader, node: loader.construct_sequence(node),
        bool: lambda loader, node: loader.construct_yaml_bool(node)
    }

    def __init__(self, type):
        assert type in self.BASE_CONSTRUCTORS
        self.type = type

    def represent(self, dumper, value) -> 'node':
        return dumper.represent_data(value)

    def construct(self, loader, node) -> 'value':
        return self.BASE_CONSTRUCTORS[self.type](loader, node)

    def assert_valid(self, value, node=None):
        if type(value) != self.type:
            raise config_error(
                "Expected type `{}`, but got value `{}` of type `{}`."
                .format(self.type.__name__, value, type(value).__name__), node)

    def type_signature(self):
        return ":class:`.{}`".format(self.type.__name__)


class ConfigField(object):
    def __init__(self, doc="", type_sig=""):
        self._doc = doc
        self._doc_type_sig = type_sig

    def assert_valid(self, value, node=None):
        raise NotImplementedError()

    def default(self):
        return None

    def construct(self, loader, node) -> 'value':
        raise NotImplementedError()

    def represent(self, dumper, value) -> 'node':
        raise NotImplementedError()

    def docstring(self, name: str) -> str:
        if self._doc_type_sig == "":
            type_sig = self.type_signature()
        else:
            type_sig = self._doc_type_sig

        return ":param {0}: {1}\n:type {0}: {2}".format(
            name, self._doc, type_sig)

    def type_signature(self):
        raise NotImplementedError()


class REQUIRED(ConfigField):
    def __init__(self, constraint, doc="", type_sig=""):
        super().__init__(doc=doc, type_sig=type_sig)
        if type(constraint) == OPTIONAL:
            raise ValueError("OPTIONAL in REQUIRED is ambiguous and therefore"
                             " not allowed.")

        self.constraint = Constraint.factory(constraint)

    def assert_valid(self, value, node=None):
        self.constraint.assert_valid(value, node)

    def construct(self, loader, node):
        value = self.constraint.construct(loader, node)
        self.assert_valid(value, node=node)
        return value

    def represent(self, dumper, value):
        node = self.constraint.represent(dumper, value)
        if isinstance(node, MappingNode):
            node.tag = 'tag:yaml.org,2002:map'
        return node

    def type_signature(self):
        return self.constraint.type_signature()


class OPTIONAL(REQUIRED):
    def __init__(self, constraint, default=None, doc="", type_sig=""):
        super().__init__(constraint, doc=doc, type_sig=type_sig)
        self._default = default

    def default(self):
        return self._default

    def construct(self, loader, node):
        try:
            value = super().construct(loader, node)
        except ValueError:
            return self._default

        return value

    def represent(self, dumper, value):
        if value is not None or value != self.default():
            return super().represent(dumper, value)

    def assert_valid(self, value, node=None):
        if value is None:
            return
        super().assert_valid(value, node=node)

    def type_signature(self):
        return "optional " + str(self.constraint)


class ENUM(ConfigField):
    def __init__(self, *args, doc="", type_sig=""):
        super().__init__(doc=doc, type_sig=type_sig)
        if len(args) < 2:
            raise ValueError("ENUM expects at least 2 elements.")
        if len(args) > len(set(args)):
            raise ValueError("Arguments of ENUM must be unique.")
        self.symbolic_strs = args

    def construct(self, loader, node):
        if not isinstance(node, ScalarNode):
            raise config_error(
                "Expected a string, but found `{}`".format(node), node)
        return str(node.value)

    def assert_valid(self, value, node=None):
        if value not in self.symbolic_strs:
            raise config_error(
                "Expect value to be one of {}, but got value `{}`".format(
                    ", ".join(["`{}`".format(s) for s in self.symbolic_strs]),
                    value,
                ), node)

    def represent(self, dumper, value):
        return dumper.represent_data(value)

    def type_signature(self):
        return " | ".join("`{}`".format(self.symbolic_strs))


class TAGS(ConfigField):
    def __init__(self, tag_types_dict, doc="", type_sig=""):
        super().__init__(doc=doc, type_sig=type_sig)

        if len(tag_types_dict) < 2:
            raise ValueError("TAGS expects at least 2 elements.")

        for tag, tpe in tag_types_dict.items():
            if type(tag) != str:
                raise ValueError(
                    "Expected tag to be a string,"
                    " but got tag of type `{}`".format(type(tag)))

        self.tag2constraint = {k: Constraint.factory(t)
                               for k, t in tag_types_dict.items()}

    def assert_valid(self, value, node=None):
        for tag, constraint in self.tag2constraint.items():
            try:
                constraint.assert_valid(value, node)
                return
            except ConfigError:
                pass
        raise config_error(
            "Expected value to be of `{}`, but got value `{}`"
            .format(", ".join(map(lambda x: x.type_signature(),
                                  self.tag2constraint.values())), value),
            node
        )

    def construct(self, loader, node):
        tag = node.tag.lstrip('!')
        if tag not in self.tag2constraint:
            raise config_error(
                "Tag `{}` not one of {}"
                .format(tag, ", ".join(self.tag2constraint.keys())),
                node)

        constraint = self.tag2constraint[tag]
        value = constraint.construct(loader, node)
        self.assert_valid(value, node)
        return value

    def represent(self, dumper, value):
        for tag, constraint in self.tag2constraint.items():
            try:
                constraint.assert_valid(value)
                node = constraint.represent(dumper, value)
                node.tag = tag
                return node
            except ConfigError:
                pass
        raise ConfigError("Could not find a suitable constraint for value `{}`"
                          .format(value))

    def type_signature(self):
        types = list(map(lambda t: str(t), self.tag2constraint.values()))

        if len(types) >= 2:
            return ",".join(types[:-2]) + " or " + str(types[-1])
        else:
            return str(types[0])


        for c in self.constraints:
class REPEAT(ConfigField):
    def __init__(self, tpe, doc="", type_sig=""):
        super().__init__(doc=doc, type_sig=type_sig)
        self.required_type = REQUIRED(tpe)

    def assert_valid(self, list, node=None):
        try:
            for item in list:
                self.required_type.assert_valid(item)
        except TypeError:
            raise config_error("Expected listlike object, but got `{}`"
                               .format(list), node)

    def represent(self, dumper, listlike):
        value = [self.required_type.represent(dumper, item)
                 for item in listlike]
        return SequenceNode('tag:yaml.org,2002:seq', value)

    def construct(self, loader, node):
        if isinstance(node, SequenceNode):
            return [self.required_type.construct(loader, item)
                    for item in node.value]

    def type_signature(self):
        return "list of " + self.required_type.type_signature()


class ConfigLoader(Loader):
    def get_single_config_data(self, cls):
        # Ensure that the stream contains a single document and construct it.
        node = self.get_single_node()
        if node is not None:
            return self.construct_config_document(cls, node)

    def construct_config_document(self, cls, node):
        data = self.construct_config_object(cls, node)
        self.constructed_objects = {}
        self.recursive_objects = {}
        self.deep_construct = False
        return data

    def construct_config_object(self, cls, node):
        return cls.from_yaml(self, node)


def load(cls, stream, Loader=ConfigLoader):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.
    """
    loader = Loader(stream)
    try:
        return loader.get_single_config_data(cls)
    finally:
        loader.dispose()


class ConfigDumper(Dumper):
    def represent(self, data):
        node = self.represent_data(data)
        node.tag = 'tag:yaml.org,2002:map'
        self.serialize(node)
        self.represented_objects = {}
        self.object_keeper = []
        self.alias_key = None


def dump(data, **kwargs):
    """
    Serialize a Python object into a YAML stream.
    If stream is None, return the produced string instead.
    """
    kwargs['default_flow_style'] = kwargs.get('default_flow_style', False)
    kwargs['Dumper'] = kwargs.get('Dumper', data.yaml_dumper)
    return dump_all([data], **kwargs)


def dump_all(documents, **kwargs):
    kwargs['Dumper'] = kwargs.get('Dumper', ConfigDumper)
    return yaml.dump_all(documents, **kwargs)


class ConfigObjectMetaclass(YAMLObjectMetaclass):
    """Meta class of :class:`.ConfigObject`."""

    def __init__(self, clsname, bases, kwds):
        super().__init__(clsname, bases, kwds)

    def __new__(cls, clsname, bases, my_fields):
        bases_fields = {}
        if bases is not None:
            bases_fields = {k: v for b in bases for k, v in b.__dict__.items()}

        if 'yaml_tag' not in my_fields or my_fields['yaml_tag'] is None:
            my_fields['yaml_tag'] = clsname

        fields = copy.copy(bases_fields)
        # update __config_fields__ form bases
        for n, f in bases_fields.items():
            if n == "__config_fields__":
                fields.update(f)

        fields.update(my_fields)
        config_fields = {name: copy.copy(field_def)
                         for name, field_def in fields.items()
                         if issubclass(type(field_def), ConfigField)}

        fields["__config_fields__"] = config_fields

        # delete all fields that are a subclass of ConfigField
        for name in config_fields.keys():
            del fields[name]

        new_cls = super(ConfigObjectMetaclass, cls). \
            __new__(cls, clsname, bases, fields)

        new_cls.__doc__ += cls._class_docstrings(new_cls)
        return new_cls

    @staticmethod
    def _docstrings(new_cls):
        return [
            field_def.docstring(attr_name)
            for attr_name, field_def
            in new_cls.__config_fields__.items()
        ]

    @staticmethod
    def _class_docstrings(new_cls):
        docstrings = ConfigObjectMetaclass._docstrings(new_cls)
        return "\n" + "\n".join(docstrings)


class ConfigObject(object, metaclass=ConfigObjectMetaclass):
    __config_fields__ = {}

    yaml_loader = ConfigLoader
    yaml_dumper = ConfigDumper

    yaml_tag = None

    def __init__(self, **kwargs):
        for field_name, field_def in self.__config_fields__.items():
            field_value = kwargs.get(field_name)
            self._add_property(field_name)
            self._set_property(field_name, field_value)

        valid_keys = list(self.__config_fields__.keys()) + ["__ctx__"]

        for k, arg in kwargs.items():
            if k not in valid_keys:
                raise ValueError("{!r} is not in the allowed keys `{!s}`"
                                 .format(k, valid_keys))

    def _add_property(self, name):
        def fget(self):
            return self._get_property(name)

        def fset(self, value):
            return self._set_property(name, value)

        setattr(self.__class__, name, property(fget, fset))

    def _get_property(self, name):
        return getattr(self, '_' + name)

    def _set_property(self, name, value, ctx=None):
        field_def = self.__config_fields__[name]
        field_def.assert_valid(value)
        if value is None:
            value = field_def.default()
        setattr(self, '_' + name, value)

    @classmethod
    def from_yaml(cls, loader, node):
        data = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key not in cls.__config_fields__:
                pass
            value = cls.__config_fields__[key].construct(loader, value_node)
            data[key] = value
        return cls(**data)

    @classmethod
    def to_yaml(cls, dumper, data):
        def to_dict_generator():
            for prop_name, config_field in data.__config_fields__.items():
                prop_value = data._get_property(prop_name)
                if prop_value is not None:
                    yield (dumper.represent_data(prop_name),
                           config_field.represent(dumper, prop_value))

        value = list(sorted(to_dict_generator(), key=lambda k: k[0].value))
        return MappingNode(data.yaml_tag, value)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        for prop_name in self.__config_fields__.keys():
            self_prop = getattr(self, '_' + prop_name)
            other_prop = getattr(other, '_' + prop_name)
            if not isinstance(self_prop,
                              type(other_prop)) or \
                    not isinstance(other_prop,
                                   type(self_prop)):
                return False
            if isinstance(self_prop, np.ndarray):
                if not np.all(self_prop == self_prop):
                    return False
            elif self_prop != other_prop:
                return False

        return True

    def __str__(self):
        fields = []
        for name, _ in self.__config_fields__.items():
            fields.append("{:}={:}".format(name, getattr(self, name)))

        fields.sort()
        return self.__class__.__name__ + "(" + ", ".join(fields) + ")"
