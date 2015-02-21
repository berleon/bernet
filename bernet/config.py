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

from contextlib import contextmanager
import copy
import inspect

import numpy as np

try:
    import simplejson as json
except ImportError:
    import json


class ConfigException(Exception):
    pass


class InitContext():
    def __init__(self, raise_exceptions=False):
        self._stack = []
        self._error_stack = []
        self._errors = []
        self.raise_exceptions = raise_exceptions

    def n_errors(self):
        """number of errors occurred"""
        return len(self._errors)

    def _push(self, obj):
        self._stack.append(obj)

    def _pop(self):
        self._stack.pop()

    @contextmanager
    def step_into(self, obj, name: str):
        try:
            self._push((obj, name))
            yield
        finally:
            self._pop()

    def error(self, msg):
        idents = [(obj + ": ").ljust(10) + name for obj, name in self._stack]
        idents.reverse()
        parse_stack = ""
        if len(idents) > 0:
            parse_stack = "\nTraceback:\n    " + \
                          ",\n    ".join(idents)

        error_str = msg + parse_stack
        self._errors.append(error_str)
        if self.raise_exceptions:
            raise ConfigException(error_str)

    def errors_str(self):
        return "\n".join(self._errors)

    def warning(self):
        raise NotImplementedError()

    def print_report(self):
        if self.n_errors() != 0:
            err_str = self.errors_str()
            print("There occurred {:} errors \n"
                  .format(self.n_errors()) + err_str)

    def error_in_scope(self):
        return self._errors


class ConfigField(object):
    def __init__(self, doc="", type_sig=""):
        self._doc = doc
        self._doc_type_sig = type_sig

    def construct(self, value, ctx=None):
        if ctx is None:
            ctx = InitContext()
        return self._construct(value, ctx)

    def valid(self, value):
        try:
            self.construct(value, ctx=InitContext(raise_exceptions=True))
            return True
        except ConfigException:
            return False

    def default(self):
        return None

    def _construct(self, value, ctx):
        raise NotImplementedError("Use a subclass of ConfigField")

    def to_builtin(self, obj):
        """Converts `obj` to a built-in data type like dict, list, tuple or a
        primitive data type like int, float and string. """

        if hasattr(obj, '_to_builtin'):
            return obj._to_builtin()
        elif type(obj) in [int, float, str, dict]:
            return obj
        else:
            raise ValueError("Cannot convert object {:} to ".format())

    def traceback_type(self):
        return "Field"

    def docstring(self, name: str) -> str:
        if self._doc_type_sig == "":
            type_sig = self._type()
        else:
            type_sig = self._doc_type_sig

        return ":param {0}: {1}\n:type {0}: {2}".format(
            name, self._doc, type_sig)

    def _type(self):
        raise NotImplementedError()


def _is_type(tpe):
    return tpe == type or tpe == _ConfigObjectType


class _TypeConstructableWrapper():
    def __init__(self, obj):
        if _is_type(type(obj)) or issubclass(type(obj), ConfigField):
            self.obj = obj
        else:
            raise ValueError("Expected a type or a subclass of JsonField or "
                             "JsonObject, but got {:}.".format(obj))

    def construct(self, value, ctx):
        if type(value) == self.obj:
            return value
        elif inspect.isclass(self.obj) and issubclass(self.obj, ConfigObject) \
                and (issubclass(type(value), dict) or isinstance(value, dict)):
            return self.obj.new_with_ctx(ctx, value)
        elif hasattr(self.obj, 'construct'):
            return self.obj.construct(value, ctx)
        else:
            ctx.error("Expected type `{:}`, but got value `{:}` with "
                      "type `{:}`."
                      .format(self.obj.__name__, value, type(value).__name__))

    def type_doc(self):
        if _is_type(type(self.obj)):
            return ":class:`.{:}`".format(self)
        elif issubclass(type(self.obj), ConfigField):
            return self.obj._type()
        else:
            return str(self)

    def __str__(self):
        if _is_type(type(self.obj)):
            return self.obj.__name__

        return str(self.obj)


class REQUIRED(ConfigField):
    def __init__(self, tpe, doc="", type_sig=""):
        super().__init__(doc=doc, type_sig=type_sig)
        if type(tpe) == OPTIONAL:
            raise ValueError("OPTIONAL in REQUIRED is ambiguous and"
                             " not allowed")
        self.tpe = _TypeConstructableWrapper(tpe)
        self._default = None

    def _construct(self, value, ctx):
        return self.tpe.construct(value, ctx)

    def _type(self):
        return self.tpe.type_doc()


def _subclass_of_json_field(tpe):
    return hasattr(tpe, 'construct') and issubclass(type(tpe), ConfigField)


class OPTIONAL(ConfigField):
    def __init__(self, tpe, default=None, doc="", type_sig=""):
        super().__init__(doc=doc, type_sig=type_sig)
        self.tpe = _TypeConstructableWrapper(tpe)
        self._default = default

    def default(self):
        return self._default

    def _construct(self, value, ctx):
        if value is None:
            return self.default()

        return self.tpe.construct(value, ctx)

    def _type(self):
        return "optional " + str(self.tpe)


class EITHER(ConfigField):
    def __init__(self, *args, doc="", type_sig=""):
        super().__init__(doc=doc, type_sig=type_sig)
        if len(args) < 2:
            raise ValueError("EITHER expects at least 2 elements")
        if len(args) > len(set(args)):
            raise ValueError("arguments of EITHER must be unique")
        self.types = list(map(lambda t: _TypeConstructableWrapper(t),
                              filter(lambda t: _is_type(type(t)), args)))
        self.fix_values = list(filter(lambda el: not _is_type(type(el)), args))

    def _construct(self, value, ctx):
        constructed_values = []
        for t in self.types:
            try:
                constructed_values.append(t.construct(value, ctx))
            except ConfigException:
                constructed_values.append(None)

        truths = [c is not None for c in constructed_values] + \
                 [value == fix for fix in self.fix_values]

        # one and only one true truth :)
        if truths.count(True) != 1:
            all = self.types + self.fix_values
            true_values = []
            for i, boolean in enumerate(truths):
                if boolean:
                    true_values.append(all[i])

            if true_values:
                assert len(true_values) >= 2
                n_minus_one = ", ".join(map(str, true_values[:-1]))
                ctx.error("Value `{:}` satisfies {:} and {:} "
                          .format(value, n_minus_one, true_values[-1]))
            else:  # no one matched
                assert 0 == len(true_values)
                n_minus_one = "`, `".join(map(str, all[:-1]))
                ctx.error("Value `{:}` doesn't satisfy any of `{:}` or `{:}`"
                          .format(str(value), n_minus_one, all[-1]))
        else:
            if constructed_values:
                return constructed_values[0]
            else:
                return value

    def _type(self):
        types = list(map(lambda t: t.type_doc(), self.types)) + \
            list(map(str, self.fix_values))
        if len(types) >= 2:
            return ",".join(types[:-2]) + " or " + str(types[-1])
        else:
            return str(types[0])


class REPEAT(ConfigField):
    def __init__(self, tpe, doc="", type_sig=""):
        super().__init__(doc=doc, type_sig=type_sig)
        self.tpe = _TypeConstructableWrapper(tpe)

    def default(self):
        return []

    def _construct(self, listlike, ctx):
        if listlike is None:
            return

        if type(listlike) == str:
            ctx.error("Expected a list, but got a str.")
            return

        try:
            listlike = iter(listlike)
        except TypeError:
            ctx.error("Expected a listlike type, but got type {:}."
                      .format(type(listlike).__name__))
            return

        constructed_list = []
        for i, el in enumerate(listlike):
            with ctx.step_into("List", "at element {:}".format(i)):
                constructed_list.append(self.tpe.construct(el, ctx))

        return list(constructed_list)

    def to_builtin(self, obj):
        return [super(REPEAT, self).to_builtin(item) for item in obj]

    def _type(self):
        return "list of " + self.tpe.type_doc()


class SUBCLASS_OF(ConfigField):
    def __init__(self, tpe, doc="", type_sig=""):
        super().__init__(doc=doc, type_sig=type_sig)
        if not _is_type(type(tpe)):
            raise AssertionError("SUBCLASS_OF requires a type. Got `{:}`"
                                 .format(tpe))
        self.tpe = tpe

    def _construct(self, value, ctx):
        if issubclass(type(value), self.tpe):
            return value
        else:
            ctx.error("Got value `{:}` of type `{:}`. Expected a subclass of "
                      "`{:}``"
                      .format(value, type(value).__name__, self.tpe.__name__))

    def _type(self):
        return "subclass of :class:`.{:}`".format(self.tpe.__name__)


class _ConfigObjectType(type):
    """Meta class of :class:`.ConfigObject`."""
    def __new__(cls, clsname, bases, my_fields):
        if bases is not None:
            bases_fields = {k: v for b in bases for k, v in b.__dict__.items()}
        else:
            bases_fields = {}

        fields = copy.copy(bases_fields)
        # update __config_fields__ form bases
        for n, f in bases_fields.items():
            if n == "__config_fields__":
                fields.update(f)

        fields.update(my_fields)
        config_properties = {name: copy.copy(field_def)
                             for name, field_def in fields.items()
                             if issubclass(type(field_def), ConfigField)}

        fields["__config_fields__"] = config_properties

        # delete all fields that are a subclass of ConfigField
        for name in config_properties.keys():
            del fields[name]

        new_cls = super(_ConfigObjectType, cls).\
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
        docstrings = _ConfigObjectType._docstrings(new_cls)
        return "\n" + "\n".join(docstrings)


class ConfigObject(object, metaclass=_ConfigObjectType):
    __config_fields__ = {}

    def __init__(self, **kwargs):
        ctx = self._get_ctx(kwargs)
        with ctx.step_into("Object", type(self).__name__):
            for field_name, field_def in self.__config_fields__.items():
                field_value = kwargs.get(field_name)
                with ctx.step_into(field_def.traceback_type(), field_name):
                    # constructed_value = field_def.construct(field_value)
                    self._add_property(field_name)
                    self._set_property(field_name, field_value, ctx=ctx)

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
        if ctx is None:
            ctx = InitContext(raise_exceptions=True)

        constructed_val = field_def.construct(value, ctx)
        if constructed_val is None:
            constructed_val = field_def.default()

        setattr(self, '_' + name, constructed_val)

    @staticmethod
    def _get_ctx(kwargs):
        return kwargs.get("__ctx__", InitContext(raise_exceptions=True))

    def _to_builtin(self):
        def to_dict_generator():
            for prop_name, definiton in self.__config_fields__.items():
                prop_value = self._get_property(prop_name)
                yield prop_name, definiton.to_builtin(prop_value)

        return {k: v for k, v in to_dict_generator()}

    @classmethod
    def new_with_ctx(cls, ctx, obj):
        obj["__ctx__"] = ctx
        return cls(**obj)

    @classmethod
    def load_json(cls, fp, **kwargs) -> 'cls':
        raw_obj = json.load(fp)
        return cls._loads_json_from_raw_obj(raw_obj)

    @classmethod
    def loads_json(cls, str, **kwargs) -> 'cls':
        raw_obj = json.loads(str)
        return cls._loads_json_from_raw_obj(raw_obj)

    @classmethod
    def _loads_json_from_raw_obj(cls, raw_obj):
        ctx = InitContext(raise_exceptions=True)
        raw_obj["__ctx__"] = ctx
        obj = cls(**raw_obj)
        if ctx.n_errors() != 0:
            ctx.print_report()
            return ValueError("Could not load {:}".format(cls.__name__))
        return obj

    def to_json(self, **kwargs):
        dict = self._to_builtin()
        return json.dumps(dict, kwargs)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        for prop_name in self.__config_fields__.keys():
            self_prop = getattr(self, '_'+prop_name)
            other_prop = getattr(other, '_'+prop_name)
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

        return self.__class__.__name__ + "(" + ",".join(fields) + ")"
