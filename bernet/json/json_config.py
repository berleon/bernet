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

try:
    import simplejson as json
except ImportError:
    import json


class JsonException(Exception):
    pass


class FromJsonContext():
    def __init__(self, raise_exceptions=False):
        self.stack = []
        self._errors = []
        self.raise_exceptions = raise_exceptions

    def n_errors(self):
        """number of errors occurred"""
        return len(self._errors)

    def _push(self, obj):
        self.stack.append(obj)

    def _pop(self):
        self.stack.pop()

    @contextmanager
    def step_into(self, pos: str):
        try:
            self._push(pos)
            yield
        finally:
            self._pop()

    def error(self, msg):
        idents = [elem for elem in self.stack]
        idents.reverse()
        parse_stack = "\n    ".join(idents)
        error_str = msg + '\n' + parse_stack
        self._errors.append(error_str)
        if self.raise_exceptions:
            raise JsonException(error_str)

    def errors_str(self):
        return "\n".join(self._errors)

    def warning(self):
        raise NotImplementedError()

    def print_report(self):
        if self.n_errors() != 0:
            err_str = self.errors_str()
            print("There occurred {:} errors \n".format(self.n_errors()) + err_str)


class JsonField(object):
    def construct(self, value, ctx=None):
        if ctx is None:
            ctx = FromJsonContext()
        with ctx.step_into(type(self).__name__):
            return self._construct(value, ctx)

    def valid(self, value):
        try:
            self.construct(value, FromJsonContext(raise_exceptions=True))
            return True
        except JsonException:
            return False

    def default(self):
        return None

    def _construct(self, value, ctx):
        return None

    def identifer(self):
        return "Constraint"

    def to_builtin(self, obj):
        """Converts `obj` to a built-in data type like dict, list, tuple or a
        primitive data type like int, float and string. """

        if hasattr(obj, '_to_builtin'):
            return obj._to_builtin()
        elif type(obj) in [int, float, str, dict]:
            return obj
        else:
            raise ValueError("Cannot convert object {:} to ".format())


class _TypeConstructableWrapper():
    def __init__(self, tpe):
        if type(tpe) == type or type(tpe) == _MetaJsonObject or \
                issubclass(type(tpe), JsonField):
            self.tpe = tpe
        else:
            raise ValueError("Expected a type or a subclass of JsonField or "
                             "JsonObject, but got {:}.".format(tpe))

    def construct(self, value, ctx):
        if type(value) == self.tpe:
            return value
        elif hasattr(self.tpe, 'construct'):
            return self.tpe.construct(value, ctx)
        else:
            ctx.error("Expected type `{:}` but found `type({:}) = {:}`"
                      .format(self.tpe, value, type(value)))


class REQUIRED(JsonField):
    def __init__(self, tpe, default=None):
        if type(tpe) == OPTIONAL:
            raise ValueError("OPTIONAL in REQUIRED is ambiguous and"
                             " not allowed")

        self.tpe = _TypeConstructableWrapper(tpe)
        self._default = default

    def _construct(self, value, ctx):
        return self.tpe.construct(value, ctx)


def _subclass_of_json_field(tpe):
    return hasattr(tpe, 'construct') and issubclass(type(tpe), JsonField)


class OPTIONAL(JsonField):
    def __init__(self, tpe, default=None):
        self.tpe = _TypeConstructableWrapper(tpe)
        self._default = default

    def default(self):
        return self._default

    def _construct(self, value, ctx):
        if value is None:
            return

        return self.tpe.construct(value, ctx)


class EITHER(JsonField):
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("EITHER expects at least 2 elements")

        self.types = list(map(lambda t: _TypeConstructableWrapper(t),
                              filter(lambda t: type(t) == type, args)))
        self.fix_values = list(filter(lambda el: type(el) != type, args))

    def _construct(self, value, ctx):
        constructed_values = [t.construct(value, ctx) for t in self.types]

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
                n_minus_one = ", ".join(true_values[:-1])
                ctx.error("Value `{:}` satisfies {:} and {:} "
                          .format(value, n_minus_one, true_values[-1]))
            else:  # no one matched
                assert len(true_values) == 0
                n_minus_one = ", ".join(all[:-1])
                ctx.error("Value `{:}` doesn't satisfy any of {:} or {:}"
                          .format(value, n_minus_one, all[-1]))
        else:
            if constructed_values:
                return constructed_values[0]
            else:
                return value


class REPEAT(JsonField):
    def __init__(self, tpe):
        self.tpe = _TypeConstructableWrapper(tpe)

    def default(self):
        return []

    def _construct(self, listlike, ctx):
        if listlike is None:
            return

        try:
            listlike = iter(listlike)
        except TypeError:
            ctx.error("Expected listlike type but got type {:}"
                      .format(type(listlike)))
            return

        # TODO: enter a new

        constructed_list = []
        for i, el in enumerate(listlike):
            with ctx.step_into("at Position {:}".format(i)):
                constructed_list.append(self.tpe.construct(el, ctx))

        return list(constructed_list)

    def to_builtin(self, obj):
        return [super(REPEAT, self).to_builtin(item) for item in obj]


class _MetaJsonObject(type):
    def __new__(cls, clsname, bases, fields):
        config_fields = {n: field_def for n, field_def in fields.items()
                         if issubclass(type(field_def), JsonField)}
        fields["__config_fields__"] = config_fields
        return super(_MetaJsonObject, cls).__new__(cls, clsname, bases, fields)


class JsonObject(object, metaclass=_MetaJsonObject):
    __config_fields__ = {}

    def __init__(self, **kwargs):
        # TODO: check for keys in kwargs, but who are not in self.ATTRIBUTES
        ctx = FromJsonContext(raise_exceptions=True)
        for attr_name, attr_def in self.__config_fields__.items():
            input_value = kwargs.get(attr_name)
            construct_value = attr_def.construct(input_value, ctx)
            if construct_value is None:
                construct_value = attr_def.default()

            setattr(self, attr_name, construct_value)

    def _to_builtin(self):
        def to_dict_generator():
            for k, v in self.__dict__.items():
                if k in self.__config_fields__:
                    validable = self.__config_fields__[k]
                    yield k, validable.to_builtin(v)

        return {k: v for k, v in to_dict_generator()}

    @classmethod
    def construct(cls, obj, ctx):
        for attr_name, field in cls.__config_fields__.items():
            obj[attr_name] = field.construct(obj[attr_name], ctx)
        return cls(**obj)

    @classmethod
    def loads(cls, str, **kwargs):
        raw_obj = json.loads(str)
        ctx = FromJsonContext()
        obj = cls.construct(raw_obj, ctx)
        if ctx.n_errors() != 0:
            ctx.print_report()
            return ValueError("Could not load {:}".format(cls.__name__))
        return obj

    def dumps(self, **kwargs):
        dict = self._to_builtin()
        return json.dumps(dict, kwargs)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        for attr in self.__config_fields__:
            if self.__dict__[attr] != other.__dict__[attr]:
                return False

        return True


def json_object(name, fields):
    return _MetaJsonObject(name, [JsonObject], fields)
