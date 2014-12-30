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

    def _push(self, obj):
        self.stack.append(obj)

    def _pop(self):
        self.stack.pop()

    @contextmanager
    def step_into(self, obj):
        try:
            self._push(obj)
            yield
        finally:
            self._pop()

    def error(self, msg):
        idents = [elem.identifer() for elem in self.stack]
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


class JsonField(object):
    def construct(self, value, ctx=None):
        if ctx is None:
            ctx = FromJsonContext()
        with ctx.step_into(self):
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


class REQUIRED(JsonField):
    def __init__(self, tpeOrConstructable, default=None):
        self.tpe = None
        self.constructable = None
        self._default = default
        if type(tpeOrConstructable) == type:
            self.tpe = tpeOrConstructable
        elif issubclass(type(tpeOrConstructable), JsonField):
            if type(tpeOrConstructable) == OPTIONAL:
                raise ValueError("OPTIONAL in REQUIRED is ambiguous and"
                                 " not allowed")
            self.constructable = tpeOrConstructable
        else:
            raise ValueError("{:} must be either a type or a subclass "
                             "of Validable")

    def _construct(self, value, ctx):
        if self.tpe is not None:
            if type(value) != self.tpe:
                ctx.error("{:} should be of type {:}, but is {:}"
                          .format(value, self.tpe, type(value)))
            else:
                return value
        else:
            return self.constructable.construct(value, ctx)


def _constructable(tpe):
    return hasattr(tpe, 'construct') or issubclass(type(tpe), JsonField)


class OPTIONAL(JsonField):
    def __init__(self, tpeOrValidable, default=None):
        self.type = None
        self.constructable = None

        if _constructable(tpeOrValidable):
            self.constructable = tpeOrValidable
        else:
            self.type = tpeOrValidable
        self._default = default

    def default(self):
        return self._default

    def _construct(self, value, ctx):
        if value is None:
            return

        if self.type is not None:
            if type(value) != self.type:
                ctx.error("Expected type {:}. Got {:}"
                          .format(self.type, type(value)))
            else:
                return value
        else:
            return self.constructable.construct(value, ctx)


class EITHER(JsonField):
    def __init__(self, *list):
        if len(list) < 2:
            raise ValueError("EITHER expects at least 2 elements")
        self.constructable = set([c for c in list if _constructable(c)])
        self.fix_values = set(list) - set(self.constructable)

    def _construct(self, value, ctx):
        constructed_values = [c.construct(value, ctx)
                              for c in self.constructable]
        truths = [c is not None for c in constructed_values] + \
                 [value == fix for fix in self.fix_values]

        # one and only one true truth :)
        if truths.count(True) != 1:
            all = list(self.constructable | self.fix_values)
            n_minus_one = ", ".join(all[:-1])
            ctx.error("Expected {:} or {:}, but got `{:}'"
                      .format(n_minus_one, all[-1], value))
        else:
            if constructed_values:
                return constructed_values[0]
            else:
                return value


class REPEAT(JsonField):
    def __init__(self, tpe):
        self.tpe = tpe

    def default(self):
        return []

    def _construct(self, listlike, ctx):
        if listlike is None:
            return

        # TODO: check for iterable
        if type(listlike) not in [list, tuple]:
            ctx.error("Expected list type but got {:}".format(type(listlike)))

        if _constructable(self.tpe):
            constructed_list = [self.tpe.construct(el) for el in listlike
                                if type(el) != self.tpe]
            listlike = constructed_list + [el for el in listlike
                                           if type(el) == self.tpe]

        wrong_el = [el for el in listlike if type(el) is not self.tpe]
        if len(wrong_el) != 0:
            types_msg = ["type({:}) = {:}".format(el, type(el))
                         for el in wrong_el]

            ctx.error("Elements should be of type {:}, " .format(self.tpe) +
                      "but found types: {:}".format(", ".join(types_msg)))
        return list(listlike)

    def to_builtin(self, obj):
        return [super(REPEAT, self).to_builtin(item) for item in obj]


class _MetaJsonObject(type):
    def __new__(cls, clsname, bases, fields):
        json_fields = {n: field_def for n, field_def in fields.items()
                       if issubclass(type(field_def), JsonField)}
        fields["__json_fields__"] = json_fields
        return super(_MetaJsonObject, cls).__new__(cls, clsname, bases, fields)


class JsonObject(object, metaclass=_MetaJsonObject):
    __json_fields__ = {}

    def __init__(self, **kwargs):
        # TODO: check for keys in kwargs, but who are not in self.ATTRIBUTES
        ctx = FromJsonContext(raise_exceptions=True)
        for attr_name, attr_def in self.__json_fields__.items():
            input_value = kwargs.get(attr_name)
            construct_value = attr_def.construct(input_value, ctx)
            if construct_value is None:
                construct_value = attr_def.default()

            setattr(self, attr_name, construct_value)

    def _to_builtin(self):
        def to_dict_generator():
            for k, v in self.__dict__.items():
                if k in self.__json_fields__:
                    validable = self.__json_fields__[k]
                    yield k, validable.to_builtin(v)

        return {k: v for k, v in to_dict_generator()}

    @classmethod
    def construct(cls, obj, ctx=None):
        for attr_name, validable in cls.__json_fields__.items():
            obj[attr_name] = validable.construct(obj[attr_name], ctx)
        return cls(**obj)

    @classmethod
    def loads(cls, str, **kwargs):
        obj = json.loads(str)
        return cls.construct(obj)

    def dumps(self, **kwargs):
        dict = self._to_builtin()
        return json.dumps(dict, kwargs)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        for attr in self.__json_fields__:
            if self.__dict__[attr] != other.__dict__[attr]:
                return False

        return True


def json_object(name, fields):
    return _MetaJsonObject(name, [JsonObject], fields)
