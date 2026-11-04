"""Translate the experimental public types into real Bilt compiler declarations."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
import inspect
import types
from typing import Annotated, TypeAliasType, TypeVar, Union, cast, get_args, get_origin, get_type_hints
import bilt
from bilt import Interface, Dim
from bilt.units import Quantity as NativeQuantity
from bilt.units.dimensions import ABSOLUTE
from bilt.core._equation import BasicEquation
from bilt.symbolic._expressions import Expr
from ._markers import DimensionType
from ._values import Quantity, Tensor, from_native, to_native


@dataclass(frozen=True)
class Requires:
    axis: Dim


@dataclass(frozen=True)
class Contract:
    dimension: type[DimensionType]
    tensor: bool | None
    required: tuple[Dim, ...]

    def native(self):
        base = bilt.Tensor if self.tensor is True else NativeQuantity
        metadata = [self.dimension.runtime]
        if self.dimension.absolute:
            metadata.append(ABSOLUTE)
        return Annotated[base, *metadata]

    def validate(self, value):
        if not isinstance(value, Quantity) or value.unit.dimension is not self.dimension:
            raise TypeError(f"Expected {self.dimension.__name__} quantity")
        is_tensor = isinstance(value.value, Tensor)
        if self.tensor is not None and self.tensor != is_tensor:
            raise TypeError("Scalar/tensor representation does not match the declared port")
        axes = value.value.domain.dims if is_tensor else ()
        if not set(self.required).issubset(axes):
            raise TypeError("Missing required tensor axis")


def contract(annotation):
    required = []

    def resolve(a, env):
        if isinstance(a, TypeVar) and a in env:
            return resolve(env[a], env)
        if isinstance(a, TypeAliasType):
            return resolve(a.__value__, env)
        origin, args = get_origin(a), get_args(a)
        if isinstance(origin, TypeAliasType):
            mapping = {k: env.get(v, v) for k, v in zip(origin.__type_params__, args, strict=True)}
            return resolve(origin.__value__, env | mapping)
        if origin is Annotated:
            required.extend(x.axis for x in args[1:] if isinstance(x, Requires))
            return resolve(args[0], env)
        if origin is Quantity:
            dim = env.get(args[0], args[0])
            return dim, resolve(args[1], env)
        if origin is Tensor:
            return True
        if origin in (Union, types.UnionType):
            reps = {resolve(x, env) for x in args}
            if reps == {False, True}:
                return None
            raise TypeError("Unsupported magnitude union")
        if a is float:
            return False
        raise TypeError(f"Unsupported annotation: {a}")

    dim, rep = resolve(annotation, {})
    if required and rep is not True:
        raise TypeError("Requires must annotate a tensor-only quantity")
    return Contract(dim, rep, tuple(required))


def equation[**P, R](function: Callable[P, R]) -> BasicEquation[R, P]:
    annotations = get_type_hints(function, include_extras=True)
    converted = {n: contract(a).native() for n, a in annotations.items() if n != "return"}
    converted["return"] = annotations.get("return", bool)
    # Clone the function so callers and checkers retain the original signature.
    clone = types.FunctionType(
        function.__code__, function.__globals__, function.__name__, function.__defaults__, function.__closure__
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = converted
    compiled = bilt.equation(clone)
    compiled.__prototype_contracts__ = {n: contract(a) for n, a in annotations.items() if n != "return"}
    return cast(BasicEquation[R, P], compiled)


def transform(function):
    annotations = get_type_hints(function, include_extras=True)
    contracts = {n: contract(a) for n, a in annotations.items()}
    clone = types.FunctionType(
        function.__code__, function.__globals__, function.__name__, function.__defaults__, function.__closure__
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = {n: c.native() for n, c in contracts.items()}
    compiled = bilt.transform(clone)
    compiled.__prototype_contracts__ = {n: c for n, c in contracts.items() if n != "return"}
    return compiled


def system[T: Interface](cls: T) -> T:
    hints = get_type_hints(cls, include_extras=True)
    contracts = {name: contract(a) for name, a in hints.items()}
    cls.__annotations__ = {name: c.native() for name, c in contracts.items()}
    bilt.system(cls)
    native_init = cls.__init__

    def init(self, **kwargs):
        for name, value in kwargs.items():
            if name in contracts:
                contracts[name].validate(value)
        native_init(self, **{name: to_native(value) for name, value in kwargs.items()})

    cls.__init__ = init
    # Keep Bilt's internal values native. Only public attribute reads wrap them.
    for name, c in contracts.items():
        native_property = getattr(cls, name)

        def getter(self, prop=native_property, declaration=c):
            result = from_native(prop.fget(self))
            declaration.validate(result)
            return result

        setattr(cls, name, property(getter))
    cls.__prototype_contracts__ = contracts
    return cls


class Implementation(bilt.Implementation):
    def __init__(self, system, components):
        components = tuple(components)
        # Required axes on equations are checked at the evaluation boundary too.
        self._prototype_requirements = {}
        for component in components:
            for name, c in getattr(component, "__prototype_contracts__", {}).items():
                if c.required:
                    self._prototype_requirements.setdefault(name, set()).update(c.required)
        super().__init__(system, components)

    def evaluate(self, inputs, output, backend=None):
        native = {name: to_native(value) for name, value in inputs.items()}
        for name, axes in self._prototype_requirements.items():
            if name in native:
                value = native[name]
                if not isinstance(value, bilt.Tensor) or not axes.issubset(value.domain.dims):
                    raise TypeError(f"Missing required tensor axis on {name}")
        result = super().evaluate(native, output, backend or bilt.PYTHON_BACKEND)
        declaration = self.system.__prototype_contracts__.get(output)
        if declaration is not None:
            declaration.validate(from_native(result))
        return result


def equal(left, right):
    if isinstance(left, Expr) or isinstance(right, Expr):
        return left == right
    if not isinstance(left, Quantity) or not isinstance(right, Quantity):
        raise TypeError("equal requires quantities")
    if left.unit.dimension is not right.unit.dimension:
        raise TypeError("Incompatible equation dimensions")
    return left == right


def pointwise(function):
    hints = get_type_hints(function, include_extras=True)
    params = list(inspect.signature(function).parameters)
    if len(params) != 1:
        raise TypeError("Prototype callable pointwise lifting supports unary kernels only")
    incoming, outgoing = contract(hints[params[0]]), contract(hints["return"])
    if incoming.tensor is not False or outgoing.tensor is not False:
        raise TypeError("Pointwise kernels must declare scalar quantities")
    if incoming.dimension is not outgoing.dimension:
        raise TypeError("Prototype unary wrapper requires a dimension-preserving kernel")

    @wraps(function)
    def wrapped(value):
        import numpy as np

        if not isinstance(value, Quantity) or value.unit.dimension is not incoming.dimension:
            raise TypeError("Wrong pointwise input dimension")
        if not isinstance(value.value, Tensor):
            result = function(value)
            outgoing.validate(result)
            return result
        # Choose output units independently of executing a sample, including empty arrays.
        unit = value.unit
        data = np.empty_like(value.value.values)
        for index in np.ndindex(data.shape):
            result = function(Quantity(float(value.value.values[index]), unit))
            outgoing.validate(result)
            data[index] = result.convert_to(unit).value
        return Quantity(Tensor(data, value.value.domain), unit)

    return wrapped
