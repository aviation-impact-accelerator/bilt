"""Instance-level derivatives of scalar model outputs."""

import dataclasses
import typing

from bilt._backend._base_backend import _backend_stack
from bilt._backend._python import PYTHON_BACKEND
from bilt.core._system._errors import BiltUnknownSystemAttributeError

from ._sentinel import MISSING, _Missing

if typing.TYPE_CHECKING:
    from bilt.core._interface import Interface

    from ._instance import SystemInstance


type FieldSelector = property | str
type FieldSelectors = tuple[FieldSelector, ...]


def _field_name(instance: SystemInstance, selector: FieldSelector) -> str:
    cls = typing.cast("Interface", type(instance))
    if isinstance(selector, str):
        name = selector
    elif isinstance(selector, property):
        matches = [port for port in cls.__bilt_system_attributes__ if getattr(cls, port) is selector]
        if len(matches) != 1:
            msg = "The field selector must be a property on this system interface."
            raise ValueError(msg)
        name = matches[0]
    else:
        msg = "Field selectors must be system field properties or their names."
        raise TypeError(msg)
    if name not in cls.__bilt_system_attributes__:
        msg = f"Unknown differentiation field {name!r} on {cls.__name__}."
        raise ValueError(msg)
    return name


def _selected_implementation(instance: SystemInstance, output: str) -> object:
    cls = typing.cast("Interface", type(instance))
    implementation = instance.__bilt_system_implementation__
    if isinstance(implementation, _Missing):
        if not cls.__bilt_system_implementation_default__:
            raise BiltUnknownSystemAttributeError(output, instance)
        implementation = cls.__bilt_system_implementation_default__[-1]
        instance.__bilt_system_implementation__ = implementation
    return implementation


def _inputs(instance: SystemInstance) -> dict[str, object]:
    return {
        field: value for field, value in instance.__bilt_system_instance_inputs__.items() if value is not MISSING
    }


def _fields(instance: SystemInstance, selectors: FieldSelectors, *, label: str) -> tuple[str, ...]:
    if not isinstance(selectors, tuple) or not selectors:
        msg = f"{label} requires a non-empty tuple of system fields."
        raise ValueError(msg)
    names = tuple(_field_name(instance, selector) for selector in selectors)
    if len(names) != len(set(names)):
        msg = f"{label} contains duplicate fields."
        raise ValueError(msg)
    return names


@dataclasses.dataclass(frozen=True)
class DerivativeQuery:
    """A field selected for a scalar sensitivity, gradient, Hessian or Jacobian."""

    instance: SystemInstance
    output: str
    kind: str = "d"

    def __call__(self, *, wrt: FieldSelector | FieldSelectors) -> object:
        implementation = typing.cast("Implementation", _selected_implementation(self.instance, self.output))
        backend = _backend_stack[-1] if _backend_stack else PYTHON_BACKEND
        inputs = _inputs(self.instance)

        if self.kind == "d":
            name = _field_name(self.instance, typing.cast("FieldSelector", wrt))
            return implementation.evaluate_derivative(inputs=inputs, output=self.output, wrt=name, backend=backend)

        names = _fields(self.instance, typing.cast("FieldSelectors", wrt), label="wrt")
        if self.kind == "grad":
            return tuple(
                implementation.evaluate_derivative(inputs=inputs, output=self.output, wrt=name, backend=backend)
                for name in names
            )
        if self.kind == "jacobian":
            row = tuple(
                implementation.evaluate_derivative(inputs=inputs, output=self.output, wrt=name, backend=backend)
                for name in names
            )
            return (row,)
        if self.kind == "hessian":
            return tuple(
                tuple(
                    implementation.evaluate_second_derivative(
                        inputs=inputs, output=self.output, wrt=name, wrt2=other, backend=backend
                    )
                    for other in names
                )
                for name in names
            )
        msg = f"Unknown differentiation namespace {self.kind!r}."
        raise ValueError(msg)


class DerivativeNamespace:
    """Select a scalar output with instance.grad.field(wrt=(System.x, System.y))."""

    def __init__(self, instance: SystemInstance, kind: str = "d") -> None:
        self._instance = instance
        self._kind = kind

    def __getattr__(self, name: str) -> DerivativeQuery:
        cls = typing.cast("Interface", type(self._instance))
        if name not in cls.__bilt_system_attributes__:
            msg = f"{cls.__name__} has no field {name!r}."
            raise AttributeError(msg)
        return DerivativeQuery(self._instance, name, self._kind)

    def __call__(
        self, outputs: FieldSelectors, *, wrt: FieldSelectors
    ) -> tuple[tuple[object, ...], ...]:
        """Build a Jacobian matrix with ordered rows and columns."""
        if self._kind != "jacobian":
            msg = "Multiple outputs are supported only by the jacobian namespace."
            raise TypeError(msg)
        output_names = _fields(self._instance, outputs, label="outputs")
        wrt_names = _fields(self._instance, wrt, label="wrt")
        implementation = typing.cast("Implementation", _selected_implementation(self._instance, output_names[0]))
        backend = _backend_stack[-1] if _backend_stack else PYTHON_BACKEND
        inputs = _inputs(self._instance)
        return tuple(
            tuple(
                implementation.evaluate_derivative(inputs=inputs, output=output, wrt=name, backend=backend)
                for name in wrt_names
            )
            for output in output_names
        )


if typing.TYPE_CHECKING:
    from ._implementation import Implementation
