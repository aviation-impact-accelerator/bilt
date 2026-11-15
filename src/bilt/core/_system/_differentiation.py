"""Derivative queries on system instances."""

import dataclasses
import typing

from bilt._backend._base_backend import _backend_stack
from bilt._backend._python import PYTHON_BACKEND
from bilt.core._system._errors import BiltUnknownSystemAttributeError

from ._sentinel import MISSING, _Missing

if typing.TYPE_CHECKING:
    from bilt.core._interface import Interface

    from ._instance import SystemInstance


@dataclasses.dataclass(frozen=True)
class DerivativeQuery:
    """A system field selected for a derivative query."""

    instance: SystemInstance
    output: str

    def __call__(self, *, wrt: property | str) -> object:
        cls = typing.cast("Interface", type(self.instance))
        if isinstance(wrt, str):
            name = wrt
        elif isinstance(wrt, property):
            matches = [port for port in cls.__bilt_system_attributes__ if getattr(cls, port) is wrt]
            if len(matches) != 1:
                msg = "wrt must refer to a property on this system interface."
                raise ValueError(msg)
            name = matches[0]
        else:
            msg = "wrt must be a system field property or its name."
            raise TypeError(msg)
        if name not in cls.__bilt_system_attributes__:
            msg = f"Unknown differentiation field {name!r} on {cls.__name__}."
            raise ValueError(msg)

        implementation = self.instance.__bilt_system_implementation__
        if isinstance(implementation, _Missing):
            if not cls.__bilt_system_implementation_default__:
                raise BiltUnknownSystemAttributeError(self.output, self.instance)
            implementation = cls.__bilt_system_implementation_default__[-1]
            self.instance.__bilt_system_implementation__ = implementation

        inputs = {
            field: value
            for field, value in self.instance.__bilt_system_instance_inputs__.items()
            if value is not MISSING
        }
        backend = _backend_stack[-1] if _backend_stack else PYTHON_BACKEND
        return implementation.evaluate_derivative(inputs=inputs, output=self.output, wrt=name, backend=backend)


class DerivativeNamespace:
    """Select an output using instance.d.field(wrt=System.input)."""

    def __init__(self, instance: SystemInstance) -> None:
        self._instance = instance

    def __getattr__(self, name: str) -> DerivativeQuery:
        cls = typing.cast("Interface", type(self._instance))
        if name not in cls.__bilt_system_attributes__:
            msg = f"{cls.__name__} has no field {name!r}."
            raise AttributeError(msg)
        return DerivativeQuery(self._instance, name)
