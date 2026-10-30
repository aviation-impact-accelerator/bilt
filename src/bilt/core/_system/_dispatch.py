__all__ = []

import typing

from bilt._backend._base_backend import _backend_stack
from bilt._backend._python import PYTHON_BACKEND
from bilt.core._system._errors import BiltUnknownSystemAttributeError

from ._sentinel import MISSING, _Missing

if typing.TYPE_CHECKING:
    from bilt.core._interface import Interface

    from ._decorator import Attribute
    from ._instance import SystemInstance


def _dispatch_evaluation[T](instance: SystemInstance, attribute: Attribute[T]) -> T:
    supplied = instance.__bilt_system_instance_inputs__.get(attribute.name, MISSING)
    if not isinstance(supplied, _Missing):
        return typing.cast("T", supplied)
    backend = _backend_stack[-1] if _backend_stack else PYTHON_BACKEND
    cls = typing.cast("Interface", instance.__class__)
    bound = instance.__bilt_system_implementation__
    active = cls.__bilt_system_implementation_default__
    implementation = active[-1] if active else bound
    if isinstance(implementation, _Missing):
        raise BiltUnknownSystemAttributeError(attribute.name, instance)
    values = instance.__bilt_system_instance_evaluation_cache__.setdefault((implementation, backend), {})
    current = typing.cast("T | _Missing", values.get(attribute.name, MISSING))
    if not isinstance(current, _Missing):
        return current
    inputs = {name: value for name, value in instance.__bilt_system_instance_inputs__.items() if value is not MISSING}
    output = attribute.name
    result = typing.cast("T", implementation.evaluate(inputs=inputs, output=output, backend=backend))
    values[attribute.name] = result
    if isinstance(bound, _Missing):
        instance.__bilt_system_implementation__ = implementation
    if implementation.options is None:
        # Preserve existing repr/equality behaviour for ordinary implementations.
        # Dispatch itself always reads the cache of the selected owner above.
        instance.__bilt_system_instance_variables__[attribute.name] = result
    return result
