__all__ = []

import typing

from ._error import BiltUnknownSystemAttributeError
from ._sentinel import MISSING, _Missing

if typing.TYPE_CHECKING:
    from ._decorator import Attribute
    from ._instance import SystemInstance
    from ._interface import Interface


def _dispatch_evaluation[T](instance: SystemInstance, attribute: Attribute[T]) -> T:
    current = typing.cast("T | _Missing", instance.__bilt_system_instance_variables__.get(attribute.name, MISSING))
    if not isinstance(current, _Missing):
        return current
    implementation = instance.__bilt_system_implementation__
    cls = typing.cast("Interface", instance.__class__)
    if isinstance(implementation, _Missing):
        if len(cls.__bilt_system_implementation_default__) == 0:
            raise BiltUnknownSystemAttributeError(attribute.name, instance)
        implementation = cls.__bilt_system_implementation_default__[-1]
        instance.__bilt_system_implementation__ = implementation
    inputs = {
        name: value for name, value in instance.__bilt_system_instance_variables__.items() if value is not MISSING
    }
    output = attribute.name
    result = typing.cast("T", implementation.evaluate(inputs=inputs, output=output))
    instance.__bilt_system_instance_variables__[attribute.name] = result
    return result
