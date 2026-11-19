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
    fixed_inputs = {
        name: value for name, value in instance.__bilt_system_instance_inputs__.items() if value is not MISSING
    }
    cached_inputs = {
        name: value for name, value in instance.__bilt_system_instance_variables__.items() if value is not MISSING
    }
    backend = _backend_stack[-1] if _backend_stack else PYTHON_BACKEND
    outputs = implementation._evaluate_with_intermediates(  # noqa: SLF001
        inputs=fixed_inputs,
        output=attribute.name,
        backend=backend,
        cached={name: value for name, value in cached_inputs.items() if name not in fixed_inputs},
    )
    if attribute.name not in outputs:
        msg = f"Evaluating {attribute.name!r} did not produce the requested value."
        raise RuntimeError(msg)
    if collisions := outputs.keys() & cached_inputs.keys():
        msg = f"Evaluating {attribute.name!r} recomputed available ports: {sorted(collisions)}."
        raise RuntimeError(msg)
    instance.__bilt_system_instance_variables__.update(outputs)
    return typing.cast("T", outputs[attribute.name])



def _prefetch_outputs(instance: SystemInstance, outputs: tuple[str, ...]) -> None:
    """Evaluate multiple instance ports together, retaining their intermediates.

    This internal prototype leaves ordinary property access unchanged.
    """
    cls = typing.cast("Interface", instance.__class__)
    requested = tuple(dict.fromkeys(outputs))
    for target in requested:
        if target not in cls.__bilt_system_attributes__:
            raise BiltUnknownSystemAttributeError(target, instance)
    missing = tuple(
        target
        for target in requested
        if isinstance(instance.__bilt_system_instance_variables__.get(target, MISSING), _Missing)
    )
    if not missing:
        return
    if len(missing) == 1:
        _dispatch_evaluation(instance, cls.__bilt_system_attributes__[missing[0]])
        return

    implementation = instance.__bilt_system_implementation__
    if isinstance(implementation, _Missing):
        if not cls.__bilt_system_implementation_default__:
            raise BiltUnknownSystemAttributeError(missing[0], instance)
        implementation = cls.__bilt_system_implementation_default__[-1]
        instance.__bilt_system_implementation__ = implementation

    fixed_inputs = {
        name: value for name, value in instance.__bilt_system_instance_inputs__.items() if value is not MISSING
    }
    known = {
        name: value for name, value in instance.__bilt_system_instance_variables__.items() if value is not MISSING
    }
    backend = _backend_stack[-1] if _backend_stack else PYTHON_BACKEND
    computed = implementation._evaluate_many_with_intermediates(  # noqa: SLF001
        fixed_inputs,
        missing,
        backend,
        cached={name: value for name, value in known.items() if name not in fixed_inputs},
    )
    if missing_outputs := set(missing) - (computed.keys() | known.keys()):
        msg = f"Multi-output evaluation omitted: {sorted(missing_outputs)}."
        raise RuntimeError(msg)
    if collisions := computed.keys() & known.keys():
        msg = f"Multi-output evaluation recomputed cached ports: {sorted(collisions)}."
        raise RuntimeError(msg)
    instance.__bilt_system_instance_variables__.update(computed)
