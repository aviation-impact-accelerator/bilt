from __future__ import annotations

__all__ = []

import typing

from bilt._backend._base_backend import _backend_stack
from bilt._backend._python import PythonBackend

from ._analysis import BiltSystemEvaluationError
from ._error import BiltUnknownSystemAttributeError
from ._interface import Interface
from ._sentinel import MISSING, _Missing

if typing.TYPE_CHECKING:
    from ._decorator import Attribute
    from ._implementation import Implementation
    from ._instance import SystemInstance


def _dispatch_evaluation[T](
    instance: SystemInstance,
    attribute: Attribute[T],
) -> T:
    current = typing.cast(
        "T | _Missing",
        instance.__bilt_system_instance_variables__.get(
            attribute.name,
            MISSING,
        ),
    )

    if not isinstance(current, _Missing):
        _bind_child(
            parent=instance,
            name=attribute.name,
            value=current,
        )
        return current

    parent = getattr(
        instance,
        "__bilt_system_parent__",
        MISSING,
    )

    if not isinstance(parent, _Missing):
        result = _evaluate_from_root(
            instance=instance,
            output=attribute.name,
        )
    else:
        implementation = _resolve_implementation(
            instance,
            attribute.name,
        )
        inputs = {
            name: value for name, value in instance.__bilt_system_instance_variables__.items() if value is not MISSING
        }
        backend = _current_backend()
        result = implementation.evaluate(
            inputs=inputs,
            output=attribute.name,
            backend=backend,
        )

    instance.__bilt_system_instance_variables__[attribute.name] = result
    return typing.cast("T", result)


def _bind_known_children(instance: SystemInstance) -> None:
    cls = typing.cast("Interface", instance.__class__)

    for name, attribute in cls.__bilt_system_attributes__.items():
        if not isinstance(attribute.annotation, Interface):
            continue

        value = instance.__bilt_system_instance_variables__.get(
            name,
            MISSING,
        )

        if value is MISSING:
            continue

        _bind_child(
            parent=instance,
            name=name,
            value=value,
        )


def _bind_child(
    *,
    parent: SystemInstance,
    name: str,
    value: object,
) -> None:
    if not isinstance(value.__class__, Interface):
        return

    existing_parent = getattr(
        value,
        "__bilt_system_parent__",
        MISSING,
    )
    existing_name = getattr(
        value,
        "__bilt_system_parent_attribute__",
        MISSING,
    )

    if isinstance(existing_parent, _Missing):
        value.__bilt_system_parent__ = parent
        value.__bilt_system_parent_attribute__ = name
        return

    if existing_parent is parent and existing_name == name:
        return

    message = f"Child system instance {value!r} is already bound to a different parent field."
    raise BiltSystemEvaluationError(message)


def _evaluate_from_root(
    *,
    instance: SystemInstance,
    output: str,
) -> object:
    root = instance
    path = [output]

    while True:
        parent = getattr(
            root,
            "__bilt_system_parent__",
            MISSING,
        )
        parent_attribute = getattr(
            root,
            "__bilt_system_parent_attribute__",
            MISSING,
        )

        if isinstance(parent, _Missing):
            break

        if isinstance(parent_attribute, _Missing):
            message = "Child system has an incomplete parent binding."
            raise BiltSystemEvaluationError(message)

        path.append(typing.cast("str", parent_attribute))
        root = typing.cast("SystemInstance", parent)

    implementation = _resolve_implementation(
        root,
        ".".join(reversed(path)),
    )
    inputs = {name: value for name, value in root.__bilt_system_instance_variables__.items() if value is not MISSING}
    backend = _current_backend()

    return implementation.evaluate(
        inputs=inputs,
        output=".".join(reversed(path)),
        backend=backend,
    )


def _resolve_implementation(
    instance: SystemInstance,
    output: str,
) -> Implementation:
    implementation = instance.__bilt_system_implementation__
    cls = typing.cast("Interface", instance.__class__)

    if isinstance(implementation, _Missing):
        if len(cls.__bilt_system_implementation_default__) == 0:
            raise BiltUnknownSystemAttributeError(
                output,
                instance,
            )

        implementation = cls.__bilt_system_implementation_default__[-1]
        instance.__bilt_system_implementation__ = implementation

    return implementation


def _current_backend():
    return _backend_stack[-1] if _backend_stack else PythonBackend()
