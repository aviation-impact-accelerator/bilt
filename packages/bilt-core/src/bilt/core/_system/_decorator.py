__all__ = ["system"]

import dataclasses
import enum
import typing

from ._dispatch import _dispatch_evaluation
from ._sentinel import MISSING

if typing.TYPE_CHECKING:
    import types

    from ._interface import Interface


@dataclasses.dataclass(frozen=True, slots=True)
class Attribute[T]:
    name: str
    annotation: type[T]


def system[T: Interface](cls: T) -> T:
    """Decorate a class as the interface of a system."""
    _create_bilt_system_class_attribute(cls)
    _create_initialize_method(cls)
    _create_properties(cls)
    _create_eq_method(cls)
    _create_repr_method(cls)
    _disable_hash(cls)
    return cls


def _create_bilt_system_class_attribute(cls: Interface) -> None:
    attributes = {
        name: Attribute(name, typing.cast("type[object]", annotation))
        for name, annotation in cls.__annotations__.items()
        if typing.get_origin(annotation) is not typing.ClassVar  # type: ignore[comparison-overlap]
    }
    setattr(cls, "__bilt_system_attributes__", attributes)  # noqa: B010


def _create_initialize_method(cls: Interface) -> None:
    parameters = cls.__bilt_system_attributes__
    args = ["self", "*", *(f"{parameter}=MISSING" for parameter in parameters)]
    body = [
        "self.__bilt_system_implementation__ = MISSING",
        f"self.__bilt_system_instance_variables__ = {{{', '.join(f'"{parameter}": {parameter}' for parameter in parameters)}}}",
    ]
    _create_method(
        cls,
        name="__init__",
        method=Method.METHOD,
        args=args,
        body=body,
        global_namespace=globals(),
        local_namespace={"MISSING": MISSING},
    )


def _create_properties(cls: Interface) -> None:
    parameters = cls.__bilt_system_attributes__
    local_namespace = {
        cls.__name__: cls,
        "_dispatch_evaluation": _dispatch_evaluation,
    }
    for parameter in parameters:
        body = [f"value = _dispatch_evaluation(self, self.__bilt_system_attributes__['{parameter}'])", "return value"]
        _create_method(
            cls,
            name=parameter,
            method=Method.PROPERTY,
            args=["self"],
            body=body,
            global_namespace=globals(),
            local_namespace=local_namespace,
        )


def _create_eq_method(cls: Interface) -> None:
    body = [
        "if other.__class__ is not self.__class__:",
        "    return NotImplemented",
        "return (tuple(self.__bilt_system_instance_variables__[name] for name in self.__bilt_system_attributes__)"
        " == tuple(other.__bilt_system_instance_variables__[name] for name in other.__bilt_system_attributes__))",
    ]
    _create_method(
        cls,
        name="__eq__",
        method=Method.METHOD,
        args=["self", "other"],
        body=body,
        global_namespace=globals(),
        local_namespace={},
    )


def _create_repr_method(cls: Interface) -> None:
    body = [
        "known_variables = {",
        "    name: self.__bilt_system_instance_variables__[name]",
        "    for name in self.__bilt_system_attributes__",
        "    if not self.__bilt_system_instance_variables__[name] is MISSING",
        "}",
        "formatted_variables = ', '.join(f'{name}={value!r}' for name, value in known_variables.items())",
        "return f'{self.__class__.__name__}({formatted_variables})'",
    ]
    _create_method(
        cls,
        name="__repr__",
        method=Method.METHOD,
        args=["self"],
        body=body,
        global_namespace=globals(),
        local_namespace={"MISSING": MISSING},
    )


def _disable_hash(cls: Interface) -> None:
    setattr(cls, "__hash__", None)  # noqa: B010


@enum.unique
class Method(enum.Enum):
    METHOD = enum.auto()
    PROPERTY = enum.auto()


def _create_decorator(method: Method) -> str:
    match method:
        case Method.METHOD:
            return ""
        case Method.PROPERTY:
            return "@property\n"
        case _:
            typing.assert_never(method)


def _create_method(  # noqa: PLR0913
    cls: Interface,
    *,
    name: str,
    method: Method,
    args: list[str],
    body: list[str],
    global_namespace: dict[str, typing.Any],
    local_namespace: dict[str, typing.Any],
) -> None:
    decorator = _create_decorator(method)
    body = [f"    {line}" for line in body]
    src = f"{decorator}def {name}({', '.join(args)}):\n{'\n'.join(body)}"
    exec(src, global_namespace, local_namespace)  # noqa: S102
    match method:
        case Method.METHOD:
            attribute = typing.cast("types.FunctionType", local_namespace[name])
            attribute.__qualname__ = f"{cls.__qualname__}.{attribute.__name__}"
        case Method.PROPERTY:
            attribute = typing.cast("property", local_namespace[name])
            fget = typing.cast("types.FunctionType", attribute.fget)
            fget.__qualname__ = f"{cls.__qualname__}.{fget.__name__}"
        case _:
            typing.assert_never(method)
    setattr(cls, name, attribute)
