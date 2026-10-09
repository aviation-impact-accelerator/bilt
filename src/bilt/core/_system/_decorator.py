__all__ = ["system"]

import dataclasses
import enum
import inspect
import typing

from bilt.core._errors import BiltInputDimensionalityError, BiltUnresolvedAnnotationError
from bilt.core._interface import Interface
from bilt.core._port_declaration import port_declarations
from bilt.core._system._errors import BiltSystemDefinitionError
from bilt.core._tensors._tensors import Tensor
from bilt.units import Quantity
from bilt.units._category import UnitCategory

from ._dispatch import _dispatch_evaluation
from ._sentinel import MISSING

if typing.TYPE_CHECKING:
    import collections.abc
    import types


@dataclasses.dataclass(frozen=True, slots=True)
class Attribute[T]:
    name: str
    annotation: type[T]


def system[T: Interface](cls: T) -> T:
    """Decorate a class as the interface of a system."""
    _validate_cls_is_interface(cls)

    _create_bilt_system_class_attribute(cls)
    _create_initialize_method(cls)
    _create_properties(cls)
    _create_eq_method(cls)
    _create_repr_method(cls)
    _disable_hash(cls)

    return cls


def _validate_cls_is_interface(cls: type) -> None:
    if not isinstance(cls, Interface):
        raise BiltSystemDefinitionError(cls)


def _create_bilt_system_class_attribute(cls: Interface) -> None:
    try:
        annotations = inspect.get_annotations(cls, eval_str=True)
    except NameError as error:
        raise BiltUnresolvedAnnotationError(cls.__name__, error) from error
    attributes = {
        name: Attribute(name, typing.cast("type[object]", annotation))
        for name, annotation in annotations.items()
        if typing.get_origin(annotation) is not typing.ClassVar  # type: ignore[comparison-overlap]
    }
    setattr(cls, "__bilt_system_attributes__", attributes)  # noqa: B010
    setattr(cls, "__bilt_parameters__", tuple(attributes.keys()))  # noqa: B010
    setattr(  # noqa: B010
        cls,
        "__bilt_port_declarations__",
        port_declarations(cls.__name__, {name: attribute.annotation for name, attribute in attributes.items()}),
    )


def _create_initialize_method(cls: Interface) -> None:
    parameters = cls.__bilt_system_attributes__
    args = ["self", "*", *(f"{parameter}=MISSING" for parameter in parameters)]
    body = [
        "self.__bilt_system_implementation__ = MISSING",
        f"self.__bilt_system_instance_variables__ = {{{', '.join(f'"{parameter}": {parameter}' for parameter in parameters)}}}",
        "_check_inputs(type(self), self.__bilt_system_instance_variables__)",
    ]
    _create_method(
        cls,
        name="__init__",
        method=Method.METHOD,
        args=args,
        body=body,
        global_namespace=globals(),
        local_namespace={"MISSING": MISSING, "_check_inputs": _check_inputs},
    )


def _check_inputs(cls: Interface, values: collections.abc.Mapping[str, object]) -> None:
    """Ensure the values an instance is given satisfy the quantities their ports declare.

    A quantity and a tensor answer to the same rule: a declared port wants a unit of its dimensionality, absolute
    if the port is and relative otherwise, and a plain port wants no unit at all. A port that declares itself over
    a domain wants that unit on a tensor, so a reduction in an equation never meets a scalar.
    """
    for port, value in values.items():
        if value is MISSING:
            continue

        unit = value.unit if isinstance(value, Quantity | Tensor) else None
        expected = cls.__bilt_port_declarations__.get(port)
        if expected is None:
            if unit is not None:
                raise BiltInputDimensionalityError(cls.__bilt_name__, port, expected, value)
        elif (
            unit is None
            or unit.dimension != expected.dimensionality
            or not (unit.category is UnitCategory.ABSOLUTE if expected.absolute else unit.is_relative)
            or (expected.over_a_domain and not isinstance(value, Tensor))
        ):
            raise BiltInputDimensionalityError(cls.__bilt_name__, port, expected, value)


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
        "return (tuple(self.__bilt_system_instance_variables__[name] for name in self.__bilt_system_attributes__)",
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
