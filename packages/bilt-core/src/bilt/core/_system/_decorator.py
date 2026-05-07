__all__ = ["system"]

import dataclasses
import enum
import typing

from ._error import BiltSystemImplementationSelectionError, BiltUnknownSystemAttributeError
from ._interface import Interface

if typing.TYPE_CHECKING:
    import types


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()


class DataclassesDataclass(typing.Protocol):
    __name__: str
    __dataclass_fields__: typing.ClassVar[dict[str, dataclasses.Field[typing.Any]]]
    __dataclass_params__: typing.ClassVar[typing.Any]
    __post_init__: typing.ClassVar[typing.Callable[..., None]]

    def __init__(self, *args: object, **kwargs: object) -> None: ...


def system[T](cls: type[T]) -> type[T]:
    def create_system(cls: type[T]) -> type[T]:
        dataclass = typing.cast("DataclassesDataclass", dataclasses.dataclass(cls))
        _create_initialize_method(dataclass)
        _create_properties(dataclass)
        _create_with_implementation_method(dataclass)
        return typing.cast("type[T]", dataclass)

    interface = typing.cast("type[T]", Interface(cls.__name__, cls.__bases__, dict(cls.__dict__)))
    return create_system(interface)


def _create_initialize_method(cls: DataclassesDataclass) -> None:
    annotations = cls.__dataclass_fields__
    args = ["self", *(f"{parameter}=MISSING" for parameter in annotations)]
    body = "\n".join(f"    self._{parameter} = {parameter}" for parameter in annotations)
    body += "\n    self.__bilt_system_implementation__ = MISSING"
    _create_method(
        cls,
        name="__init__",
        method=Method.METHOD,
        args=args,
        body=body,
        global_namespace=globals(),
        local_namespace={"MISSING": MISSING},
    )


def _create_properties(cls: DataclassesDataclass) -> None:
    annotations = cls.__dataclass_fields__
    local_namespace = {
        cls.__name__: cls,
        "BiltUnknownSystemAttributeError": BiltUnknownSystemAttributeError,
        "MISSING": MISSING,
    }
    for parameter in annotations:
        body = (
            f"    if self._{parameter} is not MISSING:\n"
            f"        return self._{parameter}\n"
            f"    if self.__bilt_system_implementation__ is MISSING:\n"
            f"        if self.__class__.__bilt_system_implementation_default__ is None:\n"
            f"            raise BiltUnknownSystemAttributeError({parameter!r}, {cls.__name__!r})\n"
            f"        self.__bilt_system_implementation__ = self.__class__.__bilt_system_implementation_default__\n"
            f"    implementation = self.__bilt_system_implementation__\n"
            f"    inputs = {{parameter: getattr(self, '_' + parameter) for parameter in {list(annotations)} if getattr(self, '_' + parameter) is not MISSING}}\n"
            f"    output = '{parameter}'\n"
            f"    result = implementation.evaluate(inputs=inputs, output=output)\n"
            f"    self._{parameter} = result\n"
            f"    return result"
        )
        _create_method(
            cls,
            name=parameter,
            method=Method.PROPERTY,
            args=["self"],
            body=body,
            global_namespace=globals(),
            local_namespace=local_namespace,
        )


def _create_with_implementation_method(cls: DataclassesDataclass) -> None:
    args = ["self", "implementation"]
    body = (
        "    if self.__bilt_system_implementation__ is not MISSING:\n"
        "        message = (\n"
        "            f'Cannot use the provided implementation with the {self.__class__.__name__!r} instance as '\n"
        "            f'it is already using an implementation.'\n"
        "        )\n"
        "        raise BiltSystemImplementationSelectionError(message)\n"
        "    if implementation not in self.__bilt_system_registry__:\n"
        "        message = (\n"
        "            f'Cannot use the provided implementation with the {self.__class__.__name__!r} instance as this '\n"
        "            f'is from the {implementation.system.__name__!r} system.'\n"
        "        )\n"
        "        raise BiltSystemImplementationSelectionError(message)\n"
        "    self.__bilt_system_implementation__ = implementation"
    )
    _create_method(
        cls,
        name="with_implementation",
        method=Method.METHOD,
        args=args,
        body=body,
        global_namespace=globals(),
        local_namespace={
            cls.__name__: cls,
            "BiltSystemImplementationSelectionError": BiltSystemImplementationSelectionError,
            "MISSING": MISSING,
        },
    )


@enum.unique
class Method(enum.Enum):
    METHOD = enum.auto()
    PROPERTY = enum.auto()
    CLASS = enum.auto()
    STATIC = enum.auto()


def _create_decorator(method: Method) -> str:
    match method:
        case Method.METHOD:
            return ""
        case Method.PROPERTY:
            return "@property\n"
        case Method.CLASS:
            return "@classmethod\n"
        case Method.STATIC:
            return "@staticmethod\n"
        case _:
            typing.assert_never(method)


def _create_method(  # noqa: PLR0913
    cls: DataclassesDataclass,
    *,
    name: str,
    method: Method,
    args: list[str],
    body: str,
    global_namespace: dict[str, typing.Any],
    local_namespace: dict[str, typing.Any],
) -> None:
    decorator = _create_decorator(method)
    src = f"{decorator}def {name}({', '.join(args)}):\n{body}"
    exec(src, global_namespace, local_namespace)  # noqa: S102
    attribute = typing.cast("types.FunctionType", local_namespace[name])
    match method:
        case Method.METHOD | Method.CLASS | Method.STATIC:
            attribute.__qualname__ = f"{cls.__qualname__}.{attribute.__name__}"  # type: ignore[attr-defined]
        case Method.PROPERTY:
            attribute.fget.__qualname__ = f"{cls.__qualname__}.{attribute.fget.__name__}"  # type: ignore[attr-defined]
        case _:
            typing.assert_never(method)
    setattr(cls, name, attribute)
