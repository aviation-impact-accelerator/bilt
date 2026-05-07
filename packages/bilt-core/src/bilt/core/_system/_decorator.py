__all__ = ["system"]

import dataclasses
import enum
import typing

from ._interface import MISSING, Attribute, Interface, _dispatch_evaluation

if typing.TYPE_CHECKING:
    import types


class DataclassesDataclass(typing.Protocol):
    __name__: str
    __dataclass_fields__: typing.ClassVar[dict[str, dataclasses.Field[typing.Any]]]
    __dataclass_params__: typing.ClassVar[typing.Any]
    __post_init__: typing.ClassVar[typing.Callable[..., None]]

    __bilt_system_attributes__: typing.ClassVar[dict[str, Attribute[object]]]

    def __init__(self, *args: object, **kwargs: object) -> None: ...


def system[T](cls: type[T]) -> type[T]:
    def create_system(cls: type[T]) -> type[T]:
        dataclass = typing.cast("DataclassesDataclass", dataclasses.dataclass(cls))
        _create_bilt_system_class_attribute(dataclass)
        _create_initialize_method(dataclass)
        _create_properties(dataclass)
        return typing.cast("type[T]", dataclass)

    interface = typing.cast("type[T]", Interface(cls.__name__, cls.__bases__, dict(cls.__dict__)))
    return create_system(interface)


def _create_bilt_system_class_attribute(cls: DataclassesDataclass) -> None:
    annotations = cls.__dataclass_fields__
    cls.__bilt_system_attributes__ = {  # type: ignore[misc]
        annotation.name: Attribute(annotation.name, typing.cast("type[object]", annotation.type))
        for annotation in annotations.values()
    }


def _create_initialize_method(cls: DataclassesDataclass) -> None:
    annotations = cls.__dataclass_fields__
    args = ["self", *(f"{parameter}=MISSING" for parameter in annotations)]
    body = [
        "self.__bilt_system_implementation__ = MISSING",
        f"self.__bilt_system_instance_variables__ = {{{', '.join(f'"{parameter}": {parameter}' for parameter in annotations)}}}",
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


def _create_properties(cls: DataclassesDataclass) -> None:
    annotations = cls.__dataclass_fields__
    local_namespace = {
        cls.__name__: cls,
        "_dispatch_evaluation": _dispatch_evaluation,
    }
    for parameter in annotations:
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
    body: list[str],
    global_namespace: dict[str, typing.Any],
    local_namespace: dict[str, typing.Any],
) -> None:
    decorator = _create_decorator(method)
    body = [f"    {line}" for line in body]
    src = f"{decorator}def {name}({', '.join(args)}):\n{'\n'.join(body)}"
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
