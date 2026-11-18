"""Definition checks should reject ambiguity without modifying valid models."""

import dataclasses
import types
import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.errors import BiltDuplicateComponentError, BiltNameError, BiltTransformSignatureError
from bilt.units import Quantity
from bilt.units.dimensions import LENGTH

Length = typing.Annotated[Quantity, LENGTH]


@system
class Model(metaclass=Interface):
    x: Length = field()
    y: Length = field()


@transform
def y(x: Length) -> Length:
    return x


def test_distinct_components_with_the_same_name_are_rejected() -> None:
    first = y
    registered = set(Model.__bilt_system_registry__)

    @transform
    def y(x: Length) -> Length:
        return 2 * x

    with pytest.raises(BiltDuplicateComponentError, match="distinct components named 'y'"):
        Implementation(Model, [first, y])

    assert Model.__bilt_system_registry__ == registered


def test_repeated_reference_to_the_same_component_is_deduplicated() -> None:
    implementation = Implementation(Model, [y, y])

    assert tuple(implementation._components) == ("y",)  # noqa: SLF001


@pytest.mark.parametrize("signature", ("position_only", "var_args", "var_keywords"))
def test_transform_rejects_unsupported_signature_forms(signature: str) -> None:
    def position_only(x: float, /) -> float:
        return x

    def var_args(*args: float) -> float:
        return sum(args)

    def var_keywords(**kwargs: float) -> float:
        return sum(kwargs.values())

    function = {"position_only": position_only, "var_args": var_args, "var_keywords": var_keywords}[signature]
    with pytest.raises(BiltTransformSignatureError, match="named parameters only"):
        transform(function)


def test_transform_rejects_reserved_parameter_separator() -> None:
    def converted(a__b: float) -> float:
        return a__b

    with pytest.raises(BiltNameError):
        transform(converted)


def test_plain_transform_can_still_omit_annotations() -> None:
    @transform
    def twice(x):  # noqa: ANN001, ANN202
        return 2 * x

    assert twice.__bilt_parameters__ == ("x",)
    assert twice.__bilt_port_declarations__ == {}


def test_port_declarations_are_frozen() -> None:
    declaration = Model.__bilt_port_declarations__["x"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(declaration, "dimensionality", LENGTH)  # noqa: B010


def test_model_declaration_maps_are_read_only() -> None:
    @equation
    def same(x: Length, y: Length) -> bool:
        return x == y

    implementation = Implementation(Model, [same, y])
    for mapping in (
        Model.__bilt_port_declarations__,
        same.__bilt_port_declarations__,
        y.__bilt_port_declarations__,
        implementation._port_declarations,  # noqa: SLF001
    ):
        assert isinstance(mapping, types.MappingProxyType)
        with pytest.raises(TypeError):
            typing.cast("dict[str, object]", mapping)["new"] = mapping["x"]
