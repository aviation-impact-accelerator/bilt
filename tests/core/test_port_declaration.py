import typing

import pytest

from bilt import Interface, equation, field, system, transform
from bilt.core._port_declaration import PortDeclaration
from bilt.errors import BiltPortDeclarationError
from bilt.units import Quantity, kilogram
from bilt.units.dimensions import LENGTH, MASS

if typing.TYPE_CHECKING:
    import collections.abc

Mass = typing.Annotated[Quantity, MASS]
Kilograms = typing.Annotated[Quantity, kilogram]
MassAsFloat = typing.Annotated[float, MASS]
MassAndLength = typing.Annotated[Quantity, MASS, LENGTH]


def test_system_records_a_declaration_per_field() -> None:
    @system
    class Rig(metaclass=Interface):
        plain: float = field()
        bearing: Quantity = field()
        mass: Mass = field()

    assert Rig.__bilt_port_declarations__ == {
        "plain": PortDeclaration(),
        "bearing": PortDeclaration(carries_quantity=True),
        "mass": PortDeclaration(carries_quantity=True, dimensionality=MASS),
    }


def test_equation_records_a_declaration_per_parameter() -> None:
    @equation
    def balance(m: Mass, k: float) -> bool:
        return m == m * k

    assert balance.__bilt_port_declarations__ == {
        "m": PortDeclaration(carries_quantity=True, dimensionality=MASS),
        "k": PortDeclaration(),
    }


def test_equation_records_the_child_declaration_of_each_referenced_sub_system_port() -> None:
    @system
    class Child(metaclass=Interface):
        v: Mass = field()
        unused: float = field()

    @equation
    def link(child: Child, x: Mass) -> bool:
        return child.v == x

    assert link.__bilt_port_declarations__ == {
        "child.v": PortDeclaration(carries_quantity=True, dimensionality=MASS),
        "x": PortDeclaration(carries_quantity=True, dimensionality=MASS),
    }


def test_transform_records_its_return_annotation_under_its_own_name() -> None:
    @transform
    def doubled(m: Mass, k: float) -> Quantity:
        return m * k

    assert doubled.__bilt_port_declarations__ == {
        "m": PortDeclaration(carries_quantity=True, dimensionality=MASS),
        "k": PortDeclaration(),
        "doubled": PortDeclaration(carries_quantity=True),
    }


def test_transform_without_a_return_annotation_has_an_unconstrained_output() -> None:
    @transform
    def doubled(m: Mass):  # noqa: ANN202
        return m * 2.0

    assert doubled.__bilt_port_declarations__["doubled"] == PortDeclaration()


def test_quoted_annotations_are_resolved() -> None:
    @system
    class Rig(metaclass=Interface):
        mass: "Mass" = field()  # noqa: UP037
        limit: "typing.ClassVar[float]" = 1.0  # noqa: UP037

    @equation
    def balance(m: "Mass", k: float) -> bool:  # noqa: UP037
        return m == m * k

    @transform
    def doubled(m: Mass) -> "Mass":  # noqa: UP037
        return m * 2.0

    mass = PortDeclaration(carries_quantity=True, dimensionality=MASS)
    assert Rig.__bilt_port_declarations__ == {"mass": mass}
    assert balance.__bilt_port_declarations__["m"] == mass
    assert doubled.__bilt_port_declarations__["doubled"] == mass


def test_foreign_metadata_is_ignored() -> None:
    @system
    class Rig(metaclass=Interface):
        noted_mass: typing.Annotated[Quantity, "note", MASS] = field()
        noted_quantity: typing.Annotated[Quantity, "note"] = field()
        noted_float: typing.Annotated[float, "note"] = field()

    assert Rig.__bilt_port_declarations__ == {
        "noted_mass": PortDeclaration(carries_quantity=True, dimensionality=MASS),
        "noted_quantity": PortDeclaration(carries_quantity=True),
        "noted_float": PortDeclaration(),
    }


def _system_field(annotation: object) -> None:
    namespace = {"__annotations__": {"m": annotation}, "m": field()}
    system(Interface("Rig", (), namespace))


def _equation_parameter(annotation: object) -> None:
    def rig(m, k: float) -> bool:  # noqa: ANN001
        return m == k

    rig.__annotations__["m"] = annotation
    equation(rig)


def _transform_parameter(annotation: object) -> None:
    def rig(m, k: float) -> float:  # noqa: ANN001
        return m * k

    rig.__annotations__["m"] = annotation
    transform(rig)


def _transform_return(annotation: object) -> None:
    def rig(k: float):  # noqa: ANN202
        return k

    rig.__annotations__["return"] = annotation
    transform(rig)


SITES = (
    pytest.param(_system_field, "Rig", "m", id="system field"),
    pytest.param(_equation_parameter, "rig", "m", id="equation parameter"),
    pytest.param(_transform_parameter, "rig", "m", id="transform parameter"),
    pytest.param(_transform_return, "rig", "rig", id="transform return"),
)


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
def test_dimensionality_on_a_non_quantity_base_raises(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}'") as raised:
        declare(MassAsFloat)

    message = str(raised.value)
    assert "mass" in message
    assert "float" in message


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
def test_several_dimensionalities_raise(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}'") as raised:
        declare(MassAndLength)

    message = str(raised.value)
    assert "mass" in message
    assert "length" in message


@pytest.mark.parametrize(("declare", "owner", "port"), SITES)
def test_unit_raises(
    declare: collections.abc.Callable[[object], None],
    owner: str,
    port: str,
) -> None:
    with pytest.raises(BiltPortDeclarationError, match=f"Port '{port}' of '{owner}'") as raised:
        declare(Kilograms)

    message = str(raised.value)
    assert "kilogram" in message
    assert "mass" in message
