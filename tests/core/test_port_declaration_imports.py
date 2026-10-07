import typing

import pytest

from bilt import Interface, Quantity, equation, field, system, transform
from bilt.errors import BiltUnresolvedAnnotationError
from bilt.units.dimensions import MASS

if typing.TYPE_CHECKING:
    from bilt.units import Quantity as TypeCheckingQuantity


def test_equation_reads_a_quantity_imported_beside_the_decorator() -> None:
    @equation
    def balance(m: typing.Annotated[Quantity, MASS], k: typing.Annotated[Quantity, MASS]) -> bool:
        return m == k

    assert balance.__bilt_port_declarations__ == {"m": MASS, "k": MASS}


def test_equation_rejects_an_annotation_imported_only_for_type_checking() -> None:
    def balance(m: typing.Annotated[TypeCheckingQuantity, MASS], k: float) -> bool:
        return m == m * k

    with pytest.raises(BiltUnresolvedAnnotationError, match="annotations of 'balance'") as raised:
        equation(balance)

    assert raised.value.name == "TypeCheckingQuantity"
    assert "TYPE_CHECKING" in str(raised.value)
    assert isinstance(raised.value.__cause__, NameError)


def test_transform_rejects_an_annotation_imported_only_for_type_checking() -> None:
    def doubled(m: typing.Annotated[TypeCheckingQuantity, MASS]) -> TypeCheckingQuantity:
        return m * 2.0

    with pytest.raises(BiltUnresolvedAnnotationError, match="annotations of 'doubled'") as raised:
        transform(doubled)

    assert raised.value.name == "TypeCheckingQuantity"
    assert "TYPE_CHECKING" in str(raised.value)
    assert isinstance(raised.value.__cause__, NameError)


def test_system_rejects_an_annotation_imported_only_for_type_checking() -> None:
    class Rig(metaclass=Interface):
        mass: typing.Annotated[TypeCheckingQuantity, MASS] = field()

    with pytest.raises(BiltUnresolvedAnnotationError, match="annotations of 'Rig'") as raised:
        system(Rig)

    assert raised.value.name == "TypeCheckingQuantity"
    assert "TYPE_CHECKING" in str(raised.value)
    assert isinstance(raised.value.__cause__, NameError)
