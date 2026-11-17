"""Structured scientific outputs use ordinary Bilt systems, not a record-specific API."""

from typing import Annotated

import numpy as np
import pytest

from bilt import Implementation, Interface, Quantity, QuantityTensor, dim, equation, field, system, tensor
from bilt.errors import BiltInputDimensionalityError
from bilt.units import UNITLESS, kilogram, percent, year
from bilt.units.dimensions import DIMENSIONLESS

type Share = Annotated[Quantity, DIMENSIONLESS]
type SharePort = Annotated[Quantity | QuantityTensor, DIMENSIONLESS]


@system
class FuelMix(metaclass=Interface):
    hefa: SharePort = field()
    btl: SharePort = field()
    saf: SharePort = field()
    fossil: SharePort = field()


@equation
def saf_balance(saf: Share, hefa: Share, btl: Share) -> bool:
    return saf == hefa + btl


@equation
def fossil_balance(fossil: Share, saf: Share) -> bool:
    return fossil + saf == 1.0 * UNITLESS


mix_model = Implementation(FuelMix, [saf_balance, fossil_balance])


@system
class Portfolio(metaclass=Interface):
    hefa: SharePort = field()
    btl: SharePort = field()
    fossil: SharePort = field()


@equation
def hefa_connection(hefa: Share, mix: FuelMix) -> bool:
    return hefa == mix.hefa


@equation
def btl_connection(btl: Share, mix: FuelMix) -> bool:
    return btl == mix.btl


@equation
def fossil_connection(fossil: Share, mix: FuelMix) -> bool:
    return fossil == mix.fossil


portfolio_model = Implementation(
    Portfolio,
    [hefa_connection, btl_connection, fossil_connection],
    internal_implementations={"mix": mix_model},
)


def test_structured_value_fields_can_be_solved_without_any_multi_output_relationship() -> None:
    with mix_model:
        fuel = FuelMix(hefa=Quantity(20, percent), btl=Quantity(30, percent))
        assert fuel.saf.convert_to(percent).value == pytest.approx(50)
        assert fuel.fossil.convert_to(percent).value == pytest.approx(50)


def test_structured_fields_keep_named_tensor_domain_dependencies() -> None:
    scenario = dim("scenario")
    with mix_model:
        fuel = FuelMix(
            hefa=tensor([10, 20, 30], over=scenario, unit=percent),
            btl=Quantity(20, percent),
        )
        saf = fuel.saf
        fossil = fuel.fossil
    assert isinstance(saf, QuantityTensor)
    assert isinstance(fossil, QuantityTensor)
    assert tuple(str(d) for d in saf.domain.dims) == ("scenario",)
    assert tuple(str(d) for d in fossil.domain.dims) == ("scenario",)
    np.testing.assert_allclose(saf.convert_to(percent).values, [30, 40, 50])
    np.testing.assert_allclose(fossil.convert_to(percent).values, [70, 60, 50])


def test_parent_uses_child_fields_without_unwrapping_a_python_record() -> None:
    with portfolio_model:
        portfolio = Portfolio(hefa=Quantity(20, percent), btl=Quantity(30, percent))
        assert portfolio.fossil.convert_to(percent).value == pytest.approx(50)


def test_structured_system_fields_check_supplied_units() -> None:
    with pytest.raises(BiltInputDimensionalityError, match="hefa"):
        FuelMix(hefa=Quantity(2, kilogram), btl=Quantity(1, percent))

    with pytest.raises(BiltInputDimensionalityError, match="btl"):
        FuelMix(hefa=Quantity(1, percent), btl=Quantity(3, year))
