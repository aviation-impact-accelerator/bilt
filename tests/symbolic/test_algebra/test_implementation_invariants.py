"""Discover power balances directly from Bilt model definitions."""

from typing import Annotated

from bilt import Implementation, Interface, Quantity, equation, field, system
from bilt.symbolic._algebra._implementation_invariants import discover_implementation_invariants
from bilt.units.dimensions import DIMENSIONLESS, MASS, POWER

Power = Annotated[Quantity, POWER]
Efficiency = Annotated[Quantity, DIMENSIONLESS]


@system
class Converter(metaclass=Interface):
    power_in: Power = field()
    efficiency: Efficiency = field()
    power_out: Power = field()
    heat: Power = field()


@equation
def electric(power_in: Power, efficiency: Efficiency, power_out: Power) -> bool:
    return power_out == efficiency * power_in


@equation
def loss(power_in: Power, efficiency: Efficiency, heat: Power) -> bool:
    return heat == (1 - efficiency) * power_in


def test_automatically_discovers_power_boundary() -> None:
    implementation = Implementation(Converter, [electric, loss])
    identities = discover_implementation_invariants(implementation, dimension=POWER)
    assert len(identities) == 1
    result = identities[0]
    coefficients = result.residual.coefficients
    assert set(coefficients) == {("power_in",), ("power_out",), ("heat",)}
    assert coefficients[("power_in",)] == -coefficients[("power_out",)]
    assert coefficients[("heat",)] == coefficients[("power_out",)]
    assert set(result.source_coefficients) == {"electric", "loss"}


def test_unrelated_dimension_has_no_identity() -> None:
    implementation = Implementation(Converter, [electric, loss])
    assert discover_implementation_invariants(implementation, dimension=MASS) == ()


def test_one_equation_does_not_imply_a_balance() -> None:
    implementation = Implementation(Converter, [electric])
    assert discover_implementation_invariants(implementation, dimension=POWER) == ()
