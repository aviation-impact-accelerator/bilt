"""Candidate discovery from multiple Bilt operating points."""

from typing import Annotated

import pytest

from bilt import Implementation, Interface, Quantity, equation, field, system
from bilt.experimental._invariants import discover_linear_invariants
from bilt.units.dimensions import POWER
from bilt.units.units import watt

Power = Annotated[Quantity, POWER]


@system
class Converter(metaclass=Interface):
    power_in: Power = field()
    efficiency: float = field()
    power_out: Power = field()
    heat: Power = field()


@equation
def electric(power_in: Power, efficiency: float, power_out: Power) -> bool:
    return power_out == power_in * efficiency


@equation
def thermal(power_in: Power, efficiency: float, heat: Power) -> bool:
    return heat == power_in * (1 - efficiency)


def _scenarios() -> list[dict[str, object]]:
    return [
        {"power_in": power * watt, "efficiency": eta}
        for power, eta in (
            (100.0, 0.2), (120.0, 0.35), (80.0, 0.55),
            (200.0, 0.7), (130.0, 0.8), (75.0, 0.4),
        )
    ]


def test_discover_energy_balance_without_a_declared_contract() -> None:
    model = Implementation(Converter, [electric, thermal])
    candidates = discover_linear_invariants(model, _scenarios())
    assert len(candidates) == 1
    coefficients = candidates[0].coefficients
    assert abs(coefficients["power_in"]) == pytest.approx(1)
    assert coefficients["power_out"] == pytest.approx(-coefficients["power_in"])
    assert coefficients["heat"] == pytest.approx(-coefficients["power_in"])
    assert candidates[0].maximum_residual == pytest.approx(0, abs=1e-7)
    assert candidates[0].status == "candidate"


def test_no_conservation_claim_from_efficiency_alone() -> None:
    model = Implementation(Converter, [electric])
    assert discover_linear_invariants(model, _scenarios()) == ()


def test_need_diverse_operating_points() -> None:
    model = Implementation(Converter, [electric, thermal])
    with pytest.raises(ValueError, match="at least four"):
        discover_linear_invariants(model, _scenarios()[:2])
