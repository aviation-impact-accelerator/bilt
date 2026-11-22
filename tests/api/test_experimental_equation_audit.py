"""Experiments: checking equations without adding them to the solver."""

import pytest

from bilt import Implementation, Interface, equation, field, system
from bilt.experimental import audit_equations


@system
class Converter(metaclass=Interface):
    power_in: float = field()
    efficiency: float = field()
    power_out: float = field()
    heat: float = field()


@equation
def efficiency(power_in: float, efficiency: float, power_out: float) -> bool:
    return power_out == efficiency * power_in


@equation
def losses(power_in: float, efficiency: float, heat: float) -> bool:
    return heat == (1 - efficiency) * power_in


MODEL = Implementation(Converter, [efficiency, losses])


def test_expands_plan_to_calculate_missing_heat() -> None:
    report = audit_equations(MODEL, {"power_in": 100.0, "efficiency": 0.4}, "power_out")
    assert report.output == pytest.approx(40.0)
    assert report.values["heat"] == pytest.approx(60.0)
    assert {item.name: item.status for item in report.checks} == {
        "efficiency": "satisfied",
        "losses": "satisfied",
    }
    assert all(not item.independent for item in report.checks)


def test_independent_observation_detects_mismatch() -> None:
    report = audit_equations(
        MODEL,
        {"power_in": 100.0, "efficiency": 0.4},
        "power_out",
        observations={"heat": 55.0},
    )
    assert report.values["heat"] == 55.0
    by_name = {item.name: item for item in report.checks}
    assert by_name["losses"].status == "violated"
    assert by_name["losses"].residual == pytest.approx(-5.0)


def test_fully_observed_equation_is_independent() -> None:
    report = audit_equations(
        MODEL,
        {"power_in": 100.0, "efficiency": 0.4, "power_out": 40.0},
        "power_out",
        observations={"heat": 55.0},
    )
    checks = {item.name: item for item in report.checks}
    assert checks["efficiency"].independent
    assert checks["losses"].independent
    assert checks["losses"].status == "violated"


def test_unavailable_terms_are_reported() -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    @equation
    def link(a: float, b: float) -> bool:
        return b == 2 * a

    @equation
    def unattainable(c: float, unknown: float) -> bool:
        return c == unknown + 1

    model = Implementation(Demo, [link, unattainable])
    report = audit_equations(model, {"a": 3.0}, "b")
    assert report.output == pytest.approx(6)
    assert next(item for item in report.checks if item.name == "unattainable").status == "unavailable"
