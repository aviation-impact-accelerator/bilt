"""Find additive invariants by exact elimination, retaining equation provenance."""

from fractions import Fraction

import pytest

from bilt.symbolic._algebra._invariants import discover_boundary_invariants
from bilt.symbolic._expressions import Expr


def test_efficiency_discovery_is_exact() -> None:
    power_in = Expr.symbol("power_in")
    power_out = Expr.symbol("power_out")
    heat = Expr.symbol("heat")
    efficiency = Expr.symbol("efficiency")
    results = discover_boundary_invariants(
        (
            ("electric", power_out == efficiency * power_in),
            ("heat", heat == (1 - efficiency) * power_in),
        ),
        frozenset({"power_in", "power_out", "heat"}),
    )
    assert len(results) == 1
    assert results[0].residual.coefficients == {
        ("power_in",): Fraction(1),
        ("power_out",): Fraction(-1),
        ("heat",): Fraction(-1),
    } or results[0].residual.coefficients == {
        ("power_in",): Fraction(-1),
        ("power_out",): Fraction(1),
        ("heat",): Fraction(1),
    }
    assert set(results[0].source_coefficients) == {"electric", "heat"}


def test_two_stage_balance_eliminates_internal_terms() -> None:
    power_in = Expr.symbol("power_in")
    intermediate = Expr.symbol("intermediate")
    power_out = Expr.symbol("power_out")
    loss1 = Expr.symbol("loss1")
    loss2 = Expr.symbol("loss2")
    e1 = Expr.symbol("e1")
    e2 = Expr.symbol("e2")
    equations = (
        ("stage1_power", intermediate == e1 * power_in),
        ("stage1_loss", loss1 == (1 - e1) * power_in),
        ("stage2_power", power_out == e2 * intermediate),
        ("stage2_loss", loss2 == (1 - e2) * intermediate),
    )
    results = discover_boundary_invariants(equations, frozenset({"power_in", "power_out", "loss1", "loss2"}))
    assert len(results) == 1
    coefficients = results[0].residual.coefficients
    assert coefficients[("power_in",)] == -coefficients[("power_out",)]
    assert coefficients[("loss1",)] == coefficients[("power_out",)]
    assert coefficients[("loss2",)] == coefficients[("power_out",)]
    assert set(results[0].source_coefficients) == {name for name, _ in equations}


def test_one_efficiency_equation_is_not_a_conservation_law() -> None:
    input_power = Expr.symbol("input_power")
    output_power = Expr.symbol("output_power")
    efficiency = Expr.symbol("efficiency")
    assert discover_boundary_invariants(
        (("efficiency", output_power == efficiency * input_power),),
        frozenset({"input_power", "output_power"}),
    ) == ()


def test_rejects_non_equality_and_duplicate_labels() -> None:
    x = Expr.symbol("x")
    with pytest.raises(ValueError, match="equality"):
        discover_boundary_invariants((("bad", x < 1),), frozenset({"x"}))
    with pytest.raises(ValueError, match="unique"):
        discover_boundary_invariants((("same", x == 1), ("same", x == 1)), frozenset({"x"}))
