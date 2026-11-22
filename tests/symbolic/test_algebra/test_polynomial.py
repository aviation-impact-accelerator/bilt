"""Exact bounded polynomial normalisation on Bilt's expression trees."""

from fractions import Fraction

import pytest

from bilt.symbolic._algebra._polynomial import PolynomialAnalysisError, as_polynomial
from bilt.symbolic._expressions import Expr
from bilt.symbolic._tree import float_tree, symbol_tree


def test_normalises_efficiency_and_heat_residuals() -> None:
    power_in = Expr.symbol("power_in")
    power_out = Expr.symbol("power_out")
    heat = Expr.symbol("heat")
    efficiency = Expr.symbol("efficiency")
    residual = (power_out - efficiency * power_in) + (heat - (1 - efficiency) * power_in)

    assert as_polynomial(residual.tree).coefficients == {
        ("power_out",): Fraction(1),
        ("heat",): Fraction(1),
        ("power_in",): Fraction(-1),
    }


def test_multiplicative_factors_are_canonical() -> None:
    x, y = Expr.symbol("x"), Expr.symbol("y")
    assert as_polynomial((x * y - y * x).tree).is_zero
    assert as_polynomial((x * x * y).tree).coefficients == {("x", "x", "y"): Fraction(1)}


def test_exact_decimal_coefficients_and_constant_division() -> None:
    x = Expr.symbol("x")
    residual = x * 0.1 + x * 0.2 - x * 0.3
    assert as_polynomial(residual.tree).is_zero
    assert as_polynomial((x / 2).tree).coefficients == {("x",): Fraction(1, 2)}


def test_expands_integer_powers() -> None:
    x = Expr.symbol("x")
    poly = as_polynomial(((x + 1) ** 3).tree)
    assert poly.coefficients == {
        ("x", "x", "x"): Fraction(1),
        ("x", "x"): Fraction(3),
        ("x",): Fraction(3),
        (): Fraction(1),
    }


def test_rejects_rational_functions_and_unsupported_relations() -> None:
    x = Expr.symbol("x")
    with pytest.raises(PolynomialAnalysisError, match="division by nonzero scalar"):
        as_polynomial((1 / x).tree)
    with pytest.raises(PolynomialAnalysisError, match="polynomial operator"):
        as_polynomial(x == 1)
    with pytest.raises(PolynomialAnalysisError, match="integer polynomial powers"):
        as_polynomial((x**0.5).tree)


def test_rejects_limits_and_nonfinite_literals() -> None:
    x, y = Expr.symbol("x"), Expr.symbol("y")
    with pytest.raises(PolynomialAnalysisError, match="degree exceeds"):
        as_polynomial((x * y).tree, max_degree=1)
    with pytest.raises(PolynomialAnalysisError, match="more than 1 terms"):
        as_polynomial((x + y).tree, max_terms=1)
    with pytest.raises(PolynomialAnalysisError, match="Non-finite"):
        as_polynomial(float_tree(float("inf")))


def test_zero_polynomial_is_immutable() -> None:
    x = symbol_tree("x")
    poly = as_polynomial(x)
    with pytest.raises(TypeError):
        poly.coefficients[("new",)] = Fraction(2)  # type: ignore[index]
