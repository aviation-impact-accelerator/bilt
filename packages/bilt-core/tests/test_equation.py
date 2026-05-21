import typing
from typing import Annotated, cast

import pytest

from bilt.core._equation import (
    Equation,
    Relation,
    Symbol,
    display_name_from_metadata,
    eq,
    equation,
    equation_symbols,
    ge,
    le,
    relation,
    relation_to_string,
    split_annotated,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable


def test_symbol_string_uses_name_when_no_display_name() -> None:
    symbol = Symbol("pressure")

    assert str(symbol) == "pressure"


def test_symbol_string_uses_display_name_when_available() -> None:
    symbol = Symbol("pressure", "p")

    assert str(symbol) == "p"


def test_relation_to_string() -> None:
    lhs = Symbol("p_in", "pᵢₙ")
    rhs = Symbol("p_out", "pₒᵤₜ")

    rel = relation("==", lhs, rhs)

    assert relation_to_string(rel) == "pᵢₙ == pₒᵤₜ"
    assert str(rel) == "pᵢₙ == pₒᵤₜ"


def test_eq_creates_equality_relation() -> None:
    lhs = Symbol("x")
    rhs = Symbol("y")

    rel = eq(lhs, rhs)

    assert isinstance(rel, Relation)
    assert rel.op == "=="
    assert rel.lhs == lhs
    assert rel.rhs == rhs


def test_le_creates_less_equal_relation() -> None:
    lhs = Symbol("x")
    rhs = Symbol("y")

    rel = le(lhs, rhs)

    assert rel.op == "<="
    assert rel.lhs == lhs
    assert rel.rhs == rhs


def test_ge_creates_greater_equal_relation() -> None:
    lhs = Symbol("x")
    rhs = Symbol("y")

    rel = ge(lhs, rhs)

    assert rel.op == ">="
    assert rel.lhs == lhs
    assert rel.rhs == rhs


def test_split_annotated_returns_base_annotation_and_metadata() -> None:
    annotation = Annotated[float, Symbol("pressure", "p")]

    base_annotation, metadata = split_annotated(annotation)

    assert base_annotation is float
    assert metadata == (Symbol("pressure", "p"),)


def test_split_annotated_returns_empty_metadata_for_plain_annotation() -> None:
    base_annotation, metadata = split_annotated(float)

    assert base_annotation is float
    assert metadata == ()


def test_display_name_from_metadata_prefers_display_name() -> None:
    metadata = (Symbol("pressure", "p"),)

    assert display_name_from_metadata(metadata) == "p"


def test_display_name_from_metadata_falls_back_to_name() -> None:
    metadata = (Symbol("pressure"),)

    assert display_name_from_metadata(metadata) == "pressure"


def test_display_name_from_metadata_returns_none_when_no_name_like_metadata() -> None:
    metadata = (object(),)

    assert display_name_from_metadata(metadata) is None


def test_equation_symbols_from_plain_annotations() -> None:
    def pressure_balance(p_in: float, p_out: float) -> Relation:
        return eq(p_in, p_out)

    symbols = equation_symbols(
        function_signature=__import__("inspect").signature(pressure_balance),
        type_hints={"p_in": float, "p_out": float},
    )

    assert symbols == {
        "p_in": Symbol("p_in"),
        "p_out": Symbol("p_out"),
    }


def test_equation_symbols_from_annotated_symbols() -> None:
    def pressure_balance(
        p_in: Annotated[float, Symbol("p_in", "pᵢₙ")],
        p_out: Annotated[float, Symbol("p_out", "pₒᵤₜ")],
    ) -> Relation:
        return eq(p_in, p_out)

    symbols = equation_symbols(
        function_signature=__import__("inspect").signature(pressure_balance),
        type_hints={
            "p_in": Annotated[float, Symbol("p_in", "pᵢₙ")],
            "p_out": Annotated[float, Symbol("p_out", "pₒᵤₜ")],
        },
    )

    assert symbols == {
        "p_in": Symbol("p_in", "pᵢₙ"),
        "p_out": Symbol("p_out", "pₒᵤₜ"),
    }


def test_equation_decorator_creates_equation_from_function() -> None:
    @equation
    def pressure_balance(
        p_in: Annotated[float, Symbol("p_in", "pᵢₙ")],
        p_out: Annotated[float, Symbol("p_out", "pₒᵤₜ")],
    ) -> Relation:
        return eq(p_in, p_out)

    assert isinstance(pressure_balance, Equation)
    assert pressure_balance.name == "pressure_balance"
    assert pressure_balance.symbols == {
        "p_in": Symbol("p_in", "pᵢₙ"),
        "p_out": Symbol("p_out", "pₒᵤₜ"),
    }
    assert pressure_balance.relation == Relation(
        "==",
        Symbol("p_in", "pᵢₙ"),
        Symbol("p_out", "pₒᵤₜ"),
    )
    assert str(pressure_balance) == "pressure_balance: pᵢₙ == pₒᵤₜ"


def test_equation_decorator_preserves_signature() -> None:
    @equation
    def pressure_balance(p_in: float, p_out: float) -> Relation:
        return eq(p_in, p_out)

    assert tuple(pressure_balance.signature.parameters) == ("p_in", "p_out")


def test_equation_decorator_uses_parameter_names_as_symbol_names_by_default() -> None:
    @equation
    def pressure_balance(p_in: float, p_out: float) -> Relation:
        return eq(p_in, p_out)

    assert pressure_balance.symbols == {
        "p_in": Symbol("p_in"),
        "p_out": Symbol("p_out"),
    }
    assert str(pressure_balance) == "pressure_balance: p_in == p_out"


def test_equation_decorator_supports_any_annotation() -> None:  # no
    @equation
    def balance(x: object, y: object) -> Relation:
        return eq(x, y)

    assert balance.symbols == {
        "x": Symbol("x"),
        "y": Symbol("y"),
    }
    assert str(balance) == "balance: x == y"


def test_equation_function_must_return_relation() -> None:
    def bad_equation(_x: float) -> object:
        return "not a relation"

    bad_equation_callable = cast("Callable[..., Relation]", bad_equation)

    with pytest.raises(TypeError, match="must return a Relation"):
        equation(bad_equation_callable)
