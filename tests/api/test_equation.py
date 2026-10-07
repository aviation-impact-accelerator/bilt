import types
import typing

import pytest

from bilt import Interface, field, system
from bilt.core._equation import equation
from bilt.errors import (
    BiltEquationReturnError,
    BiltEquationSignatureError,
    BiltNameError,
    BiltUnknownPortError,
)
from bilt.symbolic._atoms import ADD, EQUALS

# Expr is used in @equation function annotations. Those annotations are part of
# the runtime behaviour being tested, so this must stay as a runtime import.
from bilt.symbolic._expressions import Expr  # noqa: TC001
from bilt.symbolic._tree import Tree, symbol_tree
from bilt.symbolic._walk import contains


@system
class Resistor(metaclass=Interface):
    v: float = field()
    i: float = field()
    r: float = field()


@system
class Sensor(metaclass=Interface):
    _flux: float = field()
    reading: float = field()


def test_equation_builds_tree_from_function() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    assert model.__bilt_name__ == "model"
    assert model.__bilt_parameters__ == ("a", "b", "c")
    assert tuple(model.__bilt_equation_symbols__) == ("a", "b", "c")

    tree = model.__bilt_equation_tree__

    assert isinstance(tree, Tree)
    assert tree.is_relation is True
    assert tree.is_equality is True
    assert tree.head is EQUALS


def test_equation_passes_expr_symbols_to_function() -> None:
    seen: dict[str, Expr] = {}

    @equation
    def model(a: float, b: float, c: float) -> bool:
        seen["a"] = typing.cast("Expr", a)
        seen["b"] = typing.cast("Expr", b)
        seen["c"] = typing.cast("Expr", c)
        return a == b + c

    assert model.__bilt_equation_symbols__["a"] is seen["a"]
    assert model.__bilt_equation_symbols__["b"] is seen["b"]
    assert model.__bilt_equation_symbols__["c"] is seen["c"]


def test_equation_tree_uses_generated_symbols() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    symbols = model.__bilt_equation_symbols__
    tree = model.__bilt_equation_tree__

    assert tree.left is symbols["a"].rep

    rhs = tree.right

    assert rhs.head is ADD
    assert rhs.args == (symbols["b"].rep, symbols["c"].rep)


def test_equation_records_parameters_in_signature_order() -> None:
    @equation
    def model(z: float, x: float, y: float) -> bool:
        return z == x + y

    assert model.__bilt_parameters__ == ("z", "x", "y")
    assert tuple(model.__bilt_equation_symbols__) == ("z", "x", "y")


def test_equation_str_includes_name_and_tree() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    assert str(model) == f"model: {model.__bilt_equation_tree__}"


def test_equation_call_delegates_to_original_function() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    a = 1.0
    b = 2.0
    c = 3.0

    result = model(a, b, c)

    assert result == (a == b + c)


def test_equation_raises_if_function_returns_expr() -> None:
    def bad_model(x: Expr) -> object:
        return x

    with pytest.raises(
        BiltEquationReturnError,
        match="Did you forget to compare it with another expression",
    ):
        equation(bad_model)


def test_equation_raises_if_function_returns_non_relation_tree() -> None:
    def bad_model(x: Expr) -> object:
        return x.rep

    with pytest.raises(BiltEquationReturnError, match="non-relation tree"):
        equation(bad_model)


def test_equation_error_mentions_function_name() -> None:
    def bad_model(x: Expr) -> object:
        return x

    with pytest.raises(BiltEquationReturnError, match="'bad_model'"):
        equation(bad_model)


def test_equation_error_mentions_return_type() -> None:
    def bad_model(_x: Expr) -> object:
        return 1

    with pytest.raises(BiltEquationReturnError, match="int"):
        equation(bad_model)


def test_equation_expands_a_sub_system_parameter_into_the_ports_it_uses() -> None:
    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    assert voltage_eq.__bilt_parameters__ == ("voltage", "resistor.v")


def test_equation_symbols_match_its_parameters() -> None:
    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    symbols = voltage_eq.__bilt_equation_symbols__

    assert tuple(symbols) == voltage_eq.__bilt_parameters__
    assert symbols["resistor.v"].rep is symbol_tree("resistor.v")


def test_equation_tree_is_built_over_the_dotted_port() -> None:
    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    assert contains(voltage_eq.__bilt_equation_tree__, symbol_tree("resistor.v"))


def test_equation_records_the_sub_systems_it_was_declared_with() -> None:
    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    assert isinstance(voltage_eq.__bilt_sub_systems__, types.MappingProxyType)
    assert voltage_eq.__bilt_sub_systems__["resistor"] is Resistor


def test_equation_expands_several_ports_of_one_sub_system_in_sorted_order() -> None:
    @equation
    def power_eq(resistor: Resistor) -> bool:
        return resistor.v == resistor.i * resistor.r

    assert power_eq.__bilt_parameters__ == ("resistor.i", "resistor.r", "resistor.v")


def test_equation_expands_two_sub_systems_of_one_interface_independently() -> None:
    @equation
    def voltage_balance(r1: Resistor, r2: Resistor) -> bool:
        return r1.v + r2.v == 0

    assert voltage_balance.__bilt_parameters__ == ("r1.v", "r2.v")
    assert tuple(voltage_balance.__bilt_sub_systems__) == ("r1", "r2")


def test_equation_omits_a_sub_system_parameter_it_never_references() -> None:
    @equation
    def voltage_eq(voltage: float, supply: float, resistor: Resistor) -> bool:  # noqa: ARG001
        return voltage == supply

    assert voltage_eq.__bilt_parameters__ == ("voltage", "supply")
    assert tuple(voltage_eq.__bilt_sub_systems__) == ()


def test_equation_without_sub_systems_records_none() -> None:
    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    assert ohms_law.__bilt_parameters__ == ("v", "i", "r")
    assert ohms_law.__bilt_sub_systems__ == {}


def test_equation_rejects_an_unknown_sub_system_port_at_definition_time() -> None:
    with pytest.raises(BiltUnknownPortError, match=r"'resistor' has no port 'typo'"):

        @equation
        def bad_eq(resistor: Resistor) -> bool:
            return resistor.typo == 1  # pyrefly: ignore[missing-attribute]  # ty: ignore[unresolved-attribute]


def test_equation_accepts_a_sub_system_port_with_a_leading_underscore() -> None:
    @equation
    def flux_eq(value: float, sensor: Sensor) -> bool:
        return value == sensor._flux  # noqa: SLF001

    assert flux_eq.__bilt_parameters__ == ("value", "sensor._flux")


def test_equation_rejects_a_sub_system_parameter_named_with_the_mangled_separator() -> None:
    with pytest.raises(BiltNameError, match="'a__b' may not contain"):

        @equation
        def bad_eq(voltage: float, a__b: Resistor) -> bool:
            return voltage == a__b.v


def test_equation_rejects_a_plain_parameter_named_with_the_mangled_separator() -> None:
    """``a__b`` is how ``a.b`` is spelled once flattened, so the two would be one name to a backend."""
    with pytest.raises(BiltNameError, match="'a__b' may not contain"):

        @equation
        def bad_eq(voltage: float, a__b: float) -> bool:
            return voltage == a__b


def test_equation_rejects_var_keyword_parameters() -> None:
    def bad_eq(**kwargs: float) -> bool:
        return kwargs["voltage"] == kwargs["r"]

    with pytest.raises(BiltEquationSignatureError, match="must use only named parameters"):
        equation(bad_eq)
