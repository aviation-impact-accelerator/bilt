import types
import typing

import pytest

from bilt.core import Interface, field, system
from bilt.core._equation import equation
from bilt.core._error import (
    EquationReturnError,
)
from bilt.symbolic._atoms import ADD, EQUALS
from bilt.symbolic._expressions import (
    Expr,
    SubSystemExpr,
)
from bilt.symbolic._tree import Tree


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
        EquationReturnError,
        match="Did you forget to compare it with another expression",
    ):
        equation(bad_model)


def test_equation_raises_if_function_returns_non_relation_tree() -> None:
    def bad_model(x: Expr) -> object:
        return x.rep

    with pytest.raises(EquationReturnError, match="non-relation tree"):
        equation(bad_model)


def test_equation_error_mentions_function_name() -> None:
    def bad_model(x: Expr) -> object:
        return x

    with pytest.raises(EquationReturnError, match="'bad_model'"):
        equation(bad_model)


def test_equation_error_mentions_return_type() -> None:
    def bad_model(_x: Expr) -> object:
        return 1

    with pytest.raises(EquationReturnError, match="int"):
        equation(bad_model)


def test_equation_with_sub_system_parameter_has_correct_parameters() -> None:
    """Test that sub-system parameter is expanded to used ports, bare name not included."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    assert voltage_eq.__bilt_parameters__ == ("voltage", "resistor.v")
    assert "resistor" not in voltage_eq.__bilt_parameters__


def test_equation_records_sub_systems() -> None:
    """Test that __bilt_sub_systems__ maps parameter name to system class."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    assert isinstance(voltage_eq.__bilt_sub_systems__, types.MappingProxyType)
    assert voltage_eq.__bilt_sub_systems__["resistor"] is Resistor


def test_equation_with_multiple_ports_of_one_sub_system() -> None:
    """Test that multiple ports from same sub-system are expanded in sorted order."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def power_eq(resistor: Resistor) -> bool:
        return resistor.v == resistor.i * resistor.r

    assert power_eq.__bilt_parameters__ == ("resistor.i", "resistor.r", "resistor.v")


def test_equation_with_two_sub_systems_of_same_type() -> None:
    """Test that two sub-system parameters of same type are expanded independently."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def voltage_balance(r1: Resistor, r2: Resistor) -> bool:
        return r1.v + r2.v == 0

    assert "r1.v" in voltage_balance.__bilt_parameters__
    assert "r2.v" in voltage_balance.__bilt_parameters__
    r1_idx = voltage_balance.__bilt_parameters__.index("r1.v")
    r2_idx = voltage_balance.__bilt_parameters__.index("r2.v")
    assert r1_idx < r2_idx

    assert "r1" in voltage_balance.__bilt_sub_systems__
    assert "r2" in voltage_balance.__bilt_sub_systems__


def test_ordinary_equation_with_no_sub_systems_unaffected() -> None:
    """Test that equations without sub-systems have empty __bilt_sub_systems__."""

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    assert ohms_law.__bilt_parameters__ == ("v", "i", "r")
    assert ohms_law.__bilt_sub_systems__ == {}


def test_sub_system_port_access_with_invalid_port_raises_attribute_error() -> None:
    """Test that accessing non-existent port raises AttributeError at definition time."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    with pytest.raises(AttributeError, match="'resistor' has no port 'typo'"):

        @equation
        def bad_eq(resistor: Resistor) -> bool:
            return resistor.typo == 1  # pyrefly: ignore[missing-attribute]  # ty: ignore[unresolved-attribute]


def test_sub_system_port_with_leading_underscore_is_usable() -> None:
    """A single leading underscore is a legal port name and must not be mistaken for a private attribute."""

    @system
    class Sensor(metaclass=Interface):
        _flux: float = field()
        reading: float = field()

    @equation
    def flux_eq(value: float, sensor: Sensor) -> bool:
        return value == sensor._flux  # noqa: SLF001

    assert flux_eq.__bilt_parameters__ == ("value", "sensor._flux")


def test_sub_system_dunder_access_still_raises_attribute_error() -> None:
    """Dunder probes are never ports, so they must keep raising rather than becoming symbols."""

    @system
    class Sensor(metaclass=Interface):
        reading: float = field()

    sub_system_expr = SubSystemExpr("sensor", {port: f"sensor.{port}" for port in Sensor.__bilt_parameters__})

    with pytest.raises(AttributeError, match="has no attribute '__deepcopy__'"):
        _ = sub_system_expr.__deepcopy__
