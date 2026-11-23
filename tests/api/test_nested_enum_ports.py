"""Nested categorical ports: propagation, execution and exact type checking."""

import enum
import typing

import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, equation, field, system, transform
from bilt.errors import BiltDefinitionError, BiltEvaluationError
from bilt.symbolic._atoms import EnumType
from bilt.symbolic._walk import subtrees

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


class Mode(enum.Enum):
    OFF = "off"
    ON = "on"


class OtherMode(enum.Enum):
    OFF = "off"
    ON = "on"


@system
class Child(metaclass=Interface):
    mode: Mode = field()
    power: float = field()
    result: float = field()


@equation
def result(mode: Mode, power: float, result: float) -> bool:
    match mode:
        case Mode.OFF:
            chosen = 0.0
        case _:
            chosen = power
    return result == chosen


@system
class Parent(metaclass=Interface):
    total: float = field()


@equation
def total(child: Child, total: float) -> bool:
    match child.mode:
        case Mode.OFF:
            chosen = 0.0
        case _:
            chosen = child.power
    return total == chosen


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_parent_enum_match_with_qualified_child_input(backend: Backend) -> None:
    child = Implementation(Child, [result])
    parent = Implementation(Parent, [total], internal_implementations={"child": child})
    expected_on = 4.0
    assert parent.evaluate({"child.mode": Mode.ON, "child.power": expected_on}, "total", backend) == expected_on
    assert parent.evaluate({"child.mode": Mode.OFF, "child.power": 4.0}, "total", backend) == 0.0
    assert parent.evaluate({"child.mode": Mode.ON}, "child.mode", backend) is Mode.ON
    for wrong in (OtherMode.ON, 1, "on"):
        with pytest.raises(BiltEvaluationError, match=r"'child\.mode'.*requires a Mode enum member"):
            parent.evaluate({"child.mode": wrong, "child.power": 4.0}, "total", backend)
    with pytest.raises(BiltEvaluationError, match=r"forward-only"):
        parent.evaluate({"child.power": 4.0, "total": 4.0}, "child.mode", backend)


def test_child_enum_is_typed_in_the_symbolic_tree() -> None:
    leaves = [n for n in subtrees(total.__bilt_equation_tree__) if n.head.atom_type is EnumType]
    assert (Mode, "OFF") in {node.head.value for node in leaves}
    assert total.__bilt_enum_ports__["child.mode"] is Mode


def test_nested_enum_arithmetic_rejected() -> None:
    with pytest.raises(BiltDefinitionError, match="only be used as match subjects"):

        @equation
        def bad(child: Child, total: float) -> bool:
            return total == child.mode * 2


@transform
def mode() -> Mode:
    return Mode.ON


@equation
def chosen_total(child: Child, total: float) -> bool:
    match child.mode:
        case Mode.ON:
            chosen = 1.5
        case _:
            chosen = 0.0
    return total == chosen


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_nested_transform_can_produce_enum_for_parent_equation(backend: Backend) -> None:
    child = Implementation(Child, [mode])
    parent = Implementation(Parent, [chosen_total], internal_implementations={"child": child})
    assert parent.evaluate({}, "child.mode", backend) is Mode.ON
    expected_total = 1.5
    assert parent.evaluate({}, "total", backend) == expected_total
