"""Public API behaviour for scalar match/case equations."""

lazy from contextlib import AbstractContextManager

import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, equation, field, system
from bilt.errors import BiltAnalysisError, BiltDefinitionError


@system
class Engine(metaclass=Interface):
    mode: float = field()
    power: float = field()
    thrust: float = field()


@equation
def engine_thrust(mode: float, power: float, thrust: float) -> bool:
    match mode:
        case 0:
            selected = 0.0
        case 1:
            selected = power * 0.2
        case 2:
            selected = power
        case _:
            selected = power * 0.5
    return thrust == selected


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(
    ("mode", "expected"),
    ((0.0, 0.0), (1.0, 2.0), (2.0, 10.0), (3.0, 5.0), (-1.0, 5.0)),
)
def test_match_selects_each_mode(backend: AbstractContextManager[object], mode: float, expected: float) -> None:
    implementation = Implementation(Engine, [engine_thrust])
    with backend, implementation:
        assert float(Engine(mode=mode, power=10.0).thrust) == pytest.approx(expected)


def test_match_preserves_original_python_function() -> None:
    assert engine_thrust(0, 10.0, 0.0)
    assert engine_thrust(1, 10.0, 2.0)
    assert engine_thrust(2, 10.0, 10.0)
    assert engine_thrust(3, 10.0, 5.0)
    assert not engine_thrust(1, 10.0, 10.0)


@system
class Selected(metaclass=Interface):
    mode: float = field()
    first: float = field()
    second: float = field()
    result: float = field()


@equation
def choose_input(mode: float, first: float, second: float, result: float) -> bool:
    match mode:
        case 0:
            chosen = first
        case 1:
            chosen = second
        case _:
            chosen = first + second
    return result == chosen


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_match_tracks_dependencies_from_all_cases(backend: AbstractContextManager[object]) -> None:
    implementation = Implementation(Selected, [choose_input])
    with backend, implementation:
        assert float(Selected(mode=0.0, first=4.0, second=6.0).result) == pytest.approx(4.0)
        assert float(Selected(mode=1.0, first=4.0, second=6.0).result) == pytest.approx(6.0)
        assert float(Selected(mode=2.0, first=4.0, second=6.0).result) == pytest.approx(10.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_match_requires_inputs_used_by_other_cases(backend: AbstractContextManager[object]) -> None:
    implementation = Implementation(Selected, [choose_input])
    with backend, implementation, pytest.raises(BiltAnalysisError):
        _ = Selected(mode=0.0, first=4.0).result


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_match_cannot_invert_discrete_mode(backend: AbstractContextManager[object]) -> None:
    implementation = Implementation(Engine, [engine_thrust])
    with backend, implementation, pytest.raises(BiltAnalysisError, match="Cannot compute mode") as raised:
        _ = Engine(power=10.0, thrust=2.0).mode
    assert "through a conditional" in str(raised.value)


def test_match_guard_is_rejected_at_definition_time() -> None:
    with pytest.raises(BiltDefinitionError, match="match guards"):

        @equation
        def guarded(mode: float, result: float) -> bool:
            match mode:
                case 0 if mode >= 0:
                    selected = 1
                case _:
                    selected = 2
            return result == selected
