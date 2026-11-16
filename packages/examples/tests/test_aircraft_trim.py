"""Independent NumPy regression tests for four-panel acausal wing-tail trim."""

import numpy as np
import pytest
from examples.aircraft_trim.fixtures import forward_case, trim_case
from examples.aircraft_trim.model import MODEL, TAIL_SOURCE, WING_SOURCE

from bilt import QuantityTensor
from bilt.units import degree, meter, newton, second


def _influence() -> np.ndarray:
    """Dense 4x4 reference assembled independently from the four blocks."""
    return np.array(
        [
            [-0.35, -0.07, -0.015, -0.005],
            [-0.065, -0.34, -0.008, -0.014],
            [-0.06, -0.045, -0.59, -0.075],
            [-0.05, -0.065, -0.065, -0.61],
        ]
    )


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([4.1, 3.9, 2.1, 1.9]),
        np.array([0.5, 0.55, -3.9, -4.1]),
        np.array([0.005, -0.003, 0.004, -0.002]),
        np.array([0.0, 0.0, 1.0, 1.0]),
    )


def _reference_trim(mass: float) -> np.ndarray:
    """Solve four no-penetration rows plus two aggregate balance equations."""
    widths, arms, twist, control = _geometry()
    matrix = np.zeros((6, 6))
    matrix[:4, :4] = _influence()
    matrix[:4, 4] = 70.0
    matrix[:4, 5] = 70.0 * control
    matrix[4, :4] = 1.225 * 70.0 * widths
    matrix[5, :4] = 1.225 * 70.0 * widths * arms
    return np.linalg.solve(matrix, np.r_[-70.0 * twist, mass * 9.81, 0.0])


def _reference_forward(alpha: float, elevator: float) -> tuple[np.ndarray, float, float]:
    """Solve the four boundary rows, then integrate Kutta-Joukowski loading."""
    widths, arms, twist, control = _geometry()
    gamma = np.linalg.solve(_influence(), -70.0 * (alpha + twist + control * elevator))
    loads = 1.225 * 70.0 * gamma * widths
    return gamma, float(loads.sum()), float(np.dot(loads, arms))


def _combine(wing: QuantityTensor, tail: QuantityTensor) -> np.ndarray:
    """Combine independently solved wing and tail values for the dense reference."""
    return np.r_[wing.convert_to(meter**2 / second).values, tail.convert_to(meter**2 / second).values]


def test_four_independent_circulations_forward() -> None:
    """Two independent panels on each surface, coupled by four influence blocks."""
    with MODEL:
        aircraft = forward_case(3.0, -1.0)
        wing = aircraft.wing_circulation
        tail = aircraft.tail_circulation
        lift = aircraft.lift
        moment = aircraft.pitch_moment

    expected, expected_lift, expected_moment = _reference_forward(np.deg2rad(3.0), np.deg2rad(-1.0))
    assert isinstance(wing, QuantityTensor)
    assert isinstance(tail, QuantityTensor)
    assert wing.domain.dims == (WING_SOURCE,)
    assert tail.domain.dims == (TAIL_SOURCE,)
    np.testing.assert_allclose(_combine(wing, tail), expected, rtol=1e-6)
    assert np.unique(np.round(expected, 4)).size == len(expected)
    assert lift.convert_to(newton).value == pytest.approx(expected_lift, rel=1e-6)
    assert moment.convert_to(newton * meter).value == pytest.approx(expected_moment, rel=1e-6)


def test_inverse_spanwise_trim_solves_all_circulations_and_both_angles() -> None:
    """Solve four panel unknowns plus shared alpha and elevator using Bilt."""
    with MODEL:
        aircraft = trim_case(550.0)
        wing = aircraft.wing_circulation
        tail = aircraft.tail_circulation
        alpha = aircraft.alpha
        elevator = aircraft.elevator

    expected = _reference_trim(550.0)
    assert isinstance(wing, QuantityTensor)
    assert isinstance(tail, QuantityTensor)
    np.testing.assert_allclose(_combine(wing, tail), expected[:4], rtol=1e-6)
    assert alpha.convert_to(degree).value == pytest.approx(np.rad2deg(expected[4]), rel=1e-6)
    assert elevator.convert_to(degree).value == pytest.approx(np.rad2deg(expected[5]), rel=1e-6)


def test_trim_angles_reproduce_lift_and_zero_pitch() -> None:
    """Reuse the computed angles in a new forward instance of the same model."""
    with MODEL:
        trimmed = trim_case(550.0)
        forward = forward_case(
            trimmed.alpha.convert_to(degree).value,
            trimmed.elevator.convert_to(degree).value,
        )
        lift = forward.lift
        moment = forward.pitch_moment
        wing_panel_lift = forward.wing_panel_lift
        tail_panel_lift = forward.tail_panel_lift

    assert lift.convert_to(newton).value == pytest.approx(550.0 * 9.81, rel=1e-6)
    assert moment.convert_to(newton * meter).value == pytest.approx(0.0, abs=1e-6)
    assert isinstance(wing_panel_lift, QuantityTensor)
    assert isinstance(tail_panel_lift, QuantityTensor)
    assert wing_panel_lift.domain.dims == (WING_SOURCE,)
    assert tail_panel_lift.domain.dims == (TAIL_SOURCE,)


@pytest.mark.parametrize("mass", (420.0, 550.0, 680.0))
def test_inverse_trim_for_multiple_weights(mass: float) -> None:
    """Compare independent trimmed circulation solutions at different weights."""
    with MODEL:
        aircraft = trim_case(mass)
        wing = aircraft.wing_circulation
        tail = aircraft.tail_circulation
    np.testing.assert_allclose(_combine(wing, tail), _reference_trim(mass)[:4], rtol=1e-6)
