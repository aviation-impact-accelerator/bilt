"""Geometry and numerical reference tests for acausal wing-tail trim."""

from dataclasses import replace

import numpy as np
import pytest
from examples.aircraft_trim.fixtures import forward_case, trim_case
from examples.aircraft_trim.geometry import (
    RectangularSurface,
    calculate_influences,
    example_geometry,
    finite_vortex_velocity,
)
from examples.aircraft_trim.model import MODEL, TAIL_SOURCE, WING_SOURCE

from bilt import QuantityTensor
from bilt.units import degree, meter, newton, second


def _reference_operator(
    wing_panels: int, tail_panels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble a dense system independently from the four physical influences."""
    geometry = example_geometry(wing_panels=wing_panels, tail_panels=tail_panels)
    data = calculate_influences(geometry)
    a = np.block(
        [
            [data.wing_from_wing, data.wing_from_tail],
            [data.tail_from_wing, data.tail_from_tail],
        ]
    )
    widths = np.r_[data.wing.width_m, data.tail.width_m]
    arms = np.r_[data.wing.arm_m, data.tail.arm_m]
    twist = np.r_[data.wing.twist_rad, data.tail.twist_rad]
    control = np.r_[np.zeros(wing_panels), np.ones(tail_panels)]
    return a, widths, arms, twist, control


def _reference_trim(mass: float, wing_panels: int, tail_panels: int) -> np.ndarray:
    """Direct NumPy solve: N aerodynamic rows and two global trim constraints."""
    a, widths, arms, twist, control = _reference_operator(wing_panels, tail_panels)
    n = wing_panels + tail_panels
    matrix = np.zeros((n + 2, n + 2))
    matrix[:n, :n] = a
    matrix[:n, n] = 70.0
    matrix[:n, n + 1] = 70.0 * control
    matrix[n, :n] = 1.225 * 70.0 * widths
    matrix[n + 1, :n] = 1.225 * 70.0 * widths * arms
    return np.linalg.solve(matrix, np.r_[-70.0 * twist, mass * 9.81, 0.0])


def _reference_forward(
    alpha: float, elevator: float, wing_panels: int, tail_panels: int
) -> tuple[np.ndarray, float, float]:
    """Direct NumPy solve for circulation and integrated lift/moment."""
    a, widths, arms, twist, control = _reference_operator(wing_panels, tail_panels)
    gamma = np.linalg.solve(a, -70.0 * (alpha + twist + control * elevator))
    loads = 1.225 * 70.0 * gamma * widths
    return gamma, float(loads.sum()), float(np.dot(loads, arms))


def _combine(wing: QuantityTensor, tail: QuantityTensor) -> np.ndarray:
    return np.r_[wing.convert_to(meter**2 / second).values, tail.convert_to(meter**2 / second).values]


@pytest.mark.parametrize(("wing_panels", "tail_panels"), ((2, 2), (6, 4)))
def test_geometry_based_forward_matches_numpy(wing_panels: int, tail_panels: int) -> None:
    """Bilt solves each source-panel circulation from physical vortex influence."""
    geometry = example_geometry(wing_panels=wing_panels, tail_panels=tail_panels)
    with MODEL:
        aircraft = forward_case(3.0, -1.0, geometry=geometry)
        wing = aircraft.wing_circulation
        tail = aircraft.tail_circulation
        lift = aircraft.lift
        moment = aircraft.pitch_moment

    expected, expected_lift, expected_moment = _reference_forward(
        np.deg2rad(3.0), np.deg2rad(-1.0), wing_panels, tail_panels
    )
    assert isinstance(wing, QuantityTensor)
    assert isinstance(tail, QuantityTensor)
    assert wing.domain.dims == (WING_SOURCE,)
    assert tail.domain.dims == (TAIL_SOURCE,)
    np.testing.assert_allclose(_combine(wing, tail), expected, rtol=1e-6, atol=1e-7)
    assert np.ptp(expected) > 0.0
    assert lift.convert_to(newton).value == pytest.approx(expected_lift, rel=1e-6)
    assert moment.convert_to(newton * meter).value == pytest.approx(expected_moment, rel=1e-6)


@pytest.mark.parametrize(("wing_panels", "tail_panels"), ((2, 2), (6, 4)))
def test_geometry_based_inverse_trim_matches_numpy(wing_panels: int, tail_panels: int) -> None:
    """N panel unknowns plus two global unknowns form an (N+2)-square problem."""
    geometry = example_geometry(wing_panels=wing_panels, tail_panels=tail_panels)
    with MODEL:
        aircraft = trim_case(550.0, geometry=geometry)
        wing = aircraft.wing_circulation
        tail = aircraft.tail_circulation
        alpha = aircraft.alpha
        elevator = aircraft.elevator

    expected = _reference_trim(550.0, wing_panels, tail_panels)
    assert isinstance(wing, QuantityTensor)
    assert isinstance(tail, QuantityTensor)
    np.testing.assert_allclose(_combine(wing, tail), expected[:-2], rtol=1e-6, atol=1e-7)
    assert alpha.convert_to(degree).value == pytest.approx(np.rad2deg(expected[-2]), rel=1e-6)
    assert elevator.convert_to(degree).value == pytest.approx(np.rad2deg(expected[-1]), rel=1e-6)


def test_trim_angles_reproduce_lift_and_zero_pitch() -> None:
    """Re-evaluate the same equations forwards using inverse-trim angles."""
    geometry = example_geometry(wing_panels=6, tail_panels=4)
    with MODEL:
        trimmed = trim_case(550.0, geometry=geometry)
        forward = forward_case(
            trimmed.alpha.convert_to(degree).value,
            trimmed.elevator.convert_to(degree).value,
            geometry=geometry,
        )
        lift = forward.lift
        moment = forward.pitch_moment
        wing_load = forward.wing_panel_lift
        tail_load = forward.tail_panel_lift

    assert lift.convert_to(newton).value == pytest.approx(550.0 * 9.81, rel=1e-6)
    assert moment.convert_to(newton * meter).value == pytest.approx(0.0, abs=1e-6)
    assert isinstance(wing_load, QuantityTensor)
    assert isinstance(tail_load, QuantityTensor)
    assert wing_load.domain.dims == (WING_SOURCE,)
    assert tail_load.domain.dims == (TAIL_SOURCE,)


@pytest.mark.parametrize("mass", (420.0, 550.0, 680.0))
def test_inverse_trim_for_multiple_weights(mass: float) -> None:
    """Check the solver with different loads at a fixed geometry."""
    with MODEL:
        aircraft = trim_case(mass)
        wing = aircraft.wing_circulation
        tail = aircraft.tail_circulation
    np.testing.assert_allclose(_combine(wing, tail), _reference_trim(mass, 2, 2)[:-2], rtol=1e-6)


def test_finite_segment_biot_savart_against_analytic_value() -> None:
    """A centred unit straight vortex has a known velocity at unit distance."""
    velocity = finite_vortex_velocity(
        np.array([0.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(velocity, [0.0, 0.0, 1.0 / (2.0 * np.pi * np.sqrt(2.0))], atol=1e-12)


def test_symmetric_geometry_gives_symmetric_spanwise_influence() -> None:
    """Mirror symmetry provides a geometry-based influence regression."""
    data = calculate_influences(example_geometry(wing_panels=6, tail_panels=4))
    np.testing.assert_allclose(data.wing_from_wing, data.wing_from_wing[::-1, ::-1], atol=1e-12)
    np.testing.assert_allclose(data.tail_from_tail, data.tail_from_tail[::-1, ::-1], atol=1e-12)
    np.testing.assert_allclose(data.wing_from_tail, data.wing_from_tail[::-1, ::-1], atol=1e-12)
    assert np.all(np.diag(data.wing_from_wing) < 0.0)


def test_wake_length_convergence_and_cross_surface_coupling() -> None:
    """Farther wake truncation should not dramatically alter a near-field matrix."""
    near = example_geometry(wing_panels=6, tail_panels=4)
    far = replace(near, wake_length_m=800.0)
    first = calculate_influences(near)
    second = calculate_influences(far)
    np.testing.assert_allclose(first.wing_from_wing, second.wing_from_wing, rtol=2e-4, atol=2e-6)
    assert np.any(abs(first.tail_from_wing) > 0.0)


@pytest.mark.parametrize(
    ("span", "chord", "panels"),
    ((0.0, 1.0, 2), (2.0, 0.0, 2), (2.0, 1.0, 0)),
)
def test_invalid_surface_geometry_is_rejected(span: float, chord: float, panels: int) -> None:
    """Reject invalid geometry early, rather than entering numerical assembly."""
    with pytest.raises(ValueError, match="span, chord and panel count"):
        RectangularSurface(span, chord, 0.0, 0.0, panels)
