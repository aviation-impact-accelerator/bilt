"""Independent dense-reference tests for acausal wing-and-tail trim."""

import numpy as np
import pytest
from examples.aircraft_trim.fixtures import example_aircraft, trim_case
from examples.aircraft_trim.model import CASE, MODEL, SOURCE_PANEL

from bilt import QuantityTensor, tensor
from bilt.units import degree, kilogram, meter, newton


def _influence() -> np.ndarray:
    """Return the reference matrix independently of the Bilt equations."""
    return np.array(
        [
            [-0.35, -0.07, -0.01, -0.01],
            [-0.07, -0.35, -0.01, -0.01],
            [-0.06, -0.06, -0.60, -0.07],
            [-0.06, -0.06, -0.07, -0.60],
        ]
    )


def _reference_trim(mass_kg: float) -> np.ndarray:
    """Solve a dense 6x6 linear system independently of Bilt."""
    width = np.array([4.0, 4.0, 2.0, 2.0])
    arm = np.array([0.5, 0.5, -4.0, -4.0])
    control = np.array([0.0, 0.0, 1.0, 1.0])
    speed = 70.0
    rho = 1.225
    matrix = np.zeros((6, 6))
    matrix[:4, :4] = _influence()
    matrix[:4, 4] = speed
    matrix[:4, 5] = speed * control
    matrix[4, :4] = rho * speed * width
    matrix[5, :4] = rho * speed * width * arm
    return np.linalg.solve(matrix, [0.0, 0.0, 0.0, 0.0, mass_kg * 9.81, 0.0])


def test_trim_solves_panel_circulation_alpha_and_elevator() -> None:
    """Inverse trim determines both angles and all four circulations."""
    with MODEL:
        aircraft = trim_case(550.0)
        actual_gamma = aircraft.circulation
        actual_alpha = aircraft.alpha
        actual_elevator = aircraft.elevator
        actual_lift = aircraft.lift

    expected = _reference_trim(550.0)
    assert isinstance(actual_gamma, QuantityTensor)
    np.testing.assert_allclose(actual_gamma.values, expected[:4], rtol=1e-6, atol=1e-8)
    assert actual_alpha.convert_to(degree).value == pytest.approx(np.rad2deg(expected[4]), rel=1e-6)
    assert actual_elevator.convert_to(degree).value == pytest.approx(np.rad2deg(expected[5]), rel=1e-6)
    assert actual_lift.convert_to(newton).value == pytest.approx(550.0 * 9.81, rel=1e-6)


def test_forward_evaluation_reproduces_trim_without_a_mass_constraint() -> None:
    """The same equations work forwards with the solved angles as inputs."""
    with MODEL:
        inverse = trim_case(550.0)
        forward = example_aircraft(alpha=inverse.alpha, elevator=inverse.elevator)
        lift = forward.lift
        moment = forward.pitch_moment
        circulation = forward.circulation

    assert lift.convert_to(newton).value == pytest.approx(550.0 * 9.81, rel=1e-6)
    assert moment.convert_to(newton * meter).value == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(circulation.values, _reference_trim(550.0)[:4], rtol=1e-6)


def test_untrimmed_forward_calculation_retains_nonzero_moment() -> None:
    """Supplying angles alone must not silently impose a trim constraint."""
    with MODEL:
        aircraft = example_aircraft(alpha=3.0 * degree, elevator=0.0 * degree)
        circulation = aircraft.circulation
        moment = aircraft.pitch_moment
        lift = aircraft.lift

    expected = np.linalg.solve(_influence(), -70.0 * np.deg2rad(3.0) * np.ones(4))
    panel_lift = 1.225 * 70.0 * expected * np.array([4.0, 4.0, 2.0, 2.0])
    np.testing.assert_allclose(circulation.values, expected, rtol=1e-6)
    assert lift.convert_to(newton).value == pytest.approx(panel_lift.sum(), rel=1e-6)
    assert moment.convert_to(newton * meter).value == pytest.approx(
        np.dot(panel_lift, [0.5, 0.5, -4.0, -4.0]), rel=1e-6
    )


def test_multiple_masses_preserve_case_and_panel_axes() -> None:
    """One coupled trim problem per case, with panels reduced only by name."""
    masses = [350.0, 550.0, 700.0]
    with MODEL:
        aircraft = example_aircraft(
            mass=tensor(masses, over=CASE, unit=kilogram),
            pitch_moment=0.0 * newton * meter,
        )
        circulation = aircraft.circulation
        alpha = aircraft.alpha
        elevator = aircraft.elevator

    assert isinstance(circulation, QuantityTensor)
    assert set(circulation.domain.dims) == {CASE, SOURCE_PANEL}
    axis_case = circulation.domain.dims.index(CASE)
    axis_panel = circulation.domain.dims.index(SOURCE_PANEL)
    gamma = np.moveaxis(circulation.values, [axis_case, axis_panel], [0, 1])
    expected = np.stack([_reference_trim(mass) for mass in masses])
    np.testing.assert_allclose(gamma, expected[:, :4], rtol=1e-6)
    np.testing.assert_allclose(alpha.convert_to(degree).values, np.rad2deg(expected[:, 4]), rtol=1e-6)
    np.testing.assert_allclose(elevator.convert_to(degree).values, np.rad2deg(expected[:, 5]), rtol=1e-6)
