"""Validate Bilt's quasi-Monte Carlo broadcasting against NumPy.

This test deliberately uses an independently written NumPy reference rather
than reusing the transform functions from the example. It therefore checks both
the domain lifting and the numerical model.

Expected location:

    packages/examples/tests/test_quasi_mc_broadcasting.py
"""

import numpy as np
import numpy.typing as npt

import examples.quasi_mc_broadcasting.sobol_broadcasting_bilt as example


def _numpy_reference(
    *,
    specific_energy_kwh_per_t: npt.NDArray[np.float64],
    propulsive_efficiency: npt.NDArray[np.float64],
    energy_multiplier: npt.NDArray[np.float64],
    distances_km: npt.NDArray[np.float64],
    payloads_t: npt.NDArray[np.float64],
    empty_mass_t: float,
    energy_per_km_t: float,
    usable_fraction: float,
    max_battery_mass_t: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Evaluate the same model using explicit NumPy axis insertion.

    Input domains:

    - uncertain quantities: ``(scramble, sample)``
    - distances: ``(route,)``
    - payloads: ``(payload_case,)``

    Output domains:

    ``(scramble, sample, route, payload_case)``
    """

    gross_mass_t = (
        empty_mass_t
        + payloads_t[None, None, None, :]
    )

    shaft_energy_kwh = (
        energy_multiplier[:, :, None, None]
        * distances_km[None, None, :, None]
        * gross_mass_t
        * energy_per_km_t
    )

    electrical_energy_kwh = (
        shaft_energy_kwh
        / propulsive_efficiency[:, :, None, None]
    )

    battery_mass_t = (
        electrical_energy_kwh
        / (
            specific_energy_kwh_per_t[:, :, None, None]
            * usable_fraction
        )
    )

    battery_margin_t = max_battery_mass_t - battery_mass_t
    return battery_mass_t, battery_margin_t


def test_qmc_broadcasting_matches_numpy_reference() -> None:
    """Bilt should match NumPy over all four broadcast domains."""

    # Keep eight independent randomizations, but use fewer points per scramble
    # than the tutorial's full run so the validation test remains quick.
    scrambles = 8
    power = 6
    samples = 2**power
    seed = 1_000

    distances_km = np.array(
        [150.0, 250.0, 350.0, 450.0, 550.0, 650.0],
        dtype=float,
    )
    payloads_t = np.array(
        [2.0, 4.0, 6.0, 8.0],
        dtype=float,
    )

    empty_mass_t = 20.0
    energy_per_km_t = 0.18
    usable_fraction = 0.85
    max_battery_mass_t = 7.5

    draws = example.scrambled_sobol_inputs(
        scrambles=scrambles,
        power=power,
        seed=seed,
    )

    scramble = example.Dim("scramble")
    sample = example.Dim("sample")
    route = example.Dim("route")
    payload_case = example.Dim("payload_case")

    mission = example.Mission(
        specific_energy_kwh_per_t=example.tensor(
            draws.specific_energy_kwh_per_t,
            over=(scramble, sample),
        ),
        propulsive_efficiency=example.tensor(
            draws.propulsive_efficiency,
            over=(scramble, sample),
        ),
        energy_multiplier=example.tensor(
            draws.energy_multiplier,
            over=(scramble, sample),
        ),
        distance_km=example.tensor(
            distances_km,
            over=route,
        ),
        payload_t=example.tensor(
            payloads_t,
            over=payload_case,
        ),
        empty_mass_t=empty_mass_t,
        energy_per_km_t=energy_per_km_t,
        usable_fraction=usable_fraction,
        max_battery_mass_t=max_battery_mass_t,
    )

    with example.PythonBackend(), example.mission_model:
        actual_battery_mass_t = example.as_array(
            mission.battery_mass_t,
        )
        actual_battery_margin_t = example.as_array(
            mission.battery_margin_t,
        )

    expected_battery_mass_t, expected_battery_margin_t = _numpy_reference(
        specific_energy_kwh_per_t=draws.specific_energy_kwh_per_t,
        propulsive_efficiency=draws.propulsive_efficiency,
        energy_multiplier=draws.energy_multiplier,
        distances_km=distances_km,
        payloads_t=payloads_t,
        empty_mass_t=empty_mass_t,
        energy_per_km_t=energy_per_km_t,
        usable_fraction=usable_fraction,
        max_battery_mass_t=max_battery_mass_t,
    )

    expected_shape = (
        scrambles,
        samples,
        distances_km.size,
        payloads_t.size,
    )

    assert actual_battery_mass_t.shape == expected_shape
    assert actual_battery_margin_t.shape == expected_shape

    np.testing.assert_allclose(
        actual_battery_mass_t,
        expected_battery_mass_t,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual_battery_margin_t,
        expected_battery_margin_t,
        rtol=1e-12,
        atol=1e-12,
    )

    actual_feasible = actual_battery_margin_t >= 0.0
    expected_feasible = expected_battery_margin_t >= 0.0

    np.testing.assert_array_equal(
        actual_feasible,
        expected_feasible,
    )

    # Reduce the sample domain first, preserving independent scramble estimates.
    actual_probability_by_scramble = actual_feasible.mean(axis=1)
    expected_probability_by_scramble = expected_feasible.mean(axis=1)

    np.testing.assert_allclose(
        actual_probability_by_scramble,
        expected_probability_by_scramble,
        rtol=0.0,
        atol=0.0,
    )

    # Then combine the independently randomized QMC estimates.
    actual_probability = actual_probability_by_scramble.mean(axis=0)
    expected_probability = expected_probability_by_scramble.mean(axis=0)

    actual_standard_error = (
        actual_probability_by_scramble.std(axis=0, ddof=1)
        / np.sqrt(scrambles)
    )
    expected_standard_error = (
        expected_probability_by_scramble.std(axis=0, ddof=1)
        / np.sqrt(scrambles)
    )

    np.testing.assert_allclose(
        actual_probability,
        expected_probability,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual_standard_error,
        expected_standard_error,
        rtol=0.0,
        atol=0.0,
    )
