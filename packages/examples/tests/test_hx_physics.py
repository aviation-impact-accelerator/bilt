from math import exp

import pytest
from examples.hx._physics import (
    local_effectiveness,
    mixed_unmixed_element_temperatures,
)


def test_single_element_against_closed_form_solution() -> None:
    """Unit test for the minimal mixed-unmixed element model.

    The expected values are calculated independently from the closed-form
    equations.

    Given:
        mixed_inlet_temperature = 350 K
        unmixed_inlet_temperature = 300 K
        mixed_heat_capacity_rate = 200 W/K
        unmixed_heat_capacity_rate = 50 W/K
        thermal_conductance_per_element = 25 W/K

    The exact local effectiveness is:

        effectiveness = 1 - exp(
            -thermal_conductance_per_element / unmixed_heat_capacity_rate
        )

    The signed heat transfer rate is:

        heat_transfer_rate = (
            effectiveness
            * unmixed_heat_capacity_rate
            * (mixed_inlet_temperature - unmixed_inlet_temperature)
            / (
                1
                + effectiveness
                * unmixed_heat_capacity_rate
                / (2 * mixed_heat_capacity_rate)
            )
        )

    Then:

        mixed_outlet_temperature = (
            mixed_inlet_temperature
            - heat_transfer_rate / mixed_heat_capacity_rate
        )

        unmixed_outlet_temperature = (
            unmixed_inlet_temperature
            + heat_transfer_rate / unmixed_heat_capacity_rate
        )
    """
    mixed_inlet_temperature = 350.0
    unmixed_inlet_temperature = 300.0
    mixed_heat_capacity_rate = 200.0
    unmixed_heat_capacity_rate = 50.0
    thermal_conductance_per_element = 25.0

    expected_effectiveness = 1.0 - exp(-thermal_conductance_per_element / unmixed_heat_capacity_rate)

    expected_heat_transfer_rate = (
        expected_effectiveness
        * unmixed_heat_capacity_rate
        * (mixed_inlet_temperature - unmixed_inlet_temperature)
        / (1.0 + expected_effectiveness * unmixed_heat_capacity_rate / (2.0 * mixed_heat_capacity_rate))
    )

    expected_mixed_outlet_temperature = mixed_inlet_temperature - expected_heat_transfer_rate / mixed_heat_capacity_rate
    expected_unmixed_outlet_temperature = (
        unmixed_inlet_temperature + expected_heat_transfer_rate / unmixed_heat_capacity_rate
    )

    result = mixed_unmixed_element_temperatures(
        mixed_inlet_temperature=mixed_inlet_temperature,
        unmixed_inlet_temperature=unmixed_inlet_temperature,
        mixed_heat_capacity_rate=mixed_heat_capacity_rate,
        unmixed_heat_capacity_rate=unmixed_heat_capacity_rate,
        thermal_conductance_per_element=thermal_conductance_per_element,
    )

    assert expected_effectiveness == pytest.approx(0.3934693402873666)

    assert result.heat_transfer_rate_mixed_to_unmixed == pytest.approx(937.5606780351816)
    assert result.mixed_outlet_temperature == pytest.approx(345.31219660982407)
    assert result.unmixed_outlet_temperature == pytest.approx(318.7512135607036)

    assert result.heat_transfer_rate_mixed_to_unmixed == pytest.approx(expected_heat_transfer_rate)
    assert result.mixed_outlet_temperature == pytest.approx(expected_mixed_outlet_temperature)
    assert result.unmixed_outlet_temperature == pytest.approx(expected_unmixed_outlet_temperature)


def test_single_element_reverses_heat_flow_when_temperature_difference_reverses() -> None:
    """Same physical element, but now the unmixed side is hotter.

    The result should be exactly the same magnitude of heat transfer, but
    negative, because the sign convention is:

        heat_transfer_rate > 0:
            mixed/tube side heats unmixed/outer side

        heat_transfer_rate < 0:
            unmixed/outer side heats mixed/tube side
    """
    expected_effectiveness = 1.0 - exp(-25.0 / 50.0)

    result = mixed_unmixed_element_temperatures(
        mixed_inlet_temperature=300.0,
        unmixed_inlet_temperature=350.0,
        mixed_heat_capacity_rate=200.0,
        unmixed_heat_capacity_rate=50.0,
        thermal_conductance_per_element=25.0,
    )

    assert expected_effectiveness == pytest.approx(0.3934693402873666)

    assert result.heat_transfer_rate_mixed_to_unmixed == pytest.approx(-937.5606780351816)
    assert result.mixed_outlet_temperature == pytest.approx(304.68780339017593)
    assert result.unmixed_outlet_temperature == pytest.approx(331.2487864392964)


def test_zero_temperature_difference_gives_zero_heat_transfer() -> None:
    result = mixed_unmixed_element_temperatures(
        mixed_inlet_temperature=300.0,
        unmixed_inlet_temperature=300.0,
        mixed_heat_capacity_rate=200.0,
        unmixed_heat_capacity_rate=50.0,
        thermal_conductance_per_element=25.0,
    )

    assert result.heat_transfer_rate_mixed_to_unmixed == pytest.approx(0.0)
    assert result.mixed_outlet_temperature == pytest.approx(300.0)
    assert result.unmixed_outlet_temperature == pytest.approx(300.0)


def test_local_effectiveness_against_exact_exponential_value() -> None:
    assert local_effectiveness(
        thermal_conductance_per_element=25.0,
        unmixed_heat_capacity_rate=50.0,
    ) == pytest.approx(1.0 - exp(-0.5))
