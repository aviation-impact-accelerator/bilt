from math import exp

import pytest
from examples.hx._physics import (
    element_temperatures_mixed_unmixed,
    local_effectiveness,
)


def test_single_element_against_closed_form_solution() -> None:
    """Unit test for the minimal mixed-unmixed element model.

    The expected values are calculated independently from the closed-form
    equations.

    Given:
        T_mixed_in = 350 K
        T_unmixed_in = 300 K
        C_mixed = 200 W/K
        C_unmixed = 50 W/K
        UA = 25 W/K

    The exact local effectiveness is:

        eps = 1 - exp(-UA / C_unmixed)

    The signed element heat transfer is:

        q = eps * C_unmixed * (T_mixed_in - T_unmixed_in)
            / (1 + eps * C_unmixed / (2 * C_mixed))

    Then:

        T_mixed_out = T_mixed_in - q / C_mixed
        T_unmixed_out = T_unmixed_in + q / C_unmixed
    """
    t_mixed_in = 350.0
    t_unmixed_in = 300.0
    heat_capacity_rate_mixed = 200.0
    heat_capacity_rate_unmixed = 50.0
    ua_element = 25.0

    expected_local_effectiveness = 1.0 - exp(-25.0 / 50.0)

    expected_q = (
        expected_local_effectiveness
        * heat_capacity_rate_unmixed
        * (t_mixed_in - t_unmixed_in)
        / (1.0 + expected_local_effectiveness * heat_capacity_rate_unmixed / (2.0 * heat_capacity_rate_mixed))
    )

    expected_t_mixed_out = t_mixed_in - expected_q / heat_capacity_rate_mixed
    expected_t_unmixed_out = t_unmixed_in + expected_q / heat_capacity_rate_unmixed

    result = element_temperatures_mixed_unmixed(
        temp_mixed_in=t_mixed_in,
        temp_unmixed_in=t_unmixed_in,
        heat_capacity_rate_mixed=heat_capacity_rate_mixed,
        heat_capacity_rate_unmixed=heat_capacity_rate_unmixed,
        thermal_conductance_element=ua_element,
    )

    assert expected_local_effectiveness == pytest.approx(0.3934693402873666)
    assert result.q_element_mixed_unmixed == pytest.approx(937.5606780351816)
    assert result.temp_mixed_out == pytest.approx(345.31219660982407)
    assert result.temp_unmixed_out == pytest.approx(318.7512135607036)

    assert result.q_element_mixed_unmixed == pytest.approx(expected_q)
    assert result.temp_mixed_out == pytest.approx(expected_t_mixed_out)
    assert result.temp_unmixed_out == pytest.approx(expected_t_unmixed_out)


def test_single_element_reverses_heat_flow_when_temperature_difference_reverses() -> None:
    """Same physical element, but now the unmixed side is hotter.

    The result should be exactly the same magnitude of heat transfer, but
    negative, because the sign convention is:

        q > 0: mixed/tube side heats unmixed/outer side
        q < 0: unmixed/outer side heats mixed/tube side
    """
    expected_local_effectiveness = 1.0 - exp(-25.0 / 50.0)

    result = element_temperatures_mixed_unmixed(
        temp_mixed_in=300.0,
        temp_unmixed_in=350.0,
        heat_capacity_rate_mixed=200.0,
        heat_capacity_rate_unmixed=50.0,
        thermal_conductance_element=25.0,
    )

    assert expected_local_effectiveness == pytest.approx(0.3934693402873666)
    assert result.q_element_mixed_unmixed == pytest.approx(-937.5606780351816)
    assert result.temp_mixed_out == pytest.approx(304.68780339017593)
    assert result.temp_unmixed_out == pytest.approx(331.2487864392964)


def test_zero_temperature_difference_gives_zero_heat_transfer() -> None:
    result = element_temperatures_mixed_unmixed(
        temp_mixed_in=300.0,
        temp_unmixed_in=300.0,
        heat_capacity_rate_mixed=200.0,
        heat_capacity_rate_unmixed=50.0,
        thermal_conductance_element=25.0,
    )

    assert result.q_element_mixed_unmixed == pytest.approx(0.0)
    assert result.temp_mixed_out == pytest.approx(300.0)
    assert result.temp_unmixed_out == pytest.approx(300.0)


def test_local_effectiveness_against_exact_exponential_value() -> None:
    assert local_effectiveness(
        thermal_conductance_element=25.0,
        heat_capacity_rate_unmixed=50.0,
    ) == pytest.approx(1.0 - exp(-0.5))
