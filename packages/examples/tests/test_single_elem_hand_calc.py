from math import exp, log, pi

import pytest
from examples.hx._htc import overall_u_outer_basis_from_h, thermal_conductance_element
from examples.hx._physics import heat_capacity_rate, tube_outer_element_temperatures
from examples.hx._solver import march_single_tube
from examples.hx.geometry._tube_geometry import element_length, tube_inner_radius, tube_outer_area_element


def test_full_one_element_heat_exchanger_against_hand_calculation() -> None:
    """Full 1-element test for the minimal physics implementation.

    This checks the whole path:

        geometry equations -> U_outer -> thermal_conductance_element
        heat_capacity_rate -> tube_outer_element_temperatures
        march_single_tube

    against an independent hand-calculated closed-form answer.

    Geometry:
        tube_length = 2.0 m
        tube_outer_radius = 0.01 m
        tube_wall_thickness = 0.001 m
        wall_thermal_conductivity = 16 W/m/K

    Heat transfer coefficients:
        h_inner = 1000 W/m2/K
        h_outer = 100 W/m2/K

    Flow:
        tube side:
            mdot = 1.0 kg/s
            cp = 200 J/kg/K
            C_tube = 200 W/K

        outer side:
            mdot = 1.0 kg/s
            cp = 50 J/kg/K
            C_outer = 50 W/K

    Temperatures:
        T_tube_in = 350 K
        T_outer_in = 300 K

    Expected equations:

        r_i = r_o - t
        L_e = L / n
        A_o,e = 2 pi r_o L_e

        1 / U_o =
            1 / h_outer
            + r_o ln(r_o / r_i) / k_wall
            + r_o / (r_i h_inner)

        UA_e = U_o A_o,e

        eps = 1 - exp(-UA_e / C_outer)

        q = eps C_outer (T_tube_in - T_outer_in)
            / (1 + eps C_outer / (2 C_tube))

        T_tube_out = T_tube_in - q / C_tube
        T_outer_out = T_outer_in + q / C_outer
    """
    tube_length = 2.0
    tube_outer_radius = 0.01
    tube_wall_thickness = 0.001
    wall_thermal_conductivity = 16.0
    n_elements = 1

    h_inner = 1000.0
    h_outer = 100.0

    mdot_tube = 1.0
    cp_tube = 200.0

    mdot_outer = 1.0
    cp_outer = 50.0

    temp_tube_in = 350.0
    temp_outer_in = 300.0

    # Full code path through the minimal physics functions.
    tube_inner_radius_value = tube_inner_radius(
        tube_outer_radius=tube_outer_radius,
        tube_wall_thickness=tube_wall_thickness,
    )

    element_length_value = element_length(
        tube_length=tube_length,
        n_elements=n_elements,
    )

    tube_outer_area_element_value = tube_outer_area_element(
        tube_outer_radius=tube_outer_radius,
        element_length=element_length_value,
    )

    overall_u_outer_basis_value = overall_u_outer_basis_from_h(
        h_inner=h_inner,
        h_outer=h_outer,
        tube_outer_radius=tube_outer_radius,
        tube_inner_radius=tube_inner_radius_value,
        wall_thermal_conductivity=wall_thermal_conductivity,
    )

    thermal_conductance_element_value = thermal_conductance_element(
        overall_u_outer_basis=overall_u_outer_basis_value,
        tube_outer_area_element=tube_outer_area_element_value,
    )

    heat_capacity_rate_tube = heat_capacity_rate(
        mdot=mdot_tube,
        specific_heat_capacity=cp_tube,
    )

    heat_capacity_rate_outer_element = heat_capacity_rate(
        mdot=mdot_outer,
        specific_heat_capacity=cp_outer,
    )

    result = tube_outer_element_temperatures(
        temp_tube_in=temp_tube_in,
        temp_outer_in=temp_outer_in,
        heat_capacity_rate_tube=heat_capacity_rate_tube,
        heat_capacity_rate_outer_element=heat_capacity_rate_outer_element,
        thermal_conductance_element=thermal_conductance_element_value,
    )

    march_result = march_single_tube(
        temp_tube_in=temp_tube_in,
        temp_outer_in_elements=[temp_outer_in],
        heat_capacity_rate_tube=heat_capacity_rate_tube,
        heat_capacity_rate_outer_elements=[heat_capacity_rate_outer_element],
        thermal_conductance_elements=[thermal_conductance_element_value],
    )

    # Independent hand calculation.
    r_o = tube_outer_radius
    r_i = tube_outer_radius - tube_wall_thickness
    l_e = tube_length / n_elements

    expected_tube_outer_area_element = 2.0 * pi * r_o * l_e

    expected_resistance_total = 1.0 / h_outer + r_o * log(r_o / r_i) / wall_thermal_conductivity + r_o / (r_i * h_inner)

    expected_overall_u_outer_basis = 1.0 / expected_resistance_total
    expected_thermal_conductance_element = expected_overall_u_outer_basis * expected_tube_outer_area_element

    expected_local_effectiveness = 1.0 - exp(-expected_thermal_conductance_element / heat_capacity_rate_outer_element)

    expected_q = (
        expected_local_effectiveness
        * heat_capacity_rate_outer_element
        * (temp_tube_in - temp_outer_in)
        / (1.0 + expected_local_effectiveness * heat_capacity_rate_outer_element / (2.0 * heat_capacity_rate_tube))
    )

    expected_temp_tube_out = temp_tube_in - expected_q / heat_capacity_rate_tube
    expected_temp_outer_out = temp_outer_in + expected_q / heat_capacity_rate_outer_element

    assert tube_inner_radius_value == pytest.approx(0.009)
    assert tube_outer_area_element_value == pytest.approx(expected_tube_outer_area_element)
    assert overall_u_outer_basis_value == pytest.approx(expected_overall_u_outer_basis)
    assert thermal_conductance_element_value == pytest.approx(expected_thermal_conductance_element)

    assert result.q_tube_to_outer == pytest.approx(expected_q)
    assert result.T_tube_out == pytest.approx(expected_temp_tube_out)
    assert result.T_outer_out == pytest.approx(expected_temp_outer_out)

    assert march_result.T_tube_out == pytest.approx(expected_temp_tube_out)
    assert march_result.q_total_tube_to_outer == pytest.approx(expected_q)
    assert march_result.T_outer_out_values[0] == pytest.approx(expected_temp_outer_out)

    # Numerical values for regression clarity.
    assert overall_u_outer_basis_value == pytest.approx(89.46975490243314)
    assert thermal_conductance_element_value == pytest.approx(11.243100988798535)
    assert expected_local_effectiveness == pytest.approx(0.20137359419222556)
    assert result.q_tube_to_outer == pytest.approx(491.0728474430857)
    assert result.T_tube_out == pytest.approx(347.5446357627846)
    assert result.T_outer_out == pytest.approx(309.82145694886174)
