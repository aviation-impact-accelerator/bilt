from math import exp, log, pi

import pytest
from examples.hx._htc import (
    overall_heat_transfer_coefficient_outer_area_basis_from_coefficients,
    thermal_conductance_per_element,
)
from examples.hx._physics import (
    heat_capacity_rate,
    tube_outer_element_temperatures,
)
from examples.hx._solver import march_single_tube
from examples.hx.geometry._tube_geometry import (
    element_length,
    tube_inner_radius,
    tube_outer_area_element,
)


def test_full_one_element_heat_exchanger_against_hand_calculation() -> None:
    """Full one-element test for the minimal physics implementation.

    This checks the whole path:

        geometry equations
        -> overall heat transfer coefficient on outer-area basis
        -> thermal conductance per element
        -> heat capacity rate
        -> tube/outer element temperatures
        -> single-tube marching solver

    against an independent hand-calculated closed-form answer.
    """
    tube_length = 2.0
    tube_outer_radius = 0.01
    tube_wall_thickness = 0.001
    wall_thermal_conductivity = 16.0
    number_of_elements = 1

    inner_heat_transfer_coefficient = 1000.0
    outer_heat_transfer_coefficient = 100.0

    tube_mass_flow_rate = 1.0
    tube_specific_heat_capacity = 200.0

    outer_mass_flow_rate = 1.0
    outer_specific_heat_capacity = 50.0

    tube_inlet_temperature = 350.0
    outer_inlet_temperature = 300.0

    # Full code path through the minimal physics functions.
    tube_inner_radius_value = tube_inner_radius(
        tube_outer_radius=tube_outer_radius,
        tube_wall_thickness=tube_wall_thickness,
    )

    element_length_value = element_length(
        tube_length=tube_length,
        n_elements=number_of_elements,
    )

    tube_outer_area_per_element = tube_outer_area_element(
        tube_outer_radius=tube_outer_radius,
        element_length=element_length_value,
    )

    overall_heat_transfer_coefficient_outer_area_basis = (
        overall_heat_transfer_coefficient_outer_area_basis_from_coefficients(
            inner_heat_transfer_coefficient=inner_heat_transfer_coefficient,
            outer_heat_transfer_coefficient=outer_heat_transfer_coefficient,
            tube_outer_radius=tube_outer_radius,
            tube_inner_radius=tube_inner_radius_value,
            wall_thermal_conductivity=wall_thermal_conductivity,
        )
    )

    thermal_conductance_per_element_value = thermal_conductance_per_element(
        overall_heat_transfer_coefficient_outer_area_basis=(overall_heat_transfer_coefficient_outer_area_basis),
        tube_outer_area_per_element=tube_outer_area_per_element,
    )

    tube_heat_capacity_rate = heat_capacity_rate(
        mass_flow_rate=tube_mass_flow_rate,
        specific_heat_capacity=tube_specific_heat_capacity,
    )

    outer_heat_capacity_rate_per_element = heat_capacity_rate(
        mass_flow_rate=outer_mass_flow_rate,
        specific_heat_capacity=outer_specific_heat_capacity,
    )

    result = tube_outer_element_temperatures(
        tube_inlet_temperature=tube_inlet_temperature,
        outer_inlet_temperature=outer_inlet_temperature,
        tube_heat_capacity_rate=tube_heat_capacity_rate,
        outer_heat_capacity_rate_per_element=outer_heat_capacity_rate_per_element,
        thermal_conductance_per_element=thermal_conductance_per_element_value,
    )

    march_result = march_single_tube(
        tube_inlet_temperature=tube_inlet_temperature,
        outer_inlet_temperature_values=[outer_inlet_temperature],
        tube_heat_capacity_rate=tube_heat_capacity_rate,
        outer_heat_capacity_rate_values=[outer_heat_capacity_rate_per_element],
        thermal_conductance_values=[thermal_conductance_per_element_value],
    )

    # Independent hand calculation.
    outer_radius = tube_outer_radius
    inner_radius = tube_outer_radius - tube_wall_thickness
    length_per_element = tube_length / number_of_elements

    expected_tube_outer_area_per_element = 2.0 * pi * outer_radius * length_per_element

    expected_total_thermal_resistance = (
        1.0 / outer_heat_transfer_coefficient
        + outer_radius * log(outer_radius / inner_radius) / wall_thermal_conductivity
        + outer_radius / (inner_radius * inner_heat_transfer_coefficient)
    )

    expected_overall_heat_transfer_coefficient_outer_area_basis = 1.0 / expected_total_thermal_resistance

    expected_thermal_conductance_per_element = (
        expected_overall_heat_transfer_coefficient_outer_area_basis * expected_tube_outer_area_per_element
    )

    expected_effectiveness = 1.0 - exp(-expected_thermal_conductance_per_element / outer_heat_capacity_rate_per_element)

    expected_heat_transfer_rate = (
        expected_effectiveness
        * outer_heat_capacity_rate_per_element
        * (tube_inlet_temperature - outer_inlet_temperature)
        / (1.0 + expected_effectiveness * outer_heat_capacity_rate_per_element / (2.0 * tube_heat_capacity_rate))
    )

    expected_tube_outlet_temperature = tube_inlet_temperature - expected_heat_transfer_rate / tube_heat_capacity_rate

    expected_outer_outlet_temperature = (
        outer_inlet_temperature + expected_heat_transfer_rate / outer_heat_capacity_rate_per_element
    )

    assert tube_inner_radius_value == pytest.approx(0.009)

    assert tube_outer_area_per_element == pytest.approx(expected_tube_outer_area_per_element)

    assert overall_heat_transfer_coefficient_outer_area_basis == pytest.approx(
        expected_overall_heat_transfer_coefficient_outer_area_basis
    )

    assert thermal_conductance_per_element_value == pytest.approx(expected_thermal_conductance_per_element)

    assert result.heat_transfer_rate_tube_to_outer == pytest.approx(expected_heat_transfer_rate)
    assert result.tube_outlet_temperature == pytest.approx(expected_tube_outlet_temperature)
    assert result.outer_outlet_temperature == pytest.approx(expected_outer_outlet_temperature)

    assert march_result.tube_outlet_temperature == pytest.approx(expected_tube_outlet_temperature)
    assert march_result.total_heat_transfer_rate_tube_to_outer == pytest.approx(expected_heat_transfer_rate)
    assert march_result.outer_outlet_temperature_values[0] == pytest.approx(expected_outer_outlet_temperature)

    # Numerical values for regression clarity.
    assert overall_heat_transfer_coefficient_outer_area_basis == pytest.approx(89.46975490243314)
    assert thermal_conductance_per_element_value == pytest.approx(11.243100988798535)
    assert expected_effectiveness == pytest.approx(0.20137359419222556)
    assert result.heat_transfer_rate_tube_to_outer == pytest.approx(491.0728474430857)
    assert result.tube_outlet_temperature == pytest.approx(347.5446357627846)
    assert result.outer_outlet_temperature == pytest.approx(309.82145694886174)
