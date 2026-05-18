from math import log

from examples.hx._non_dimesionals import (
    nusselt_dittus_boelter_cooling,
    nusselt_dittus_boelter_heating,
    prandtl_number_properties,
    reynolds_number,
)
from examples.hx._physics import velocity


def heat_transfer_coefficient(
    nusselt_number: float,
    thermal_conductivity: float,
    tube_inner_diameter: float,
) -> float:
    return nusselt_number * thermal_conductivity / tube_inner_diameter


def tube_side_reynolds_number(
    mass_flow_rate: float,
    density: float,
    dynamic_viscosity: float,
    tube_inner_diameter: float,
    tube_flow_area: float,
) -> float:
    tube_velocity = velocity(
        mass_flow_rate=mass_flow_rate,
        density=density,
        flow_area=tube_flow_area,
    )

    return reynolds_number(
        density=density,
        velocity=tube_velocity,
        length=tube_inner_diameter,
        dynamic_viscosity=dynamic_viscosity,
    )


def tube_side_prandtl_number(
    specific_heat_capacity: float,
    dynamic_viscosity: float,
    thermal_conductivity: float,
) -> float:
    return prandtl_number_properties(
        specific_heat_capacity=specific_heat_capacity,
        dynamic_viscosity=dynamic_viscosity,
        thermal_conductivity=thermal_conductivity,
    )


def heat_transfer_coefficient_dittus_boelter_heating(
    reynolds_number: float,
    prandtl_number: float,
    thermal_conductivity: float,
    tube_inner_diameter: float,
) -> float:
    nusselt_number = nusselt_dittus_boelter_heating(
        reynolds_number=reynolds_number,
        prandtl_number=prandtl_number,
    )

    return heat_transfer_coefficient(
        nusselt_number=nusselt_number,
        thermal_conductivity=thermal_conductivity,
        tube_inner_diameter=tube_inner_diameter,
    )


def heat_transfer_coefficient_dittus_boelter_cooling(
    reynolds_number: float,
    prandtl_number: float,
    thermal_conductivity: float,
    tube_inner_diameter: float,
) -> float:
    nusselt_number = nusselt_dittus_boelter_cooling(
        reynolds_number=reynolds_number,
        prandtl_number=prandtl_number,
    )

    return heat_transfer_coefficient(
        nusselt_number=nusselt_number,
        thermal_conductivity=thermal_conductivity,
        tube_inner_diameter=tube_inner_diameter,
    )


def tube_side_heat_transfer_coefficient_heating(  # noqa: PLR0913
    mass_flow_rate: float,
    density: float,
    specific_heat_capacity: float,
    dynamic_viscosity: float,
    thermal_conductivity: float,
    tube_inner_diameter: float,
    tube_flow_area: float,
) -> float:
    reynolds_number_value = tube_side_reynolds_number(
        mass_flow_rate=mass_flow_rate,
        density=density,
        dynamic_viscosity=dynamic_viscosity,
        tube_inner_diameter=tube_inner_diameter,
        tube_flow_area=tube_flow_area,
    )

    prandtl_number_value = tube_side_prandtl_number(
        specific_heat_capacity=specific_heat_capacity,
        dynamic_viscosity=dynamic_viscosity,
        thermal_conductivity=thermal_conductivity,
    )

    return heat_transfer_coefficient_dittus_boelter_heating(
        reynolds_number=reynolds_number_value,
        prandtl_number=prandtl_number_value,
        thermal_conductivity=thermal_conductivity,
        tube_inner_diameter=tube_inner_diameter,
    )


def tube_side_heat_transfer_coefficient_cooling(  # noqa: PLR0913
    mass_flow_rate: float,
    density: float,
    specific_heat_capacity: float,
    dynamic_viscosity: float,
    thermal_conductivity: float,
    tube_inner_diameter: float,
    tube_flow_area: float,
) -> float:
    reynolds_number_value = tube_side_reynolds_number(
        mass_flow_rate=mass_flow_rate,
        density=density,
        dynamic_viscosity=dynamic_viscosity,
        tube_inner_diameter=tube_inner_diameter,
        tube_flow_area=tube_flow_area,
    )

    prandtl_number_value = tube_side_prandtl_number(
        specific_heat_capacity=specific_heat_capacity,
        dynamic_viscosity=dynamic_viscosity,
        thermal_conductivity=thermal_conductivity,
    )

    return heat_transfer_coefficient_dittus_boelter_cooling(
        reynolds_number=reynolds_number_value,
        prandtl_number=prandtl_number_value,
        thermal_conductivity=thermal_conductivity,
        tube_inner_diameter=tube_inner_diameter,
    )


def outer_convection_thermal_resistance(
    outer_heat_transfer_coefficient: float,
) -> float:
    return 1.0 / outer_heat_transfer_coefficient


def wall_thermal_resistance_outer_area_basis(
    tube_outer_radius: float,
    tube_inner_radius: float,
    wall_thermal_conductivity: float,
) -> float:
    return tube_outer_radius * log(tube_outer_radius / tube_inner_radius) / wall_thermal_conductivity


def inner_convection_thermal_resistance_outer_area_basis(
    tube_outer_radius: float,
    tube_inner_radius: float,
    inner_heat_transfer_coefficient: float,
) -> float:
    return tube_outer_radius / (tube_inner_radius * inner_heat_transfer_coefficient)


def total_thermal_resistance_outer_area_basis(
    outer_convection_thermal_resistance: float,
    wall_thermal_resistance_outer_area_basis: float,
    inner_convection_thermal_resistance_outer_area_basis: float,
) -> float:
    return (
        outer_convection_thermal_resistance
        + wall_thermal_resistance_outer_area_basis
        + inner_convection_thermal_resistance_outer_area_basis
    )


def overall_heat_transfer_coefficient_outer_area_basis(
    total_thermal_resistance_outer_area_basis: float,
) -> float:
    return 1.0 / total_thermal_resistance_outer_area_basis


def overall_heat_transfer_coefficient_outer_area_basis_from_coefficients(
    inner_heat_transfer_coefficient: float,
    outer_heat_transfer_coefficient: float,
    tube_outer_radius: float,
    tube_inner_radius: float,
    wall_thermal_conductivity: float,
) -> float:
    outer_convection_resistance = outer_convection_thermal_resistance(
        outer_heat_transfer_coefficient=outer_heat_transfer_coefficient,
    )

    wall_resistance = wall_thermal_resistance_outer_area_basis(
        tube_outer_radius=tube_outer_radius,
        tube_inner_radius=tube_inner_radius,
        wall_thermal_conductivity=wall_thermal_conductivity,
    )

    inner_convection_resistance = inner_convection_thermal_resistance_outer_area_basis(
        tube_outer_radius=tube_outer_radius,
        tube_inner_radius=tube_inner_radius,
        inner_heat_transfer_coefficient=inner_heat_transfer_coefficient,
    )

    total_resistance = total_thermal_resistance_outer_area_basis(
        outer_convection_thermal_resistance=outer_convection_resistance,
        wall_thermal_resistance_outer_area_basis=wall_resistance,
        inner_convection_thermal_resistance_outer_area_basis=inner_convection_resistance,
    )

    return overall_heat_transfer_coefficient_outer_area_basis(
        total_thermal_resistance_outer_area_basis=total_resistance,
    )


def thermal_conductance_per_element(
    overall_heat_transfer_coefficient_outer_area_basis: float,
    tube_outer_area_per_element: float,
) -> float:
    return overall_heat_transfer_coefficient_outer_area_basis * tube_outer_area_per_element
