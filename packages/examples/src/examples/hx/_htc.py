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
    mdot: float,
    density: float,
    dynamic_viscosity: float,
    tube_inner_diameter: float,
    tube_flow_area: float,
) -> float:
    velocity_value = velocity(
        mdot=mdot,
        density=density,
        tube_flow_area=tube_flow_area,
    )

    return reynolds_number(
        density=density,
        velocity=velocity_value,
        length=tube_inner_diameter,
        mu=dynamic_viscosity,
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
    mdot: float,
    density: float,
    specific_heat_capacity: float,
    dynamic_viscosity: float,
    thermal_conductivity: float,
    tube_inner_diameter: float,
    tube_flow_area: float,
) -> float:
    reynolds_number_value = tube_side_reynolds_number(
        mdot=mdot,
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


def tube_side_heat_transfer_coefficient_cooling(  # noqa : PLR0913
    mdot: float,
    density: float,
    specific_heat_capacity: float,
    dynamic_viscosity: float,
    thermal_conductivity: float,
    tube_inner_diameter: float,
    tube_flow_area: float,
) -> float:
    reynolds_number_value = tube_side_reynolds_number(
        mdot=mdot,
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


def resistance_outer_convection(
    h_outer: float,
) -> float:
    return 1.0 / h_outer


def resistance_wall_outer_basis(
    tube_outer_radius: float,
    tube_inner_radius: float,
    wall_thermal_conductivity: float,
) -> float:
    return tube_outer_radius * log(tube_outer_radius / tube_inner_radius) / wall_thermal_conductivity


def resistance_inner_convection_outer_basis(
    tube_outer_radius: float,
    tube_inner_radius: float,
    h_inner: float,
) -> float:
    return tube_outer_radius / (tube_inner_radius * h_inner)


def resistance_total_outer_basis(
    resistance_outer_convection: float,
    resistance_wall_outer_basis: float,
    resistance_inner_convection_outer_basis: float,
) -> float:
    return resistance_outer_convection + resistance_wall_outer_basis + resistance_inner_convection_outer_basis


def overall_u_outer_basis(
    resistance_total_outer_basis: float,
) -> float:
    return 1.0 / resistance_total_outer_basis


def overall_u_outer_basis_from_h(
    h_inner: float,
    h_outer: float,
    tube_outer_radius: float,
    tube_inner_radius: float,
    wall_thermal_conductivity: float,
) -> float:
    resistance_outer_convection_value = resistance_outer_convection(
        h_outer=h_outer,
    )

    resistance_wall_outer_basis_value = resistance_wall_outer_basis(
        tube_outer_radius=tube_outer_radius,
        tube_inner_radius=tube_inner_radius,
        wall_thermal_conductivity=wall_thermal_conductivity,
    )

    resistance_inner_convection_outer_basis_value = resistance_inner_convection_outer_basis(
        tube_outer_radius=tube_outer_radius,
        tube_inner_radius=tube_inner_radius,
        h_inner=h_inner,
    )

    resistance_total_outer_basis_value = resistance_total_outer_basis(
        resistance_outer_convection=resistance_outer_convection_value,
        resistance_wall_outer_basis=resistance_wall_outer_basis_value,
        resistance_inner_convection_outer_basis=resistance_inner_convection_outer_basis_value,
    )

    return overall_u_outer_basis(
        resistance_total_outer_basis=resistance_total_outer_basis_value,
    )


def thermal_conductance_element(
    overall_u_outer_basis: float,
    tube_outer_area_element: float,
) -> float:
    return overall_u_outer_basis * tube_outer_area_element
