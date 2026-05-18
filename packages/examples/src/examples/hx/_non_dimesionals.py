import math

DITTUS_BOELTER_HEATING_EXPONENT = 0.4
DITTUS_BOELTER_COOLING_EXPONENT = 0.3


def reynolds_number(
    density: float,
    velocity: float,
    length: float,
    dynamic_viscosity: float,
) -> float:
    return density * velocity * length / dynamic_viscosity


def prandtl_number_diffusivities(
    momentum_diffusivity: float,
    thermal_diffusivity: float,
) -> float:
    return momentum_diffusivity / thermal_diffusivity


def prandtl_number_properties(
    specific_heat_capacity: float,
    dynamic_viscosity: float,
    thermal_conductivity: float,
) -> float:
    return specific_heat_capacity * dynamic_viscosity / thermal_conductivity


def biot_number(
    heat_transfer_coefficient: float,
    thermal_conductivity: float,
    length: float,
) -> float:
    return heat_transfer_coefficient * length / thermal_conductivity


def nusselt_number(
    heat_transfer_coefficient: float,
    thermal_conductivity: float,
    length: float,
) -> float:
    return heat_transfer_coefficient * length / thermal_conductivity


def _nusselt_dittus_boelter(
    reynolds_number: float,
    prandtl_number: float,
    prandtl_number_exponent: float,
) -> float:
    return 0.023 * reynolds_number**0.8 * prandtl_number**prandtl_number_exponent


def nusselt_dittus_boelter_heating(
    reynolds_number: float,
    prandtl_number: float,
) -> float:
    return _nusselt_dittus_boelter(
        reynolds_number=reynolds_number,
        prandtl_number=prandtl_number,
        prandtl_number_exponent=DITTUS_BOELTER_HEATING_EXPONENT,
    )


def nusselt_dittus_boelter_cooling(
    reynolds_number: float,
    prandtl_number: float,
) -> float:
    return _nusselt_dittus_boelter(
        reynolds_number=reynolds_number,
        prandtl_number=prandtl_number,
        prandtl_number_exponent=DITTUS_BOELTER_COOLING_EXPONENT,
    )


def effectiveness(
    overall_heat_transfer_coefficient: float,
    area: float,
    minimum_heat_capacity_rate: float,
) -> float:
    return 1.0 - math.exp(
        -overall_heat_transfer_coefficient
        * area
        / minimum_heat_capacity_rate
    )