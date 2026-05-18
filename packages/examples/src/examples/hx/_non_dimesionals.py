import math

DITTUS_BOELTER_HEATING_EXPONENT = 0.4
DITTUS_BOELTER_COOLING_EXPONENT = 0.3


def reynolds_number(density: float, velocity: float, length: float, mu: float) -> float:
    return (density * velocity * length) / mu


def prandtl_number_diffusivities(momentum_diffusivity: float, kinematic_viscosity: float) -> float:
    return momentum_diffusivity / kinematic_viscosity


def prandtl_number_properties(
    specific_heat_capacity: float, dynamic_viscosity: float, thermal_conductivity: float
) -> float:
    return specific_heat_capacity * dynamic_viscosity / thermal_conductivity


def biot_number(htc: float, thermal_conductivity: float, length: float) -> float:
    return htc * thermal_conductivity / length


def nusselt_number(htc: float, thermal_conductivity: float, length: float) -> float:
    return htc * thermal_conductivity / length


def _nusselt_dittus_boelter(
    reynolds_number: float,
    prandtl_number: float,
    exponent: float,
) -> float:
    return 0.023 * reynolds_number**0.8 * prandtl_number**exponent


def nusselt_dittus_boelter_heating(
    reynolds_number: float,
    prandtl_number: float,
) -> float:
    return _nusselt_dittus_boelter(
        reynolds_number=reynolds_number,
        prandtl_number=prandtl_number,
        exponent=0.4,
    )


def nusselt_dittus_boelter_cooling(
    reynolds_number: float,
    prandtl_number: float,
) -> float:
    return _nusselt_dittus_boelter(
        reynolds_number=reynolds_number,
        prandtl_number=prandtl_number,
        exponent=0.3,
    )


def effectiveness(overall_htc: float, area: float, min_heat_capacity: float) -> float:
    return 1 - math.exp(-overall_htc * area / min_heat_capacity)
