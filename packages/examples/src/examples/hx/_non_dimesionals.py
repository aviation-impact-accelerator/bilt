import math


def reynolds_number(density: float, velocity: float, length: float, mu: float) -> float:
    return (density * velocity * length) / mu


def prandtl_number(momentum_diffusivity: float, kinematic_viscosity: float) -> float:
    return momentum_diffusivity / kinematic_viscosity


def biot_number(htc: float, thermal_conductivity: float, length: float) -> float:
    return htc * thermal_conductivity / length


def nusselt_number(htc: float, thermal_conductivity: float, length: float) -> float:
    return htc * thermal_conductivity / length


def nusselt_dittus_boelter_heating(reynolds_number: float, prandtl_number: float) -> float:
    return 0.023 * reynolds_number ** (0.8) * prandtl_number**0.4

def nusselt_dittus_boelter_cooling(reynolds_number: float, prandtl_number: float) -> float:
    return 0.023 * reynolds_number ** (0.8) * prandtl_number**0.3


def effectiveness(overall_htc: float, area: float, min_heat_capacity: float) -> float:
    return 1 - math.exp(-overall_htc * area / min_heat_capacity)
