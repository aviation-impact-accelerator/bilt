import math

def reynolds_number(density: float, velocity: float, length: float, mu: float) -> float:
    return (density * velocity * length) / mu

def prandtl_number(momentum_diffusivity: float, kinematic_viscosity: float) -> float:
    return momentum_diffusivity / kinematic_viscosity

def biot_number(htc: float, thermal_conductivity: float, length: float) -> float:
    return htc * thermal_conductivity / length

def nusselt_number(htc: float, thermal_conductivity: float, length: float) -> float:
    return htc * thermal_conductivity / length

def nusselt_dittus_boelter(Reynolds_number: float, Prandtl_number: float, heat: bool) -> float:
    if heat:
        n = 0.4
    else:
        n = 0.3
    return 0.023 * Reynolds_number ** (0.8) * Prandtl_number ** n

def effectiveness(overall_htc: float, area: float, C_min: float) -> float:
    return 1-math.exp(-overall_htc * area / C_min)

