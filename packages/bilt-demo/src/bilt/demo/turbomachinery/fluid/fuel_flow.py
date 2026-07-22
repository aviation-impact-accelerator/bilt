import typing

import bilt.core
from bilt.demo.turbomachinery.fluid.interface import FuelFlow, FuelProperties
from bilt.units import joule, kilogram, second


@bilt.core.equation
def heat_release_relation(
    fuel: FuelProperties,
    mass_flow_rate: typing.Annotated[float, kilogram / second],
    heat_release: typing.Annotated[float, joule / second],
) -> bool:
    return heat_release == fuel.combustion_efficiency * fuel.lower_heating_value * mass_flow_rate


simple_fuel_flow = bilt.core.Implementation(
    FuelFlow,
    {heat_release_relation},
)
