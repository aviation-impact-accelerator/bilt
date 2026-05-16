import typing

import bilt
from bilt.demo.fluid.interface import FuelFlow, FuelProperties


@bilt.core.equation
def heat_release_relation(
    fuel: FuelProperties,
    mass_flow_rate: typing.Annotated[float, bilt.unit.kilogram / bilt.unit.second],
    heat_release: typing.Annotated[float, bilt.unit.joule / bilt.unit.second],
) -> bilt.core.equation:
    return heat_release == fuel.combustion_efficiency * fuel.lower_heating_value * mass_flow_rate


simple_fuel_flow = bilt.Implementation(
    FuelFlow,
    {heat_release_relation},
)
