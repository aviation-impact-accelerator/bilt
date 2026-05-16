import typing

import bilt
from bilt.demo.turbomachinery.component.interface import Combustor, Compressor, WorkProducing


@bilt.core.transform
def net_specific_work(
    *components: WorkProducing,
) -> typing.Annotated[float, bilt.unit.joule / bilt.unit.kilogram]:
    return sum(component.specific_work for component in components)


@bilt.core.equation
def thermal_efficiency_relation(
    net_specific_work: typing.Annotated[float, bilt.unit.joule / bilt.unit.kilogram],
    thermal_efficiency: typing.Annotated[float, bilt.unit.fraction],
    air_mass_flow_rate: typing.Annotated[float, bilt.unit.kilogram / bilt.unit.second],
    combustor: Combustor,
) -> bilt.core.equation:
    return thermal_efficiency == net_specific_work * air_mass_flow_rate / combustor.heat_addition


@bilt.core.equation
def pressure_ratio_relation(
    compressor: Compressor,
    pressure_ratio: typing.Annotated[float, bilt.unit.DIMENSIONLESS],
) -> bilt.core.equation:
    return pressure_ratio == compressor.outlet.pressure / compressor.inlet.pressure
