import typing

import bilt.core
from bilt.demo.turbomachinery.component.interface import Combustor, Compressor, WorkProducing
from bilt.units import DIMENSIONLESS, fraction, joule, kilogram, second


@bilt.core.transform
def net_specific_work(
    *components: WorkProducing,
) -> typing.Annotated[float, joule / kilogram]:
    return sum(component.specific_work for component in components)


@bilt.core.equation
def thermal_efficiency_relation(
    net_specific_work: typing.Annotated[float, joule / kilogram],
    thermal_efficiency: typing.Annotated[float, fraction],
    air_mass_flow_rate: typing.Annotated[float, kilogram / second],
    combustor: Combustor,
) -> bool:
    return thermal_efficiency == net_specific_work * air_mass_flow_rate / combustor.heat_addition


@bilt.core.equation
def pressure_ratio_relation(
    compressor: Compressor,
    pressure_ratio: typing.Annotated[float, DIMENSIONLESS],
) -> bool:
    return pressure_ratio == compressor.outlet.pressure / compressor.inlet.pressure
