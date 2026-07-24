import bilt

# Scalar quantities intentionally use plain ``float`` until units participate
# in structural analysis and backend lowering.


@bilt.core.system
class FluidProperties(metaclass=bilt.core.Interface):
    specific_heat_capacity: float
    heat_capacity_ratio: float


@bilt.core.system
class FuelProperties(metaclass=bilt.core.Interface):
    lower_heating_value: float
    combustion_efficiency: float


@bilt.core.system
class ThermodynamicState(metaclass=bilt.core.Interface):
    temperature: float
    pressure: float


@bilt.core.system
class FuelFlow(metaclass=bilt.core.Interface):
    fuel: FuelProperties
    mass_flow_rate: float
    heat_release: float


@bilt.core.system
class Compressor(metaclass=bilt.core.Interface):
    fluid: FluidProperties
    inlet: ThermodynamicState
    outlet: ThermodynamicState
    isentropic_efficiency: float
    specific_work: float


@bilt.core.system
class Combustor(metaclass=bilt.core.Interface):
    fluid: FluidProperties
    fuel: FuelProperties
    inlet: ThermodynamicState
    outlet: ThermodynamicState
    air_mass_flow_rate: float
    fuel_mass_flow_rate: float
    pressure_recovery_factor: float
    heat_addition: float


@bilt.core.system
class Turbine(metaclass=bilt.core.Interface):
    fluid: FluidProperties
    inlet: ThermodynamicState
    outlet: ThermodynamicState
    isentropic_efficiency: float
    specific_work: float


@bilt.core.system
class BraytonCycle(metaclass=bilt.core.Interface):
    # Operating inputs.
    fluid: FluidProperties
    fuel: FuelProperties
    inlet: ThermodynamicState
    air_mass_flow_rate: float
    fuel_mass_flow_rate: float
    pressure_ratio: float

    # Cycle outputs.
    net_specific_work: float
    shaft_power: float
    thermal_efficiency: float
    exhaust_temperature: float
    exhaust_pressure: float
    waste_heat: float
