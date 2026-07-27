import bilt

from examples.brayton_model.combustor import simple_combustor
from examples.brayton_model.compressor import isentropic_compressor
from examples.brayton_model.cycle import (
    air_mass_flow,
    combustion_efficiency,
    combustor_inlet_pressure,
    combustor_inlet_temperature,
    exhaust_pressure,
    exhaust_temperature,
    fuel_mass_flow,
    heat_capacity_ratio,
    inlet_pressure,
    inlet_temperature,
    lower_heating_value,
    net_specific_work,
    shaft_power,
    specific_heat_capacity,
    thermal_efficiency,
    turbine_inlet_pressure,
    turbine_inlet_temperature,
    turbine_outlet_pressure,
    waste_heat,
    compressor_outlet_pressure,
)
from examples.brayton_model.interfaces import (
    Combustor,
    Compressor,
    FluidProperties,
    FuelProperties,
    ThermodynamicState,
    Turbine,
)
from examples.brayton_model.turbine import isentropic_turbine


@bilt.core.system
class BraytonAssembly(metaclass=bilt.core.Interface):
    """Brayton cycle whose component structure is part of its public API."""

    # Operating inputs.
    fluid: FluidProperties
    fuel: FuelProperties
    inlet: ThermodynamicState
    air_mass_flow_rate: float
    fuel_mass_flow_rate: float
    pressure_ratio: float

    # Publicly visible component instances.
    compressor: Compressor
    combustor: Combustor
    turbine: Turbine

    # Cycle outputs.
    net_specific_work: float
    shaft_power: float
    thermal_efficiency: float
    exhaust_temperature: float
    exhaust_pressure: float
    waste_heat: float


def public_brayton_cycle(
    *,
    compressor_efficiency: float,
    combustor_pressure_recovery: float,
    turbine_efficiency: float,
) -> bilt.core.Implementation:
    """Compose the same cycle with its components exposed as public children."""

    compressor = isentropic_compressor(
        efficiency=compressor_efficiency,
    )
    combustor = simple_combustor(
        pressure_recovery_factor=combustor_pressure_recovery,
    )
    turbine = isentropic_turbine(
        efficiency=turbine_efficiency,
    )

    return bilt.core.Implementation(
        BraytonAssembly,
        {
            # Cycle fluid -> component fluids.
            "compressor.fluid.specific_heat_capacity":
                specific_heat_capacity,
            "compressor.fluid.heat_capacity_ratio":
                heat_capacity_ratio,
            "combustor.fluid.specific_heat_capacity":
                specific_heat_capacity,
            "combustor.fluid.heat_capacity_ratio":
                heat_capacity_ratio,
            "turbine.fluid.specific_heat_capacity":
                specific_heat_capacity,
            "turbine.fluid.heat_capacity_ratio":
                heat_capacity_ratio,

            # Cycle fuel -> combustor fuel.
            "combustor.fuel.lower_heating_value":
                lower_heating_value,
            "combustor.fuel.combustion_efficiency":
                combustion_efficiency,

            # Cycle inlet -> compressor inlet.
            "compressor.inlet.temperature":
                inlet_temperature,
            "compressor.inlet.pressure":
                inlet_pressure,

            # Compressor pressure boundary.
            "compressor.outlet.pressure":
                compressor_outlet_pressure,

            # Compressor -> combustor.
            "combustor.inlet.temperature":
                combustor_inlet_temperature,
            "combustor.inlet.pressure":
                combustor_inlet_pressure,
            "combustor.air_mass_flow_rate":
                air_mass_flow,
            "combustor.fuel_mass_flow_rate":
                fuel_mass_flow,

            # Combustor -> turbine.
            "turbine.inlet.temperature":
                turbine_inlet_temperature,
            "turbine.inlet.pressure":
                turbine_inlet_pressure,

            # Expansion boundary.
            "turbine.outlet.pressure":
                turbine_outlet_pressure,

            # Public cycle outputs.
            "net_specific_work":
                net_specific_work,
            "shaft_power":
                shaft_power,
            "thermal_efficiency":
                thermal_efficiency,
            "exhaust_temperature":
                exhaust_temperature,
            "exhaust_pressure":
                exhaust_pressure,
            "waste_heat":
                waste_heat,
        },
        children={
            "compressor": compressor,
            "combustor": combustor,
            "turbine": turbine,
        },
    )
