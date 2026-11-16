"""Nested coupled-equation regression with competing subsystem routes.

The algebra is synthetic; the benchmark measures structural search, not a physical process.
"""

# ruff: noqa: D101, D103, INP001 - copied equation fixtures

import typing

from bilt import Implementation, Interface, Quantity, equation, field, system
from bilt.units import fraction
from bilt.units.dimensions import AMOUNT, DIMENSIONLESS, ENERGY, MASS, POWER, TIME

MassFlow = typing.Annotated[Quantity, MASS / TIME]
UnitFlow = typing.Annotated[Quantity, AMOUNT / TIME]
SpecificEnergy = typing.Annotated[Quantity, ENERGY / MASS]
EnergyPerUnit = typing.Annotated[Quantity, ENERGY / AMOUNT]
Power = typing.Annotated[Quantity, POWER]
Fraction = typing.Annotated[Quantity, DIMENSIONLESS]


@system
class InputStage(metaclass=Interface):
    input_flow: MassFlow = field()
    input_specific_energy: SpecificEnergy = field()
    conversion_efficiency: Fraction = field()
    intermediate_power: Power = field()


@equation
def input_stage_balance(
    input_flow: MassFlow,
    input_specific_energy: SpecificEnergy,
    conversion_efficiency: Fraction,
    intermediate_power: Power,
) -> bool:
    return intermediate_power == input_flow * input_specific_energy * conversion_efficiency


@system
class CoupledStage(metaclass=Interface):
    intermediate_power: Power = field()
    energy_per_unit: EnergyPerUnit = field()
    flow_ratio: Fraction = field()
    primary_flow: UnitFlow = field()
    secondary_flow: UnitFlow = field()


@equation
def original_secondary_flow(primary_flow: UnitFlow, secondary_flow: UnitFlow, flow_ratio: Fraction) -> bool:
    return secondary_flow == primary_flow * flow_ratio


@equation
def original_total_energy(
    primary_flow: UnitFlow,
    secondary_flow: UnitFlow,
    energy_per_unit: EnergyPerUnit,
    intermediate_power: Power,
) -> bool:
    return intermediate_power == (primary_flow + secondary_flow) * energy_per_unit


@equation
def explicit_primary_flow(
    primary_flow: UnitFlow,
    intermediate_power: Power,
    energy_per_unit: EnergyPerUnit,
    flow_ratio: Fraction,
) -> bool:
    return primary_flow == intermediate_power / (energy_per_unit * (1.0 * fraction + flow_ratio))


@equation
def explicit_secondary_flow(primary_flow: UnitFlow, secondary_flow: UnitFlow, flow_ratio: Fraction) -> bool:
    return secondary_flow == primary_flow * flow_ratio


@system
class OutputStage(metaclass=Interface):
    intermediate_power: Power = field()
    output_fraction: Fraction = field()
    final_power: Power = field()
    primary_flow: UnitFlow = field()
    secondary_flow: UnitFlow = field()
    flow_ratio: Fraction = field()


@equation
def output_stage_energy(intermediate_power: Power, output_fraction: Fraction, final_power: Power) -> bool:
    return final_power == intermediate_power * output_fraction


@equation
def output_stage_ratio(primary_flow: UnitFlow, secondary_flow: UnitFlow, flow_ratio: Fraction) -> bool:
    return secondary_flow == primary_flow * flow_ratio


@equation
def output_stage_selection(flow_ratio: Fraction, output_fraction: Fraction) -> bool:
    if flow_ratio >= 2.0 * fraction:
        selected = 0.45 * fraction
    elif flow_ratio >= 1.0 * fraction:
        selected = 0.35 * fraction
    else:
        selected = 0.25 * fraction
    return output_fraction == selected


@system
class NestedPipeline(metaclass=Interface):
    input_flow: MassFlow = field()
    input_specific_energy: SpecificEnergy = field()
    input_efficiency: Fraction = field()
    output_fraction_alias: Fraction = field()
    intermediate_power: Power = field()
    final_power: Power = field()
    flow_ratio: Fraction = field()
    energy_per_unit: EnergyPerUnit = field()
    primary_flow: UnitFlow = field()
    secondary_flow: UnitFlow = field()


@equation
def connect_input_flow(input_flow: MassFlow, input_stage: InputStage) -> bool:
    return input_flow == input_stage.input_flow


@equation
def connect_input_specific_energy(input_specific_energy: SpecificEnergy, input_stage: InputStage) -> bool:
    return input_specific_energy == input_stage.input_specific_energy


@equation
def connect_input_efficiency(input_efficiency: Fraction, input_stage: InputStage) -> bool:
    return input_efficiency == input_stage.conversion_efficiency


@equation
def connect_coupled_from_input_stage(intermediate_power: Power, input_stage: InputStage) -> bool:
    return intermediate_power == input_stage.intermediate_power


@equation
def connect_coupled_to_output_stage(intermediate_power: Power, output_stage: OutputStage) -> bool:
    return intermediate_power == output_stage.intermediate_power


@equation
def connect_output_fraction_alias(output_fraction_alias: Fraction, output_stage: OutputStage) -> bool:
    return output_fraction_alias == output_stage.output_fraction


@equation
def connect_final_power(final_power: Power, output_stage: OutputStage) -> bool:
    return final_power == output_stage.final_power


@equation
def connect_power_to_coupled(intermediate_power: Power, coupled_stage: CoupledStage) -> bool:
    return intermediate_power == coupled_stage.intermediate_power


@equation
def connect_assumed_unit_energy(energy_per_unit: EnergyPerUnit, coupled_stage: CoupledStage) -> bool:
    return energy_per_unit == coupled_stage.energy_per_unit


@equation
def connect_assumed_ratio(flow_ratio: Fraction, coupled_stage: CoupledStage) -> bool:
    return flow_ratio == coupled_stage.flow_ratio


@equation
def connect_primary_from_coupled(primary_flow: UnitFlow, coupled_stage: CoupledStage) -> bool:
    return primary_flow == coupled_stage.primary_flow


@equation
def connect_secondary_from_coupled(secondary_flow: UnitFlow, coupled_stage: CoupledStage) -> bool:
    return secondary_flow == coupled_stage.secondary_flow


@equation
def connect_primary_to_output_stage(primary_flow: UnitFlow, output_stage: OutputStage) -> bool:
    return primary_flow == output_stage.primary_flow


@equation
def connect_secondary_to_output_stage(secondary_flow: UnitFlow, output_stage: OutputStage) -> bool:
    return secondary_flow == output_stage.secondary_flow


PARENT_CONNECTIONS = (
    connect_input_flow,
    connect_input_specific_energy,
    connect_input_efficiency,
    connect_coupled_from_input_stage,
    connect_coupled_to_output_stage,
    connect_output_fraction_alias,
    connect_final_power,
    connect_power_to_coupled,
    connect_assumed_unit_energy,
    connect_assumed_ratio,
    connect_primary_from_coupled,
    connect_secondary_from_coupled,
    connect_primary_to_output_stage,
    connect_secondary_to_output_stage,
)


def build_nested_coupled_model(*, coupled: bool) -> Implementation:
    """Keep the nested connections fixed and vary only the coupled equations."""
    input_stage = Implementation(InputStage, [input_stage_balance])
    coupled_stage = Implementation(
        CoupledStage,
        [original_secondary_flow, original_total_energy]
        if coupled
        else [explicit_primary_flow, explicit_secondary_flow],
    )
    output_stage = Implementation(OutputStage, [output_stage_energy, output_stage_ratio, output_stage_selection])
    return Implementation(
        NestedPipeline,
        PARENT_CONNECTIONS,
        internal_implementations={
            "input_stage": input_stage,
            "coupled_stage": coupled_stage,
            "output_stage": output_stage,
        },
    )


# Input rate 2, input energy 18 MJ/kg, efficiency 0.65, output fraction 0.45.
# Independently expected final output is 10.53 MW.
EXPECTED_FINAL_POWER_W = 10.53e6
RAW_INPUTS = {
    "input_flow": 2.0,
    "input_specific_energy": 18e6,
    "input_efficiency": 0.65,
    "energy_per_unit": 0.25e6,
    "flow_ratio": 2.0,
}
