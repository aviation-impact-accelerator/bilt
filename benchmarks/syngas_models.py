"""Nested syngas model reproducing expensive structural planning in bilt-fuels.

These illustrative relations are benchmark inputs, not validated fuel chemistry.
"""

# ruff: noqa: D101, D103, INP001 - copied equation fixtures

import typing

from bilt import Implementation, Interface, Quantity, equation, field, system
from bilt.units import fraction
from bilt.units.dimensions import AMOUNT, DIMENSIONLESS, ENERGY, MASS, POWER, TIME

MassFlow = typing.Annotated[Quantity, MASS / TIME]
MolarFlow = typing.Annotated[Quantity, AMOUNT / TIME]
SpecificEnergy = typing.Annotated[Quantity, ENERGY / MASS]
EnergyPerMole = typing.Annotated[Quantity, ENERGY / AMOUNT]
Power = typing.Annotated[Quantity, POWER]
Fraction = typing.Annotated[Quantity, DIMENSIONLESS]


@system
class Gasification(metaclass=Interface):
    biomass_flow: MassFlow = field()
    biomass_lhv: SpecificEnergy = field()
    conversion_efficiency: Fraction = field()
    syngas_power: Power = field()


@equation
def gasification_energy(
    biomass_flow: MassFlow, biomass_lhv: SpecificEnergy, conversion_efficiency: Fraction, syngas_power: Power
) -> bool:
    return syngas_power == biomass_flow * biomass_lhv * conversion_efficiency


@system
class SyngasComposition(metaclass=Interface):
    syngas_power: Power = field()
    energy_per_mole: EnergyPerMole = field()
    h2_co_ratio: Fraction = field()
    carbon_monoxide_flow: MolarFlow = field()
    hydrogen_flow: MolarFlow = field()


@equation
def original_hydrogen_flow(carbon_monoxide_flow: MolarFlow, hydrogen_flow: MolarFlow, h2_co_ratio: Fraction) -> bool:
    return hydrogen_flow == carbon_monoxide_flow * h2_co_ratio


@equation
def original_syngas_energy(
    carbon_monoxide_flow: MolarFlow,
    hydrogen_flow: MolarFlow,
    energy_per_mole: EnergyPerMole,
    syngas_power: Power,
) -> bool:
    return syngas_power == (carbon_monoxide_flow + hydrogen_flow) * energy_per_mole


@equation
def explicit_carbon_monoxide_flow(
    carbon_monoxide_flow: MolarFlow,
    syngas_power: Power,
    energy_per_mole: EnergyPerMole,
    h2_co_ratio: Fraction,
) -> bool:
    return carbon_monoxide_flow == syngas_power / (energy_per_mole * (1.0 * fraction + h2_co_ratio))


@equation
def explicit_hydrogen_flow(carbon_monoxide_flow: MolarFlow, hydrogen_flow: MolarFlow, h2_co_ratio: Fraction) -> bool:
    return hydrogen_flow == carbon_monoxide_flow * h2_co_ratio


@system
class FischerTropsch(metaclass=Interface):
    syngas_power: Power = field()
    jet_yield: Fraction = field()
    jet_power: Power = field()
    carbon_monoxide_flow: MolarFlow = field()
    hydrogen_flow: MolarFlow = field()
    h2_co_ratio: Fraction = field()


@equation
def ft_energy(syngas_power: Power, jet_yield: Fraction, jet_power: Power) -> bool:
    return jet_power == syngas_power * jet_yield


@equation
def ft_ratio(carbon_monoxide_flow: MolarFlow, hydrogen_flow: MolarFlow, h2_co_ratio: Fraction) -> bool:
    return hydrogen_flow == carbon_monoxide_flow * h2_co_ratio


@equation
def ft_conditional_yield(h2_co_ratio: Fraction, jet_yield: Fraction) -> bool:
    if h2_co_ratio >= 2.0 * fraction:
        selected = 0.45 * fraction
    elif h2_co_ratio >= 1.0 * fraction:
        selected = 0.35 * fraction
    else:
        selected = 0.25 * fraction
    return jet_yield == selected


@system
class BiomassToLiquid(metaclass=Interface):
    biomass_flow: MassFlow = field()
    biomass_lhv: SpecificEnergy = field()
    gasification_efficiency: Fraction = field()
    ft_jet_yield: Fraction = field()
    syngas_power: Power = field()
    jet_power: Power = field()
    h2_co_ratio: Fraction = field()
    syngas_energy_per_mole: EnergyPerMole = field()
    carbon_monoxide_flow: MolarFlow = field()
    hydrogen_flow: MolarFlow = field()


@equation
def connect_biomass_flow(biomass_flow: MassFlow, gasifier: Gasification) -> bool:
    return biomass_flow == gasifier.biomass_flow


@equation
def connect_biomass_lhv(biomass_lhv: SpecificEnergy, gasifier: Gasification) -> bool:
    return biomass_lhv == gasifier.biomass_lhv


@equation
def connect_gasification_efficiency(gasification_efficiency: Fraction, gasifier: Gasification) -> bool:
    return gasification_efficiency == gasifier.conversion_efficiency


@equation
def connect_syngas_from_gasifier(syngas_power: Power, gasifier: Gasification) -> bool:
    return syngas_power == gasifier.syngas_power


@equation
def connect_syngas_to_ft(syngas_power: Power, ft: FischerTropsch) -> bool:
    return syngas_power == ft.syngas_power


@equation
def connect_ft_jet_yield(ft_jet_yield: Fraction, ft: FischerTropsch) -> bool:
    return ft_jet_yield == ft.jet_yield


@equation
def connect_jet_power(jet_power: Power, ft: FischerTropsch) -> bool:
    return jet_power == ft.jet_power


@equation
def connect_power_to_syngas(syngas_power: Power, stream: SyngasComposition) -> bool:
    return syngas_power == stream.syngas_power


@equation
def connect_assumed_molar_energy(syngas_energy_per_mole: EnergyPerMole, stream: SyngasComposition) -> bool:
    return syngas_energy_per_mole == stream.energy_per_mole


@equation
def connect_assumed_ratio(h2_co_ratio: Fraction, stream: SyngasComposition) -> bool:
    return h2_co_ratio == stream.h2_co_ratio


@equation
def connect_co_from_syngas(carbon_monoxide_flow: MolarFlow, stream: SyngasComposition) -> bool:
    return carbon_monoxide_flow == stream.carbon_monoxide_flow


@equation
def connect_h2_from_syngas(hydrogen_flow: MolarFlow, stream: SyngasComposition) -> bool:
    return hydrogen_flow == stream.hydrogen_flow


@equation
def connect_co_to_ft(carbon_monoxide_flow: MolarFlow, ft: FischerTropsch) -> bool:
    return carbon_monoxide_flow == ft.carbon_monoxide_flow


@equation
def connect_h2_to_ft(hydrogen_flow: MolarFlow, ft: FischerTropsch) -> bool:
    return hydrogen_flow == ft.hydrogen_flow


PARENT_CONNECTIONS = (
    connect_biomass_flow,
    connect_biomass_lhv,
    connect_gasification_efficiency,
    connect_syngas_from_gasifier,
    connect_syngas_to_ft,
    connect_ft_jet_yield,
    connect_jet_power,
    connect_power_to_syngas,
    connect_assumed_molar_energy,
    connect_assumed_ratio,
    connect_co_from_syngas,
    connect_h2_from_syngas,
    connect_co_to_ft,
    connect_h2_to_ft,
)


def build_syngas_model(*, coupled: bool) -> Implementation:
    """Keep the parent and FT fixed; vary only the two syngas equations."""
    gasifier = Implementation(Gasification, [gasification_energy])
    syngas = Implementation(
        SyngasComposition,
        [original_hydrogen_flow, original_syngas_energy]
        if coupled
        else [explicit_carbon_monoxide_flow, explicit_hydrogen_flow],
    )
    ft = Implementation(FischerTropsch, [ft_energy, ft_ratio, ft_conditional_yield])
    return Implementation(
        BiomassToLiquid,
        PARENT_CONNECTIONS,
        internal_implementations={"gasifier": gasifier, "stream": syngas, "ft": ft},
    )


# The benchmark checks the independent expected result for 2 kg/s of biomass,
# 18 MJ/kg LHV, 65% gasification conversion and 45% conditional FT yield.
EXPECTED_JET_POWER_W = 10.53e6
RAW_INPUTS = {
    "biomass_flow": 2.0,
    "biomass_lhv": 18e6,
    "gasification_efficiency": 0.65,
    "syngas_energy_per_mole": 0.25e6,
    "h2_co_ratio": 2.0,
}
