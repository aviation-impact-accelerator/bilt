from __future__ import annotations
from typing import Annotated
from bilt.units import DIMENSIONLESS, kilogram, meter, second, kelvin, pascal, joule, watt
from examples.symbolic_api_and_basic_backend import Display, Eq, Equation, EquationModel, Quantity, equation

@equation
def reynolds_number_definition(
    reynolds_number: Annotated[Quantity, DIMENSIONLESS, Display("Re")],
    density: Annotated[Quantity, kilogram / meter**3],
    velocity: Annotated[Quantity, meter / second],
    length: Annotated[Quantity, meter],
    dynamic_viscosity: Annotated[Quantity, pascal * second],
) -> Equation:
    return Eq(reynolds_number, density * velocity * length / dynamic_viscosity)

@equation
def prandtl_number_definition(
    prandtl_number: Annotated[Quantity, DIMENSIONLESS, Display("Pr")],
    specific_heat_capacity: Annotated[Quantity, joule / kilogram / kelvin],
    dynamic_viscosity: Annotated[Quantity, pascal * second],
    thermal_conductivity: Annotated[Quantity, watt / meter / kelvin],
) -> Equation:
    return Eq(prandtl_number, specific_heat_capacity * dynamic_viscosity / thermal_conductivity)

@equation
def nusselt_number_definition(
    nusselt_number: Annotated[Quantity, DIMENSIONLESS, Display("Nu")],
    reynolds_number: Annotated[Quantity, DIMENSIONLESS, Display("Re")],
    prandtl_number: Annotated[Quantity, DIMENSIONLESS, Display("Pr")],
    exponent: Annotated[Quantity, DIMENSIONLESS],
) -> Equation:
    return Eq(nusselt_number, 0.023 * reynolds_number**0.8 * prandtl_number**exponent)

@equation
def heat_transfer_coefficient_definition(
    heat_transfer_coefficient: Annotated[Quantity, watt / meter**2 / kelvin, Display("h")],
    nusselt_number: Annotated[Quantity, DIMENSIONLESS, Display("Nu")],
    thermal_conductivity: Annotated[Quantity, watt / meter / kelvin],
    length: Annotated[Quantity, meter],
) -> Equation:
    return Eq(heat_transfer_coefficient, nusselt_number * thermal_conductivity / length)

def build_velocity_from_htc_model() -> EquationModel:
    model = EquationModel("velocity_from_desired_htc")
    model.add(reynolds_number_definition)
    model.add(prandtl_number_definition)
    model.add(nusselt_number_definition)
    model.add(heat_transfer_coefficient_definition)
    return model

def solve_velocity_from_htc_demo() -> None:
    model = build_velocity_from_htc_model()
    result = model.solve(inputs={
        "heat_transfer_coefficient": 4754.327547388351,
        "density": 1_000.0,
        "length": 0.01,
        "dynamic_viscosity": 0.001,
        "specific_heat_capacity": 4_180.0,
        "thermal_conductivity": 0.6,
        "exponent": 0.4,
    }, outputs=["velocity"])
    print("Requested outputs:"); print(result.requested_outputs)
    print("\nPlan:"); print(result.graph.describe())
    print("\nRearranged expressions:")
    for transform in result.graph.transforms:
        for output, expression in transform.expressions.items(): print(f"  {output} = {expression}")

if __name__ == "__main__":
    solve_velocity_from_htc_demo()
