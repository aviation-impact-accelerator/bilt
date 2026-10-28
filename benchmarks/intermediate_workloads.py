"""Engineering reporting workloads built from the repository's existing models."""

# ruff: noqa: INP001 - benchmark helper, not an installed package

import dataclasses
import math

import numpy as np
from difficult_models import build_mixed_model
from examples.hx.geometry import _tube_geometry as geometry
from thermal_models import ThermalPorts, build_thermal_model

from bilt import Implementation, Interface, field, system, tensor, transform
from bilt.core._equation import equation
from bilt.symbolic._dimensions import dim


@system
class TubeGeometry(metaclass=Interface):
    """The tube example's five geometry outputs."""

    tube_outer_radius: float = field()
    tube_wall_thickness: float = field()
    tube_length: float = field()
    n_elements: int = field()
    tube_inner_radius: float = field()
    tube_inner_diameter: float = field()
    tube_flow_area: float = field()
    element_length: float = field()
    tube_outer_area_element: float = field()


@system
class ThermalReport(metaclass=Interface):
    """Boundary conditions plus three report values for a nested thermal network."""

    inlet: float = field()
    heat: float = field()
    flow: float = field()
    capacity: float = field()
    outlet: float = field()
    rise: float = field()
    heat_per_flow: float = field()


@equation
def connect_inlet(inlet: float, network: ThermalPorts) -> bool:
    """Forward the inlet boundary condition."""
    return inlet == network.inlet


@equation
def connect_heat(heat: float, network: ThermalPorts) -> bool:
    """Forward the heat boundary condition."""
    return heat == network.heat


@equation
def connect_flow(flow: float, network: ThermalPorts) -> bool:
    """Forward the flow boundary condition."""
    return flow == network.flow


@equation
def connect_capacity(capacity: float, network: ThermalPorts) -> bool:
    """Forward the fluid heat capacity."""
    return capacity == network.capacity


@equation
def connect_outlet(outlet: float, network: ThermalPorts) -> bool:
    """Expose the computed outlet temperature."""
    return outlet == network.outlet


@transform
def rise(outlet: float, inlet: float) -> float:
    """Report the network temperature rise."""
    return outlet - inlet


@transform
def heat_per_flow(outlet: float, inlet: float, capacity: float) -> float:
    """Report specific heat addition from the computed temperatures."""
    return capacity * (outlet - inlet)


@transform
def tube_flow_area(tube_inner_radius: float) -> float:
    """The existing example formula, with its imported pi constant inlined."""
    return 3.141592653589793 * tube_inner_radius**2


@transform
def tube_outer_area_element(tube_outer_radius: float, element_length: float) -> float:
    """The existing example formula, with its imported pi constant inlined."""
    return 2.0 * 3.141592653589793 * tube_outer_radius * element_length


@dataclasses.dataclass
class Workload:
    """A model, supplied values, and outputs in reporting order."""

    implementation: Implementation
    inputs: dict[str, object]
    expected: dict[str, object]


def prepare_workload(family: str, size: int) -> Workload:
    """Use scalar inputs or a batch of design/operating points, with independent references."""
    batch = dim("operating_point")
    if family == "geometry":
        radii = 0.01 if size == 1 else np.linspace(0.01, 0.02, size)
        inputs: dict[str, object] = {
            "tube_outer_radius": radii if size == 1 else tensor(radii, over=batch),
            "tube_wall_thickness": 0.001,
            "tube_length": 1.0,
            "n_elements": 10,
        }
        inner = radii - 0.001
        expected: dict[str, object] = {
            "tube_flow_area": math.pi * inner**2,
            "tube_inner_diameter": 2 * inner,
            "tube_outer_area_element": 2 * math.pi * radii * 0.1,
            "tube_inner_radius": inner,
        }
        return Workload(
            Implementation(
                TubeGeometry,
                [
                    transform(geometry.tube_inner_radius),
                    transform(geometry.tube_inner_diameter),
                    tube_flow_area,
                    transform(geometry.element_length),
                    tube_outer_area_element,
                ],
            ),
            inputs,
            expected,
        )
    if family not in {"series", "mixed"}:
        raise ValueError(family)
    heat = 120_000.0 if size == 1 else np.linspace(60_000.0, 120_000.0, size)
    inputs = {
        "inlet": 300.0,
        "heat": heat if size == 1 else tensor(heat, over=batch),
        "flow": 2.0,
        "capacity": 4_000.0,
    }
    network = build_thermal_model("series", 1) if family == "series" else build_mixed_model(2)
    implementation = Implementation(
        ThermalReport,
        [connect_inlet, connect_heat, connect_flow, connect_capacity, connect_outlet, rise, heat_per_flow],
        internal_implementations={"network": network},
    )
    delta = heat / (2.0 * 4_000.0)
    return Workload(implementation, inputs, {"rise": delta, "outlet": 300.0 + delta, "heat_per_flow": heat / 2.0})
