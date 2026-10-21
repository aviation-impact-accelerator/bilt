"""Steady heat-exchanger networks; all scalar values use consistent SI units."""

# ruff: noqa: INP001 - benchmark fixtures, not an installed package

import typing

from bilt import Implementation, Interface, equation, field, system


@system
class ThermalPorts(metaclass=Interface):
    """Temperature in K, heat in W, flow in kg/s and heat capacity in J/(kg K)."""

    inlet: float = typing.cast("float", field())
    outlet: float = typing.cast("float", field())
    heat: float = typing.cast("float", field())
    flow: float = typing.cast("float", field())
    capacity: float = typing.cast("float", field())


@equation
def heat_balance(inlet: float, outlet: float, heat: float, flow: float, capacity: float) -> bool:
    """Conserve thermal energy across one exchanger."""
    return heat == flow * capacity * (outlet - inlet)


@equation
def left_inlet(inlet: float, left: ThermalPorts) -> bool:
    """Feed the first exchanger from the network inlet."""
    return left.inlet == inlet


@equation
def right_inlet_series(left: ThermalPorts, right: ThermalPorts) -> bool:
    """Connect the first exchanger's outlet to the second exchanger's inlet."""
    return right.inlet == left.outlet


@equation
def right_inlet_parallel(inlet: float, right: ThermalPorts) -> bool:
    """Feed both branches from the common inlet."""
    return right.inlet == inlet


@equation
def left_heat(heat: float, left: ThermalPorts) -> bool:
    """Allocate half the total heat to the first branch."""
    return left.heat == heat / 2


@equation
def right_heat(heat: float, right: ThermalPorts) -> bool:
    """Allocate half the total heat to the second branch."""
    return right.heat == heat / 2


@equation
def left_capacity(capacity: float, left: ThermalPorts) -> bool:
    """Propagate fluid heat capacity."""
    return left.capacity == capacity


@equation
def right_capacity(capacity: float, right: ThermalPorts) -> bool:
    """Propagate fluid heat capacity."""
    return right.capacity == capacity


@equation
def left_flow_series(flow: float, left: ThermalPorts) -> bool:
    """Pass the whole stream through each series exchanger."""
    return left.flow == flow


@equation
def right_flow_series(flow: float, right: ThermalPorts) -> bool:
    """Pass the whole stream through each series exchanger."""
    return right.flow == flow


@equation
def left_flow_parallel(flow: float, left: ThermalPorts) -> bool:
    """Split the stream equally between parallel branches."""
    return left.flow == flow / 2


@equation
def right_flow_parallel(flow: float, right: ThermalPorts) -> bool:
    """Split the stream equally between parallel branches."""
    return right.flow == flow / 2


@equation
def outlet_series(outlet: float, right: ThermalPorts) -> bool:
    """Expose the final exchanger's outlet."""
    return outlet == right.outlet


@equation
def outlet_parallel(outlet: float, left: ThermalPorts, right: ThermalPorts) -> bool:
    """Mix equal-flow parallel streams with the same heat capacity."""
    return outlet == (left.outlet + right.outlet) / 2


def build_thermal_model(layout: str, depth: int) -> Implementation:
    """Build 2**depth exchangers in series or balanced parallel branches."""
    if layout not in {"series", "parallel"} or depth < 0:
        raise ValueError((layout, depth))
    implementation = Implementation(ThermalPorts, [heat_balance])
    common = (left_inlet, left_heat, right_heat, left_capacity, right_capacity)
    connections = (
        (right_inlet_series, left_flow_series, right_flow_series, outlet_series)
        if layout == "series"
        else (right_inlet_parallel, left_flow_parallel, right_flow_parallel, outlet_parallel)
    )
    for _ in range(depth):
        implementation = Implementation(
            ThermalPorts,
            (*common, *connections),
            internal_implementations=dict.fromkeys(("left", "right"), implementation),
        )
    return implementation
