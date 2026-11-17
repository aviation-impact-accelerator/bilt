r"""Wheatstone bridge: a linear constraint solve built from the circuits library.

Topology (node C is ground):

               A
             /   \
         arm1     arm3
           /       \
          B -- G -- D        G: galvanometer (a resistor), current B -> D
           \       /
         arm2     arm4
             \   /
               C
    Source: + at A, - at C.

Kirchhoff's current law at A, B and D, equal potentials at each node, Ohm's law in
every arm. The same implementation answers the forward problem (currents from
resistances) and the inverse problem (the arm resistance that balances the bridge).
"""

from typing import Annotated

from bilt import Implementation, Interface, Quantity, QuantityTensor, equation, field, system
from bilt.units.dimensions import CURRENT, RESISTANCE, VOLTAGE
from examples.circuits._resistor import Resistor, resistor
from examples.circuits._voltage_source import VoltageSource, voltage_source


@system
class WheatstoneBridge(metaclass=Interface):
    """Four resistive arms, a supply across A-C and a galvanometer across B-D."""

    supply_voltage: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
    ground_potential: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
    arm1_resistance: Annotated[Quantity | QuantityTensor, RESISTANCE] = field()
    arm2_resistance: Annotated[Quantity | QuantityTensor, RESISTANCE] = field()
    arm3_resistance: Annotated[Quantity | QuantityTensor, RESISTANCE] = field()
    arm4_resistance: Annotated[Quantity | QuantityTensor, RESISTANCE] = field()
    galvanometer_resistance: Annotated[Quantity | QuantityTensor, RESISTANCE] = field()
    galvanometer_current: Annotated[Quantity | QuantityTensor, CURRENT] = field()
    supply_current: Annotated[Quantity | QuantityTensor, CURRENT] = field()


# --- Public values <-> component parameters -------------------------------------------


@equation
def supply_connection(supply_voltage: Annotated[Quantity, VOLTAGE], source: VoltageSource) -> bool:
    """Supply voltage is the source EMF."""
    return supply_voltage == source.voltage


@equation
def supply_current_connection(supply_current: Annotated[Quantity, CURRENT], source: VoltageSource) -> bool:
    """Supply current, positive when the source delivers power."""
    # Passive sign convention: a source delivering power carries negative current.
    return supply_current == -source.current


@equation
def arm1_connection(arm1_resistance: Annotated[Quantity, RESISTANCE], arm1: Resistor) -> bool:
    """Resistance of arm A-B."""
    return arm1_resistance == arm1.resistance


@equation
def arm2_connection(arm2_resistance: Annotated[Quantity, RESISTANCE], arm2: Resistor) -> bool:
    """Resistance of arm B-C."""
    return arm2_resistance == arm2.resistance


@equation
def arm3_connection(arm3_resistance: Annotated[Quantity, RESISTANCE], arm3: Resistor) -> bool:
    """Resistance of arm A-D."""
    return arm3_resistance == arm3.resistance


@equation
def arm4_connection(arm4_resistance: Annotated[Quantity, RESISTANCE], arm4: Resistor) -> bool:
    """Resistance of arm D-C."""
    return arm4_resistance == arm4.resistance


@equation
def galvanometer_resistance_connection(
    galvanometer_resistance: Annotated[Quantity, RESISTANCE], galvanometer: Resistor
) -> bool:
    """Resistance of the galvanometer."""
    return galvanometer_resistance == galvanometer.resistance


@equation
def galvanometer_current_connection(galvanometer_current: Annotated[Quantity, CURRENT], galvanometer: Resistor) -> bool:
    """Galvanometer current, positive from B to D."""
    return galvanometer_current == galvanometer.current


# --- Node A: source +, arm1 +, arm3 + -------------------------------------------------


@equation
def node_a_potential_1(source: VoltageSource, arm1: Resistor) -> bool:
    """Source and arm1 share node A."""
    return source.positive_potential == arm1.positive_potential


@equation
def node_a_potential_2(source: VoltageSource, arm3: Resistor) -> bool:
    """Source and arm3 share node A."""
    return source.positive_potential == arm3.positive_potential


@equation
def node_a_current(source: VoltageSource, arm1: Resistor, arm3: Resistor) -> bool:
    """Kirchhoff's current law at node A."""
    return source.current == -(arm1.current + arm3.current)


# --- Node B: arm1 -, arm2 +, galvanometer + -------------------------------------------


@equation
def node_b_potential_1(arm1: Resistor, arm2: Resistor) -> bool:
    """Arm1 and arm2 share node B."""
    return arm1.negative_potential == arm2.positive_potential


@equation
def node_b_potential_2(arm1: Resistor, galvanometer: Resistor) -> bool:
    """Arm1 and the galvanometer share node B."""
    return arm1.negative_potential == galvanometer.positive_potential


@equation
def node_b_current(arm1: Resistor, arm2: Resistor, galvanometer: Resistor) -> bool:
    """Kirchhoff's current law at node B."""
    return arm2.current + galvanometer.current == arm1.current


# --- Node D: arm3 -, arm4 +, galvanometer - -------------------------------------------


@equation
def node_d_potential_1(arm3: Resistor, arm4: Resistor) -> bool:
    """Arm3 and arm4 share node D."""
    return arm3.negative_potential == arm4.positive_potential


@equation
def node_d_potential_2(arm3: Resistor, galvanometer: Resistor) -> bool:
    """Arm3 and the galvanometer share node D."""
    return arm3.negative_potential == galvanometer.negative_potential


@equation
def node_d_current(arm3: Resistor, arm4: Resistor, galvanometer: Resistor) -> bool:
    """Kirchhoff's current law at node D."""
    return arm4.current == arm3.current + galvanometer.current


# --- Node C (ground): source -, arm2 -, arm4 - ----------------------------------------
# KCL at C is implied by the other three nodes, so only potentials are stated.


@equation
def node_c_ground(ground_potential: Annotated[Quantity, VOLTAGE], source: VoltageSource) -> bool:
    """Node C is at ground potential."""
    return source.negative_potential == ground_potential


@equation
def node_c_potential_1(source: VoltageSource, arm2: Resistor) -> bool:
    """Source and arm2 share node C."""
    return source.negative_potential == arm2.negative_potential


@equation
def node_c_potential_2(source: VoltageSource, arm4: Resistor) -> bool:
    """Source and arm4 share node C."""
    return source.negative_potential == arm4.negative_potential


wheatstone_bridge = Implementation(
    WheatstoneBridge,
    [
        supply_connection,
        supply_current_connection,
        arm1_connection,
        arm2_connection,
        arm3_connection,
        arm4_connection,
        galvanometer_resistance_connection,
        galvanometer_current_connection,
        node_a_potential_1,
        node_a_potential_2,
        node_a_current,
        node_b_potential_1,
        node_b_potential_2,
        node_b_current,
        node_d_potential_1,
        node_d_potential_2,
        node_d_current,
        node_c_ground,
        node_c_potential_1,
        node_c_potential_2,
    ],
    internal_implementations={
        "source": voltage_source,
        "arm1": resistor,
        "arm2": resistor,
        "arm3": resistor,
        "arm4": resistor,
        "galvanometer": resistor,
    },
)
