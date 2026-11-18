r"""Wheatstone bridge with connected pins and public children.

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

A Wheatstone bridge is by definition four resistive arms, a galvanometer and a supply,
so every implementation of it has these children. They are therefore declared as
fields: their ports are part of the bridge's interface, and no pass-through equations
are needed. The implementation is just the topology, one `@connection` per node.
"""

from bilt_proposals import connection, field

from bilt import Implementation, Interface, system
from examples.circuits._ground import Ground, ground
from examples.circuits._pin import Pin
from examples.circuits._resistor import Resistor, resistor
from examples.circuits._voltage_source import VoltageSource, voltage_source


@system
class WheatstoneBridge(metaclass=Interface):
    """Four resistive arms, a supply across A-C and a galvanometer across B-D."""

    source: VoltageSource = field()
    ground: Ground = field()
    arm1: Resistor = field()
    arm2: Resistor = field()
    arm3: Resistor = field()
    arm4: Resistor = field()
    galvanometer: Resistor = field()


@connection
def node_a(source: VoltageSource, arm1: Resistor, arm3: Resistor) -> tuple[Pin, Pin, Pin]:
    """Node A: source +, arm1 +, arm3 +."""
    return source.positive, arm1.positive, arm3.positive


@connection
def node_b(arm1: Resistor, arm2: Resistor, galvanometer: Resistor) -> tuple[Pin, Pin, Pin]:
    """Node B: arm1 -, arm2 +, galvanometer +."""
    return arm1.negative, arm2.positive, galvanometer.positive


@connection
def node_c(source: VoltageSource, arm2: Resistor, arm4: Resistor, ground: Ground) -> tuple[Pin, Pin, Pin, Pin]:
    """Node C: source -, arm2 -, arm4 -, ground."""
    return source.negative, arm2.negative, arm4.negative, ground.terminal


@connection
def node_d(arm3: Resistor, arm4: Resistor, galvanometer: Resistor) -> tuple[Pin, Pin, Pin]:
    """Node D: arm3 -, arm4 +, galvanometer -."""
    return arm3.negative, arm4.positive, galvanometer.negative


wheatstone_bridge = Implementation(
    WheatstoneBridge,
    [node_a, node_b, node_c, node_d],
    internal_implementations={
        "source": voltage_source,
        "ground": ground,
        "arm1": resistor,
        "arm2": resistor,
        "arm3": resistor,
        "arm4": resistor,
        "galvanometer": resistor,
    },
)
