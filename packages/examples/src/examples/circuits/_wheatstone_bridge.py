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
are needed. The implementation is just the topology, one `Connection` per node.
"""

from bilt_proposals import Connection, equation, field

from bilt import Implementation, Interface, system
from examples.circuits._ground import Ground
from examples.circuits._pin import Pin
from examples.circuits._resistor import Resistor
from examples.circuits._voltage_source import VoltageSource


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


@equation
def node_a(source: VoltageSource, arm1: Resistor, arm3: Resistor) -> Connection[Pin]:
    """Node A: source +, arm1 +, arm3 +."""
    return Connection(source.positive, arm1.positive, arm3.positive)


@equation
def node_b(arm1: Resistor, arm2: Resistor, galvanometer: Resistor) -> Connection[Pin]:
    """Node B: arm1 -, arm2 +, galvanometer +."""
    return Connection(arm1.negative, arm2.positive, galvanometer.positive)


@equation
def node_c(source: VoltageSource, arm2: Resistor, arm4: Resistor, ground: Ground) -> Connection[Pin]:
    """Node C: source -, arm2 -, arm4 -, ground."""
    return Connection(source.negative, arm2.negative, arm4.negative, ground.terminal)


@equation
def node_d(arm3: Resistor, arm4: Resistor, galvanometer: Resistor) -> Connection[Pin]:
    """Node D: arm3 -, arm4 +, galvanometer -."""
    return Connection(arm3.negative, arm4.positive, galvanometer.negative)


ideal_wheatstone_bridge = Implementation(WheatstoneBridge, [node_a, node_b, node_c, node_d])
