"""Proposal: connecting components through groups of across and flow ports.

Modelled on Modelica connectors and ModelingToolkit's `connect`, but adding as little
as possible to Bilt. It adds one type, one keyword, and one change in meaning.

**Interface-typed fields are groups of ports (change in meaning).** A field typed by an
interface contributes that interface's ports to the system, named with dots, the way a
sub-system's ports already are inside a parent implementation. An electrical pin is an
ordinary system with no implementation:

    @system
    class Pin(metaclass=Interface):
        potential: Annotated[Quantity, VOLTAGE] = field()
        current: Annotated[Quantity, CURRENT] = field(flow=True)

and a resistor with `positive: Pin = field()` has ports `positive.potential` and
`positive.current`. A parent can then read `arm1.positive.potential` through `arm1`'s
interface, so children stay encapsulated.

The same rule lets a system declare children publicly. Declare a child as a field when
every implementation the system could have contains that child (a Wheatstone bridge is
always four resistive arms and a galvanometer): its ports are then part of the
interface and no pass-through equations are needed. Keep a child private, as an
equation parameter only, when implementations may differ in their children (a generic
`Circuit` exposing only `supply_voltage` and `supply_current`). The choice trades
access to children's ports against freedom to implement the system differently.

A request gives what it knows about a public child as an instance of the child's
interface, with the usual meaning (given ports known, omitted ports unknown), and reads
results through it:

    bridge = WheatstoneBridge(arm1=Resistor(resistance=100.0 * ohm), ...)
    bridge.arm4.resistance

Bilt today reads an interface-typed field as an undeclared sub-system, so the
proposed `field()` raises `ProposalNotImplementedError` when declared on one.

**`field(flow=True)` (keyword).** Marks a port as a flow: summing to zero over every
group joined at a junction, each counted positive into its component. Every other port
is an across port, equal at every group joined at a junction. Across is the default,
as in Modelica and ModelingToolkit, so it needs no marker.

**`Connection` (type).** The groups of ports joined at one junction. An `@equation`
returns one instead of a relation, which makes the equation a junction: its parameters
take children by name and type, as any equation's do, and its return annotation is
honest, since called with concrete instances the function returns that connection:

    @equation
    def node_a(source: VoltageSource, arm1: Resistor, arm3: Resistor) -> Connection[Pin]:
        return Connection(source.positive, arm1.positive, arm3.positive)

Joining n groups generates n - 1 equalities per across port and one conservation
equation per flow port:

    source.positive.potential == arm1.positive.potential
    source.positive.potential == arm3.positive.potential
    source.positive.current + arm1.positive.current + arm3.positive.current == 0

This reuses `@equation` rather than adding a kind of relation, as Modelica's `connect`
sits in an `equation` section and ModelingToolkit's in the equations list. A junction
is named for its place on a diagram (`node_a`), not as a fact. Writing
`Connection[Pin]` documents the kind of group, and lets a type checker flag a mix.

Rules, checked when the `Implementation` is built:

- **Groups only.** Bare ports are related with `==`: a bare flow port in a connection
  would sum where `==` equates.
- **One interface.** Every group has the same interface, by identity: ports match by
  name only within it.
- **At least two groups.** One group would silently zero its flow ports, hiding a
  forgotten group. An open end is written as the equation it is
  (`spring.flange_b.force == 0.0 * newton`).
- **A group in at most one connection.** Each node is written once. Connections
  sharing a group would have to be merged to conserve flow, and merging hides a wrong
  pin (`arm1.positive` for `arm1.negative` shorts an arm).

**The system's own groups.** A system exposes groups to its parent as fields of its
interface, and connects them to its children. An own group's flow is positive into the
system, from outside, so it enters the junction and counts with the opposite sign:

    @system
    class VoltageDivider(metaclass=Interface):
        top: Pin = field()
        upper: Resistor = field()
        ...

    @equation
    def top_node(top: Pin, upper: Resistor) -> Connection[Pin]:
        return Connection(top, upper.positive)
        # top.potential == upper.positive.potential
        # upper.positive.current - top.current == 0

A parameter that is one of the system's own interface-typed fields, used directly, is an
own group; a group reached through a child (`upper.positive`) is not. In the parent,
`divider.top` is a child's group and counts positively, so flow is conserved across the
boundary (Modelica's inside and outside connectors).
"""

import annotationlib
import typing

import bilt
from bilt_proposals._errors import not_implemented


class _Field:
    """Stands in for `bilt.field()` until the class is created, then checks the annotation."""

    def __set_name__(self, owner: type, name: str) -> None:
        annotations = annotationlib.get_annotations(owner, format=annotationlib.Format.FORWARDREF)
        annotation = annotations.get(name)
        if isinstance(annotation, bilt.Interface):
            not_implemented(f"interface-typed field ({owner.__name__}.{name}: {annotation.__bilt_name__})")
        setattr(owner, name, bilt.field())


def field(*, flow: bool = False) -> typing.Any:  # noqa: ANN401
    """Declare a port, as `bilt.field()`, optionally as a flow port or a group of ports.

    For an ordinary port this is `bilt.field()`; it exists so a module can import a
    single `field` for all its ports.
    """
    if flow:
        not_implemented("field(flow=True)")
    return _Field()


class Connection[T]:
    """The groups of ports joined at one junction, returned by an `@equation`."""

    def __init__(self, *groups: T) -> None:
        """Join `groups`, all of one interface."""
        del groups
        not_implemented("Connection")
