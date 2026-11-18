"""Proposal: connecting components through groups of across and flow ports.

Modelled on Modelica connectors and ModelingToolkit's `connect`, but adding as little
as possible to Bilt. It adds one decorator, one keyword, and one change in meaning.

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

**`@connection` (decorator).** A third kind of relation, alongside `@equation` and
`@transform`. Its parameters take children, or the system's own groups of ports, by
name and type, exactly as an equation's do, and it returns the groups of ports joined at
one junction:

    @connection
    def node_a(source: VoltageSource, arm1: Resistor, arm3: Resistor) -> tuple[Pin, Pin, Pin]:
        return source.positive, arm1.positive, arm3.positive

Joining n groups of the same interface generates n - 1 equalities per across port and
one conservation equation per flow port. A connection is not an equation because an
equation is exactly one relation, and a junction is several. The return annotation is
honest: called with concrete instances, the function returns those groups.
"""

import annotationlib
import typing

import bilt
from bilt_proposals._errors import not_implemented

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._port_declaration import PortDeclaration


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


class Connection[R: tuple[object, ...], **P]:
    """A junction declared with `@connection`, usable in an `Implementation` like an equation."""

    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_port_declarations__: dict[str, PortDeclaration]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Return the groups of ports this connection joins, for the given children."""
        del args, kwargs
        not_implemented(f"calling a @connection ({self.__bilt_name__})")


def connection[R: tuple[object, ...], **P](function: collections.abc.Callable[P, R]) -> Connection[R, P]:
    """Decorate a function returning the groups of ports joined at one junction."""
    not_implemented(f"@connection ({getattr(function, '__name__', function)})")
