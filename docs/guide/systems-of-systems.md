# Systems of systems

So far, every system in the guide has stood on its own.

Real engineering models rarely do. A larger model might contain components for propulsion, aerodynamics, structures, controls, or individual pieces of hardware.

Bilt lets an implementation use other implementations internally.

> A larger system can use smaller systems while keeping its own public boundary.

We will start with a resistor.

## Start with an ordinary system

```python exec="on" source="above" session="systems-of-systems"
from bilt import Implementation, Interface, equation, field, system


@system
class Resistor(metaclass=Interface):
    voltage: float = field()
    current: float = field()
    resistance: float = field()


@equation
def ohms_law(
    voltage: float,
    current: float,
    resistance: float,
) -> bool:
    return voltage == current * resistance


resistor_model = Implementation(
    Resistor,
    [ohms_law],
)
```

The resistor has its own boundary:

```text
Resistor
├── voltage
├── current
└── resistance
```

and one relationship:

```text
voltage = current × resistance
```

It is a complete Bilt model in its own right.

Now we will use it inside something larger.

## Build a parent system

Suppose a board contains a resistive load.

Its public boundary might be:

```python exec="on" source="above" session="systems-of-systems"
@system
class Board(metaclass=Interface):
    supply_voltage: float = field()
    load_resistance: float = field()
    supply_current: float = field()
```

From outside the model, these are the values we care about:

```text
Board
├── supply_voltage
├── load_resistance
└── supply_current
```

The resistor is not part of this public interface. It is part of one implementation of the board.

## Add an internal implementation

The board implementation will contain a named resistor implementation:

```text
Board implementation
├── public: Board
└── internal: resistor (Resistor)
```

Containment alone does not connect the two boundaries. Bilt requires each named child to be referenced by a parent equation.

## Connect parent and child values

Parent equations can refer to the child interface:

```python exec="on" source="above" session="systems-of-systems"
@equation
def voltage_connection(
    supply_voltage: float,
    resistor: Resistor,
) -> bool:
    return supply_voltage == resistor.voltage
```

Here:

```python exec="off"
resistor: Resistor
```

gives the equation access to values on the child boundary. The parameter name `resistor` must match the key in `internal_implementations`, and its annotation must match the child's system interface.

We can connect the remaining values in the same way:

```python exec="on" source="above" session="systems-of-systems"
@equation
def resistance_connection(
    load_resistance: float,
    resistor: Resistor,
) -> bool:
    return load_resistance == resistor.resistance


@equation
def current_connection(
    supply_current: float,
    resistor: Resistor,
) -> bool:
    return supply_current == resistor.current
```

Now build the complete implementation:

```python exec="on" source="above" session="systems-of-systems"
board_model = Implementation(
    Board,
    [
        voltage_connection,
        resistance_connection,
        current_connection,
    ],
    internal_implementations={
        "resistor": resistor_model,
    },
)
```

The model now contains:

```text
supply_voltage = resistor.voltage

load_resistance = resistor.resistance

supply_current = resistor.current

resistor.voltage =
    resistor.current × resistor.resistance
```

The board owns the connection equations.

The resistor owns Ohm's law.

## Use the parent through its normal interface

Create the values we know:

```python exec="on" source="above" session="systems-of-systems"
board = Board(
    supply_voltage=6.0,
    load_resistance=3.0,
)
```

Then request the value we want:

```python exec="on" source="above" session="systems-of-systems"
with board_model:
    assert board.supply_current == 2.0
```

The caller only interacts with `Board`.

Internally, Bilt can follow:

```text
supply_voltage
      │
      ▼
resistor.voltage

load_resistance
      │
      ▼
resistor.resistance

resistor.voltage
resistor.resistance
      │
      ▼
resistor.current
      │
      ▼
supply_current
```

The internal system participates in the computation without becoming part of the public API of `Board`.

## The relationships remain acausal

The resistor still contains the same equation:

```text
voltage = current × resistance
```

So we can supply current and resistance instead:

```python exec="on" source="above" session="systems-of-systems"
board = Board(
    supply_current=2.0,
    load_resistance=3.0,
)

with board_model:
    assert board.supply_voltage == 6.0
```

The system hierarchy does not impose a computation direction.

As before:

> The model describes relationships. The request determines the computation.

## Reuse implementations

The same child implementation can be used in many larger models:

```python exec="off"
resistor_model = Implementation(
    Resistor,
    [ohms_law],
)
```

It does not know which parent will contain it or what name it will have there.

This is what makes small systems useful as reusable modelling components.

## Use more than one internal system

A circuit containing two resistors in series can reuse the same implementation twice. Each child has its own voltage, resistance and current.

```python exec="on" source="above" session="systems-of-systems"
@system
class Circuit(metaclass=Interface):
    supply_voltage: float = field()
    supply_current: float = field()
    first_resistance: float = field()
    second_resistance: float = field()
```

Connect each resistance to the corresponding child:

```python exec="on" source="above" session="systems-of-systems"
@equation
def first_resistance_connection(
    first_resistance: float,
    first_resistor: Resistor,
) -> bool:
    return first_resistance == first_resistor.resistance


@equation
def second_resistance_connection(
    second_resistance: float,
    second_resistor: Resistor,
) -> bool:
    return second_resistance == second_resistor.resistance
```

The current is the same through both resistors. The voltage drops add up to the supply voltage. Equations on the parent can refer directly to more than one child:

```python exec="on" source="above" session="systems-of-systems"
@equation
def series_current(
    first_resistor: Resistor,
    second_resistor: Resistor,
) -> bool:
    return first_resistor.current == second_resistor.current


@equation
def supply_current_connection(
    supply_current: float,
    first_resistor: Resistor,
) -> bool:
    return supply_current == first_resistor.current


@equation
def series_voltage(
    supply_voltage: float,
    first_resistor: Resistor,
    second_resistor: Resistor,
) -> bool:
    return supply_voltage == first_resistor.voltage + second_resistor.voltage
```

Collect the connections and bind the two named children:

```python exec="on" source="above" session="systems-of-systems"
circuit_model = Implementation(
    Circuit,
    [
        first_resistance_connection,
        second_resistance_connection,
        series_current,
        supply_current_connection,
        series_voltage,
    ],
    internal_implementations={
        "first_resistor": resistor_model,
        "second_resistor": resistor_model,
    },
)
```

The two children are distinct even though they use the same implementation.

```text
Circuit
├── first_resistor
│   └── Resistor
└── second_resistor
    └── Resistor
```

Supply the current and both resistances to calculate the voltage:

```python exec="on" source="above" session="systems-of-systems"
with circuit_model:
    circuit = Circuit(
        supply_current=2.0,
        first_resistance=2.0,
        second_resistance=4.0,
    )
    assert circuit.supply_voltage == 12.0
```

Or supply the voltage and both resistances to calculate the current:

```python exec="on" source="above" session="systems-of-systems"
with circuit_model:
    circuit = Circuit(
        supply_voltage=12.0,
        first_resistance=2.0,
        second_resistance=4.0,
    )
    assert abs(circuit.supply_current - 2.0) < 1e-10
```

In the second request, the unknown currents and voltages are coupled. Bilt can solve the supported linear equations together without changing the model.

Alternatively, expose each voltage drop on the parent and relate those parent values. This is useful when callers need to inspect the intermediate values. It is not required for connecting children.

## Put it together

The complete board example is:

```python exec="on" source="above" session="board-complete"
from bilt import Implementation, Interface, equation, field, system


@system
class Resistor(metaclass=Interface):
    voltage: float = field()
    current: float = field()
    resistance: float = field()


@equation
def ohms_law(
    voltage: float,
    current: float,
    resistance: float,
) -> bool:
    return voltage == current * resistance


resistor_model = Implementation(
    Resistor,
    [ohms_law],
)


@system
class Board(metaclass=Interface):
    supply_voltage: float = field()
    load_resistance: float = field()
    supply_current: float = field()


@equation
def voltage_connection(
    supply_voltage: float,
    resistor: Resistor,
) -> bool:
    return supply_voltage == resistor.voltage


@equation
def resistance_connection(
    load_resistance: float,
    resistor: Resistor,
) -> bool:
    return load_resistance == resistor.resistance


@equation
def current_connection(
    supply_current: float,
    resistor: Resistor,
) -> bool:
    return supply_current == resistor.current


board_model = Implementation(
    Board,
    [
        voltage_connection,
        resistance_connection,
        current_connection,
    ],
    internal_implementations={
        "resistor": resistor_model,
    },
)
```

Use it like any other Bilt model:

```python exec="on" source="above" session="board-complete"
board = Board(
    supply_voltage=6.0,
    load_resistance=3.0,
)

with board_model:
    assert board.supply_current == 2.0
```

## What to remember

A larger implementation can contain smaller implementations:

```python exec="on" source="above" session="board-complete"
internal_implementations = {
    "resistor": resistor_model,
}
```

The child keeps its own interface and relationships.

Parent equations connect parent values to child values:

```python exec="off"
@equation
def voltage_connection(
    supply_voltage: float,
    resistor: Resistor,
) -> bool:
    return supply_voltage == resistor.voltage
```

Parent equations can connect children directly. Parent values can also provide a convenient coupling layer when callers need those intermediate values:

```text
child
  ↕
parent value
  ↕
parent relationship
  ↕
parent value
  ↕
child
```

Users still interact with the larger system through its normal public interface:

```python exec="on" source="above" session="board-complete"
with board_model:
    current = board.supply_current
```

The hierarchy organises the model.

The relationships determine what can be computed.

## When the parent and child must be solved together

A subsystem boundary does not always define an independent calculation. A parent's balance may depend on a child output while that output depends on a value determined by the same balance.

Bilt's structural analyser can retain those component equations in one supported linear block, preserving each child instance's identity. See [Coupled systems](coupled-systems.md) and the [recuperated gas turbine](../worked_examples/two-shaft-gas-turbine.md#14-close-the-loop-with-recuperation).

[← Dimensions, domains and tensors](dimensions-domains-and-tensors.md) · [Index](../index.md) · [Coupled systems →](coupled-systems.md)
