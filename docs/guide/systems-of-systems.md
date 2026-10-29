## Systems of systems

So far, every system in the guide has stood on its own.

Real engineering models rarely do. A larger model might contain components for propulsion, aerodynamics, structures, controls, or individual pieces of hardware.

Bilt lets an implementation use other implementations internally.

> A larger system can use smaller systems while keeping its own public boundary.

We will start with a resistor.

### Start with an ordinary system

```python
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

### Build a parent system

Suppose a board contains a resistive load.

Its public boundary might be:

```python
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

### Add an internal implementation

Make the resistor available inside the board implementation:

```python
board_model = Implementation(
    Board,
    [],
    internal_implementations={
        "resistor": resistor_model,
    },
)
```

Conceptually:

```text
Board
│
├── supply_voltage
├── load_resistance
├── supply_current
│
└── internal: resistor
    ├── voltage
    ├── current
    └── resistance
```

At this point, the resistor is contained by the board implementation, but nothing connects their boundaries.

Containment alone does not imply a relationship.

### Connect parent and child values

Parent equations can refer to the child interface:

```python
@equation
def voltage_connection(
    supply_voltage: float,
    resistor: Resistor,
) -> bool:
    return supply_voltage == resistor.voltage
```

Here:

```python
resistor: Resistor
```

gives the equation access to values on the child boundary.

We can connect the remaining values in the same way:

```python
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

```python
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

### Use the parent through its normal interface

Create the values we know:

```python
board = Board(
    supply_voltage=6.0,
    load_resistance=3.0,
)
```

Then request the value we want:

```python
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

### The relationships remain acausal

The resistor still contains the same equation:

```text
voltage = current × resistance
```

So we can supply current and resistance instead:

```python
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

### Reuse implementations

The same child implementation can be used in many larger models:

```python
resistor_model = Implementation(
    Resistor,
    [ohms_law],
)
```

It does not know which parent will contain it or what name it will have there.

This is what makes small systems useful as reusable modelling components.

### Use more than one internal system

A parent can contain several internal implementations:

```python
circuit_model = Implementation(
    Circuit,
    [...],
    internal_implementations={
        "first_resistor": resistor_model,
        "second_resistor": resistor_model,
    },
)
```

The two occurrences are distinct even though they share the same implementation:

```text
Circuit
├── first_resistor
│   └── Resistor
│
└── second_resistor
    └── Resistor
```

One useful modelling pattern is to connect each child to values on the parent and relate those values there. This makes intermediate boundary quantities easy to inspect.

This is not a restriction on which child interfaces an equation may mention. The default causalization analyser can solve supported coupled linear equations across parent and child boundaries.

For example:

```python
@equation
def first_voltage_connection(
    first_voltage: float,
    first_resistor: Resistor,
) -> bool:
    return first_voltage == first_resistor.voltage


@equation
def second_voltage_connection(
    second_voltage: float,
    second_resistor: Resistor,
) -> bool:
    return second_voltage == second_resistor.voltage
```

Then relate the parent values:

```python
@equation
def total_voltage_relation(
    total_voltage: float,
    first_voltage: float,
    second_voltage: float,
) -> bool:
    return total_voltage == first_voltage + second_voltage
```

The coupling therefore passes through the parent:

```text
first_resistor.voltage
          ↕
     first_voltage
          │
          │
     total_voltage
          │
          │
    second_voltage
          ↕
second_resistor.voltage
```

When the boundary inputs can be established independently, each child can be evaluated on its own. Parent aliases do not remove mathematical coupling; a linear feedback loop may require the equations to be solved together.

### Put it together

The complete board example is:

```python
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

```python
board = Board(
    supply_voltage=6.0,
    load_resistance=3.0,
)

with board_model:
    assert board.supply_current == 2.0
```

### What to remember

A larger implementation can contain smaller implementations:

```python
internal_implementations = {
    "resistor": resistor_model,
}
```

The child keeps its own interface and relationships.

Parent equations connect parent values to child values:

```python
@equation
def voltage_connection(
    supply_voltage: float,
    resistor: Resistor,
) -> bool:
    return supply_voltage == resistor.voltage
```

Parent values can provide a convenient coupling layer when several children interact:

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

```python
with board_model:
    current = board.supply_current
```

The hierarchy organises the model.

The relationships determine what can be computed.



### When the parent and child must be solved together

A subsystem boundary does not always define an independent calculation. A parent's balance may depend on a child output while that output depends on a value determined by the same balance.

The SCC analyser can retain those primitive equations in one supported linear block, preserving each child instance's identity. See [Coupled linear systems](coupled-linear-systems.md) and the [recuperated gas turbine](../worked_examples/two-shaft-gas-turbine.md#14-close-the-loop-with-recuperation).

[← Dimensions, domains and tensors](dimensions-domains-and-tensors.md) · [Index](../index.md) · [Coupled linear systems →](coupled-linear-systems.md)
