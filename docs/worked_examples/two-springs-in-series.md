# Two springs in series: a small acausal system of systems

This worked example uses **two linear springs connected in series** to introduce acausal systems-of-systems modelling in Bilt.

The physical arrangement is:

```text
fixed support
    │
    ├── spring 1 ── joint ── spring 2 ── applied load
    │
```

More schematically:

```mermaid
flowchart LR
    support[Fixed support] --- s1["Spring 1: k₁, x₁"]
    s1 --- joint[Intermediate joint]
    joint --- s2["Spring 2: k₂, x₂"]
    s2 --- load["Applied load: F"]
```

Because the springs are in **series**:

- the same force passes through both springs;
- each spring can extend by a different amount;
- the total extension is the sum of the two spring extensions.

So:

```text
F₁ = F₂ = F

x_total = x₁ + x₂
```

and each spring obeys Hooke's law:

```text
F = k × x
```

The useful part of the example is how Bilt composes those relationships.

We will:

- define one reusable `Spring` model;
- use it twice inside a larger `DoubleSpring` implementation;
- connect each child to ordinary values on the parent;
- ask the same equations for different quantities.

The requests in this example can be decomposed into independent child-system evaluations and ordinary parent equations. They do not require a coupled solver.

For requests that do require equations from multiple children to be solved together, see [Coupled linear systems](../guide/coupled-linear-systems.md) and its current analyser availability note.

```python exec="on" session="springs"
from io import StringIO
import importlib.util
from pathlib import Path
import tempfile

import matplotlib

# markdown-exec runs code in a worker thread.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from bilt.units import meter, newton


def render_plot() -> None:
    buffer = StringIO()
    plt.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close()
    print(buffer.getvalue())
```

## 1. Start with one spring

A spring has three boundary values: stiffness, extension, and force. We declare
their dimensions rather than fixing particular units. Stiffness has dimension
force per length, extension has dimension length, and force has dimension force.

The system and equations use scalar quantity declarations. These accept scalar
quantities or compatible unit-carrying tensors, allowing the same relationships
to be reused over named domains. In this example we supply stiffness in N/m,
extension in m, and force in N:

```python
from typing import Annotated

from bilt import Implementation, Interface, Quantity, equation, field, system
from bilt.units import meter, newton
from bilt.units.dimensions import FORCE, LENGTH


Stiffness = Annotated[Quantity, FORCE / LENGTH]
Length = Annotated[Quantity, LENGTH]
Force = Annotated[Quantity, FORCE]


@system
class Spring(metaclass=Interface):
    stiffness: Stiffness = field()
    extension: Length = field()
    force: Force = field()
```

Hooke's law is an equation:

```python
@equation
def hookes_law(
    stiffness: Stiffness,
    extension: Length,
    force: Force,
) -> bool:
    return force == stiffness * extension


linear_spring = Implementation(
    Spring,
    [hookes_law],
)
```

Because this is an equation, none of the three values is permanently designated as the output.

Given stiffness and extension, Bilt can find force:

```text
k + x → F
```

Given force and stiffness, the same equation can find extension:

```text
F + k → x
```

That is the acausal part of the component model.

## 2. Build a larger system

The parent system exposes the values we want to use when working with two springs:

```python
@system
class DoubleSpring(metaclass=Interface):
    first_stiffness: Stiffness = field()
    second_stiffness: Stiffness = field()

    first_extension: Length = field()
    second_extension: Length = field()
    total_extension: Length = field()

    force: Force = field()
```

The two `Spring` occurrences are not public fields. They belong to this implementation of `DoubleSpring`.

## 3. Connect the first spring

Parent values are connected to the first child independently:

```python
@equation
def first_stiffness_connection(
    first_stiffness: Stiffness,
    first: Spring,
) -> bool:
    return first_stiffness == first.stiffness


@equation
def first_extension_connection(
    first_extension: Length,
    first: Spring,
) -> bool:
    return first_extension == first.extension


@equation
def first_force_connection(
    force: Force,
    first: Spring,
) -> bool:
    return force == first.force
```

Conceptually:

```text
parent stiffness  ↔  first.stiffness
parent extension  ↔  first.extension
parent force      ↔  first.force
```

Each equation touches only one child system.

## 4. Connect the second spring

The second spring uses the same pattern:

```python
@equation
def second_stiffness_connection(
    second_stiffness: Stiffness,
    second: Spring,
) -> bool:
    return second_stiffness == second.stiffness


@equation
def second_extension_connection(
    second_extension: Length,
    second: Spring,
) -> bool:
    return second_extension == second.extension


@equation
def second_force_connection(
    force: Force,
    second: Spring,
) -> bool:
    return force == second.force
```

The same parent `force` value is connected independently to each spring.

```mermaid
flowchart LR
    force["Series force F"] --> s1["Spring 1: F = k₁ x₁"]
    force --> s2["Spring 2: F = k₂ x₂"]

    s1 --> x1["Extension x₁"]
    s2 --> x2["Extension x₂"]

    x1 --> sum["x_total = x₁ + x₂"]
    x2 --> sum
    sum --> total["Total extension"]
```

This diagram mirrors the mechanics: one common series force acts through both springs, while the two extensions are added on the parent system.

No equation directly contains both spring subsystems.

## 5. Add the extensions on the parent

The total extension is an ordinary parent equation:

```python
@equation
def total_extension_relation(
    first_extension: Length,
    second_extension: Length,
    total_extension: Length,
) -> bool:
    return total_extension == first_extension + second_extension
```

This equation is also acausal.

For example:

```text
first extension + second extension → total extension
```

but also:

```text
total extension + first extension → second extension
```

## 6. Put the implementation together

```python
double_spring = Implementation(
    DoubleSpring,
    [
        first_stiffness_connection,
        first_extension_connection,
        first_force_connection,
        second_stiffness_connection,
        second_extension_connection,
        second_force_connection,
        total_extension_relation,
    ],
    internal_implementations={
        "first": linear_spring,
        "second": linear_spring,
    },
)
```

The hierarchy is therefore:

```text
DoubleSpring
│
├── parent equations
│   └── total extension relation
│
└── internal implementations
    ├── first  → Spring
    └── second → Spring
```

For the executable documentation below, the same model is loaded from a temporary Python module. This gives Bilt a real source file for `inspect.getsource()` during the `markdown-exec` build.

```python exec="on" session="springs"
model_source = """
from typing import Annotated

from bilt import Implementation, Interface, Quantity, equation, field, system
from bilt.units import meter, newton
from bilt.units.dimensions import FORCE, LENGTH


Stiffness = Annotated[Quantity, FORCE / LENGTH]
Length = Annotated[Quantity, LENGTH]
Force = Annotated[Quantity, FORCE]

@system
class Spring(metaclass=Interface):
    stiffness: Stiffness = field()
    extension: Length = field()
    force: Force = field()

@equation
def hookes_law(
    stiffness: Stiffness,
    extension: Length,
    force: Force,
) -> bool:
    return force == stiffness * extension

linear_spring = Implementation(
    Spring,
    [hookes_law],
)

@system
class DoubleSpring(metaclass=Interface):
    first_stiffness: Stiffness = field()
    second_stiffness: Stiffness = field()

    first_extension: Length = field()
    second_extension: Length = field()
    total_extension: Length = field()

    force: Force = field()

@equation
def first_stiffness_connection(
    first_stiffness: Stiffness,
    first: Spring,
) -> bool:
    return first_stiffness == first.stiffness

@equation
def first_extension_connection(
    first_extension: Length,
    first: Spring,
) -> bool:
    return first_extension == first.extension

@equation
def first_force_connection(
    force: Force,
    first: Spring,
) -> bool:
    return force == first.force

@equation
def second_stiffness_connection(
    second_stiffness: Stiffness,
    second: Spring,
) -> bool:
    return second_stiffness == second.stiffness

@equation
def second_extension_connection(
    second_extension: Length,
    second: Spring,
) -> bool:
    return second_extension == second.extension

@equation
def second_force_connection(
    force: Force,
    second: Spring,
) -> bool:
    return force == second.force

@equation
def total_extension_relation(
    first_extension: Length,
    second_extension: Length,
    total_extension: Length,
) -> bool:
    return total_extension == first_extension + second_extension

double_spring = Implementation(
    DoubleSpring,
    [
        first_stiffness_connection,
        first_extension_connection,
        first_force_connection,
        second_stiffness_connection,
        second_extension_connection,
        second_force_connection,
        total_extension_relation,
    ],
    internal_implementations={
        "first": linear_spring,
        "second": linear_spring,
    },
)
"""

model_path = Path(tempfile.gettempdir()) / "bilt_two_springs.py"
model_path.write_text(model_source)

spec = importlib.util.spec_from_file_location(
    "_bilt_two_springs",
    model_path,
)
assert spec is not None
assert spec.loader is not None

model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

DoubleSpring = model_module.DoubleSpring
double_spring = model_module.double_spring
```

## 7. Evaluate a compatible spring state

Start by supplying the individual spring extensions and stiffnesses.

We deliberately choose compatible values:

```text
Spring 1: 100 N/m × 0.075 m = 7.5 N
Spring 2: 300 N/m × 0.025 m = 7.5 N
```

so both springs carry the same series force.

```python exec="on" source="above" session="springs"
forward_case = DoubleSpring(
    first_stiffness=100.0 * newton / meter,
    second_stiffness=300.0 * newton / meter,
    first_extension=0.075 * meter,
    second_extension=0.025 * meter,
)

with double_spring:
    force = forward_case.force
    total_extension = forward_case.total_extension

print(f"Force:           {force.convert_to(newton).value:.3f} N")
print(f"Total extension: {total_extension.convert_to(meter).value:.3f} m")
```

The result is:

```text
Force:           7.500 N
Total extension: 0.100 m
```

The force comes from a child spring equation, while total extension comes from the parent equation.

### Visualise the operating point

```python exec="on" html="on" source="above" session="springs"
extension_m = np.linspace(0.0, 0.10, 100)
first_stiffness_n_per_m = forward_case.first_stiffness.convert_to(newton / meter).value
second_stiffness_n_per_m = forward_case.second_stiffness.convert_to(newton / meter).value

plt.plot(
    extension_m,
    first_stiffness_n_per_m * extension_m,
    label=f"Spring 1: k = {first_stiffness_n_per_m:g} N/m",
)
plt.plot(
    extension_m,
    second_stiffness_n_per_m * extension_m,
    label=f"Spring 2: k = {second_stiffness_n_per_m:g} N/m",
)

plt.scatter(
    [forward_case.first_extension.convert_to(meter).value],
    [force.convert_to(newton).value],
)
plt.scatter(
    [forward_case.second_extension.convert_to(meter).value],
    [force.convert_to(newton).value],
)

plt.xlabel("Extension (m)")
plt.ylabel("Force (N)")
plt.title("Two spring force-extension relationships")
plt.legend()
render_plot()
```

The plot uses numerical values explicitly converted to N/m, m, and N, so
they match the labels. The same force intersects the two spring curves at
different extensions because their stiffnesses differ.

## 8. Use the same model in another direction

Now supply the common force and the two stiffnesses, but **do not** supply either spring extension:

```python exec="on" source="above" session="springs"
reverse_case = DoubleSpring(
    first_stiffness=100.0 * newton / meter,
    second_stiffness=300.0 * newton / meter,
    force=7.5 * newton,
)

with double_spring:
    first_extension = reverse_case.first_extension
    second_extension = reverse_case.second_extension
    total_extension = reverse_case.total_extension

print(f"First extension:  {first_extension.convert_to(meter).value:.3f} m")
print(f"Second extension: {second_extension.convert_to(meter).value:.3f} m")
print(f"Total extension:  {total_extension.convert_to(meter).value:.3f} m")
```

Bilt can use Hooke's law in the opposite direction inside each child:

```text
known force + stiffness
        ↓
solve spring extension
```

The parent can then add the two extensions:

```text
first extension + second extension
                ↓
          total extension
```

So the complete route is:

```mermaid
flowchart LR
    force["Known series force F"] --> spring1["Spring 1: known k₁"]
    force --> spring2["Spring 2: known k₂"]

    spring1 --> x1["Solve x₁ = F / k₁"]
    spring2 --> x2["Solve x₂ = F / k₂"]

    x1 --> total["Add extensions: x_total = x₁ + x₂"]
    x2 --> total
```

The model is still describing the same two springs in series. Bilt is simply using Hooke's law in the opposite direction inside each spring.

The component equations have not changed. Only the known values and requested values have.

## 9. A request that needs a coupled solve

A tempting query is:

```text
known:
    total extension
    first stiffness
    second stiffness

requested:
    common force
```

For two springs in series, that requires Bilt to determine the shared force and both unknown spring extensions together.

That is a genuinely coupled solve across the two child systems.

The calculations above deliberately use decomposable requests, so they also work with the default exhaustive analyser.

The SCC analyser on the coupled-solve feature branch can support this linear coupled formulation. See [Coupled linear systems](../guide/coupled-linear-systems.md) for its current availability and a runnable introduction, or the [recuperated gas turbine](two-shaft-gas-turbine.md#14-close-the-loop-with-recuperation) for a larger feedback example.

## 10. Explore the model yourself

Try changing the common force:

```python
force = 10.0 * newton
```

or either stiffness:

```python
first_stiffness = 150.0 * newton / meter
second_stiffness = 250.0 * newton / meter
```

Then ask again for:

```python
spring_system = DoubleSpring(
    first_stiffness=first_stiffness,
    second_stiffness=second_stiffness,
    force=force,
)

with double_spring:
    total_extension = spring_system.total_extension
```

You can also go back to the forward form and provide the individual extensions instead of the force.

To explore locally, edit the values while `zensical serve` is running and save the Markdown file. Zensical will rebuild the page and rerun the executable blocks.

## 11. What this example demonstrates

The spring example separates three ideas clearly.

### Reusable component physics

```text
Spring
    force = stiffness × extension
```

One implementation is reused twice.

### Systems-of-systems composition

Each child connects independently to parent values:

```text
parent value
    ↕
child boundary
```

and the parent owns the relationship between the two extensions:

```text
total extension = first extension + second extension
```

### Acausal evaluation

The same Hooke's-law equation supports both:

```text
stiffness + extension → force
```

and:

```text
force + stiffness → extension
```

No transforms or manually inverted equations are needed.

This is a small example of the same pattern used by the larger two-shaft turbomachinery model.
