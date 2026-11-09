# Quantity annotations and typing

This page describes how to annotate quantities and tensors in Bilt systems and equations.
For unit conversion and arithmetic, see the [Units API](units.md). For named axes,
reductions and selections, see the [Tensor API](tensors.md).

## Choose the annotation for the job

A **system** declares which values its fields can accept. An **equation** describes
a mathematical relationship. Pointwise equations can use the same scalar-or-tensor
quantity annotation as their system fields: Bilt builds symbolic expressions and
lifts the relationship over compatible domains.

| Value | System field (when either shape is allowed) | Equation parameter |
| --- | --- | --- |
| Plain scalar or unitless tensor | `float | Tensor` | `float` for scalar-local arithmetic; `Tensor` for named tensor methods |
| Pointwise quantity with a physical unit | `Quantity \| QuantityTensor` | Same annotation for pointwise arithmetic |
| Tensor quantity with a physical unit | `QuantityTensor` | `QuantityTensor` for named tensor methods |
| Strictly scalar quantity with a physical unit | `QuantityLike[float]` | `Quantity` for scalar-only operations |

For example, a system may accept either a scalar or a tensor of mass:

```python
from typing import Annotated

from bilt import Implementation, Interface, Quantity, QuantityTensor, dim, equation, field, system, tensor
from bilt.units import kilogram
from bilt.units.dimensions import MASS

type Mass = Annotated[Quantity | QuantityTensor, MASS]


@system
class DoubleMass(metaclass=Interface):
    mass: Mass = field()
    doubled: Mass = field()


@equation
def doubling(mass: Mass, doubled: Mass) -> bool:
    return doubled == mass * 2.0


time = dim("time")

with Implementation(DoubleMass, [doubling]):
    scalar_result = DoubleMass(mass=2.0 * kilogram).doubled
    tensor_result = DoubleMass(mass=tensor([2.0, 3.0], over=time, unit=kilogram)).doubled
```

The equation is pointwise, so Bilt automatically lifts it when the input
varies over `time`. Its Python annotations describe the scalar-or-tensor
values accepted at the boundary; Bilt replaces the arguments with symbolic
expressions while constructing the equation.

## Available types

| Annotation | Type checker understands | Runtime interpretation |
| --- | --- | --- |
| `Quantity \| QuantityTensor` | Unit-bearing scalar **or** tensor, with pointwise operators | Flexible pointwise quantity boundary and equation |
| `QuantityLike` | Unit-bearing scalar **or** tensor, exposing only `.unit` and `.value` | Read-only quantity boundary |
| `QuantityLike[float]` | Scalar magnitude | Strict scalar quantity at a system boundary; scalar-local in an equation |
| `QuantityLike[Tensor]` | Tensor magnitude | Tensor quantity at a system boundary; tensor expression in an equation |
| `Quantity` | Concrete scalar `Quantity`, with its arithmetic methods | Scalar quantity arithmetic; lifted inside equations |
| `QuantityTensor` | Concrete unit-bearing tensor, including tensor methods | Tensor quantity with a non-optional unit |
| `Tensor` | Tensor with or without a unit | Plain `Tensor` fields remain outside the unit system unless given dimensionality metadata |
| `float | Tensor` | Plain scalar or tensor | Unitless system boundary |

`QuantityLike[M]` is a small read-only structural interface, not a second
quantity implementation. Use `Quantity | QuantityTensor` for pointwise equations
when scalar and tensor values are both valid. Use concrete `Quantity` for
scalar-only code, and `QuantityTensor` when named tensor operations are required.

A unit-bearing tensor can be constructed with `tensor(..., unit=kilogram)`.
The factory returns `QuantityTensor` when a non-optional `Unit` is supplied
and returns `Tensor` if the unit is omitted. If the unit argument has
type `Unit | None`, the conservative static return type is `Tensor`
although the runtime object will be a `QuantityTensor` when the unit is present.

## Named tensor operations in equations

Annotate an equation parameter as `QuantityTensor` when the equation
uses `sum`, `mean`, `isel`, `sel`, or other named tensor methods.

```python
from typing import Annotated

from bilt import Quantity, QuantityTensor, dim, equation
from bilt.units.dimensions import MASS

time = dim("time")
type ScalarMass = Annotated[Quantity, MASS]
type TensorMass = Annotated[QuantityTensor, MASS]


@equation
def total_mass(masses: TensorMass, total: ScalarMass) -> bool:
    return total == masses.sum(time)
```

A tensor with domain `(scenario, time)` can be reduced over `time`;
the result retains `scenario`. The compiler determines the resulting
domain and physical dimension. The concrete `Tensor` method return
annotations do not encode the remaining domain statically.

## Minimum domain requirements

A dimensioned tensor quantity port may require one or more named axes:

```python
from typing import Annotated

from bilt import QuantityTensor, dim
from bilt.symbolic._domains import Domain
from bilt.units.dimensions import MASS

time = dim("time")
type TimeMass = Annotated[QuantityTensor, MASS, Domain((time,))]
```

This declares a **minimum** domain, not an exact shape:

| Supplied or inferred domain | Meets `Domain((time,))`? |
| --- | --- |
| `(time,)` | Yes |
| `(scenario, time)` | Yes |
| `(time, scenario)` | Yes |
| `(scenario,)` | No |
| `()` | No |

Minimum-domain checking applies to the new `QuantityTensor` and
`QuantityLike[Tensor]` annotations that also carry a physical
dimension. At a system boundary, Bilt checks supplied and computed values.
Within an implementation, Bilt checks the requirements on **selected**
equations against the inferred domains of the values those equations use
or produce. Requirements on equations that are not selected do not
constrain that evaluation.

`Annotated[Quantity, DIM, Domain(...)]` and
`Annotated[Quantity | QuantityTensor, DIM, Domain(...)]` remain **exact equation**
domain hints for structural/linear-block analysis. It is not a system-field
annotation, and is not reinterpreted as a minimum-domain requirement.
Dimensioned `Annotated[Tensor, DIM]` declarations are no longer supported;
use `QuantityTensor` for tensor equations and `QuantityLike[Tensor]` for
system boundaries.
Domain constraints in `Annotated` are validated by Bilt, not statically
interpreted by ty or Pyrefly.

## Limitations

- The pointwise union is an **equation annotation**, not permission to use eager
  tensor arithmetic. Python's type checkers may accept `value * 2.0` for a
  union-typed quantity outside `@equation`, but concrete `Tensor` arithmetic
  deliberately raises at runtime. The checker cannot infer Bilt's symbolic
  execution context from the decorator alone. Avoid eager tensor arithmetic
  outside a Bilt equation.

`Quantity(value, unit)` remains the concrete scalar quantity constructor.
For dimensioned pointwise system fields, use `Quantity | QuantityTensor`.
Strict scalar boundaries can still use `QuantityLike[float]`. Dimensioned
`Annotated[Quantity, DIM]` and `Annotated[Tensor, DIM]` are rejected at
system boundaries; equations and transforms may use concrete `Quantity`
when scalar-only operations are intended.

- A `float | Tensor` system field accepts scalar or unitless tensor
  values. `QuantityTensor` subclasses `Tensor`, so ty and Pyrefly
  cannot reject unit-bearing tensors for this union; Bilt rejects those
  values at runtime. A separate tensor-typing design would be needed to
  express unitless tensors statically.
- Static type checkers do not prove physical dimensions, unit categories,
  named axis sets, or Bilt's structural execution choices. These are
  Bilt runtime and compiler checks.
- Bilt retains the existing distinction between symbolic tensor methods
  inside equations and concrete tensor operations outside a model. The
  latter do not offer general eager tensor arithmetic.

Use the [Domains and tensors guide](../guide/dimensions-domains-and-tensors.md)
for broadcasting, axis alignment and examples of domain inference.
