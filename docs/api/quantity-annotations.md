# Quantity annotations and typing

This is the reference for quantity annotations on Bilt system fields, equations
and transforms. For unit definitions, conversions and arithmetic, see the
[Units API](units.md); for a guided introduction, see [Units in models](../guide/units.md).
For named axes, reductions and selections, see the [Tensor API](tensors.md).

## Choose the annotation for the job

A **system** declares which values its fields can accept. An **equation** describes
a mathematical relationship. A system field can accept either a scalar or a
tensor quantity, while a scalar-annotated equation can still operate pointwise
over tensor inputs: Bilt builds symbolic expressions and lifts the relationship
over compatible domains.

| Value | System field (when either shape is allowed) | Equation parameter |
| --- | --- | --- |
| Plain scalar or unitless tensor | `float | Tensor` | `float` for scalar-local arithmetic; `Tensor` for named tensor methods |
| Pointwise quantity with a physical unit | `Quantity \| QuantityTensor` | `Quantity` for scalar-local arithmetic (or the union for pointwise operators) |
| Tensor quantity with a physical unit | `QuantityTensor` | `QuantityTensor` for named tensor methods |
| Strictly scalar quantity with a physical unit | `Quantity` | `Quantity` for scalar arithmetic |

For example, a system may accept either a scalar or a tensor of mass. Named
aliases keep the system and equation declarations readable:

```python exec="off"
from typing import Annotated

from bilt import Implementation, Interface, Quantity, QuantityTensor, dim, equation, field, system, tensor
from bilt.units import kilogram
from bilt.units.dimensions import MASS

type Mass = Annotated[Quantity, MASS]
type PointwiseMass = Annotated[Quantity | QuantityTensor, MASS]


@system
class DoubleMass(metaclass=Interface):
    mass: PointwiseMass = field()
    doubled: PointwiseMass = field()


@equation
def doubling(mass: Mass, doubled: Mass) -> bool:
    return doubled == mass * 2.0


time = dim("time")

with Implementation(DoubleMass, [doubling]):
    scalar_result = DoubleMass(mass=2.0 * kilogram).doubled
    tensor_result = DoubleMass(mass=tensor([2.0, 3.0], over=time, unit=kilogram)).doubled
```

The equation is pointwise, so Bilt automatically lifts its scalar `Mass` annotation
when an input varies over `time`. The system's `PointwiseMass` annotation accepts
either shape; Bilt builds symbolic expressions when constructing the equation.

## Available types

| Annotation | Type checker understands | Runtime interpretation |
| --- | --- | --- |
| `Quantity \| QuantityTensor` | Unit-bearing scalar **or** tensor, with pointwise operators | Flexible pointwise quantity boundary and equation |
| `Quantity` | Concrete scalar `Quantity`, with its arithmetic methods | Scalar quantity arithmetic; lifted inside equations |
| `QuantityTensor` | Concrete unit-bearing tensor, including tensor methods | Tensor quantity with a non-optional unit |
| `Tensor` | Tensor with or without a unit | Plain `Tensor` system fields are unitless; use `QuantityTensor` for dimensioned tensors |
| `float | Tensor` | Plain scalar or tensor | Unitless system boundary |

Use `Quantity | QuantityTensor` for pointwise quantities that may be scalar or tensor. Use `Quantity` for scalar-only inputs and operations, and `QuantityTensor` when named tensor methods are required. In longer models, use paired aliases such as `Mass = Annotated[Quantity, MASS]` and `PointwiseMass = Annotated[Quantity | QuantityTensor, MASS]` instead of repeating the full annotations.

A unit-bearing tensor can be constructed with `tensor(..., unit=kilogram)`.
The factory returns `QuantityTensor` when a non-optional `Unit` is supplied
and returns `Tensor` if the unit is omitted. If the unit argument has
type `Unit | None`, the conservative static return type is `Tensor`
although the runtime object will be a `QuantityTensor` when the unit is present.

## Named tensor operations in equations

Annotate an equation parameter as `QuantityTensor` when the equation
uses `sum`, `mean`, `isel`, `sel`, or other named tensor methods.

```python exec="off"
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

```python exec="off"
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

Minimum-domain checking applies to `QuantityTensor` annotations carrying a physical dimension. At a system boundary, Bilt checks supplied and computed values.
Within an implementation, Bilt checks the requirements on **selected**
equations against the inferred domains of the values those equations use
or produce. Requirements on equations that are not selected do not
constrain that evaluation.

`Annotated[Quantity, DIM, Domain(...)]` and
`Annotated[Quantity | QuantityTensor, DIM, Domain(...)]` remain **exact equation**
domain hints for structural/linear-block analysis. It is not a system-field
annotation, and is not reinterpreted as a minimum-domain requirement.
Dimensioned `Annotated[Tensor, DIM]` declarations are no longer supported;
use `QuantityTensor` for tensor-only fields and equations.
Domain constraints in `Annotated` are validated by Bilt, not statically
interpreted by ty or Pyrefly.

## Symbolic and eager arithmetic

`Quantity | QuantityTensor` is useful both for a flexible **system field**
and for a **pointwise equation**. An `@equation` builds a symbolic expression
tree: pointwise operators are then lifted across compatible scalar and tensor
domains. It does **not** execute the concrete `QuantityTensor` methods.

Outside equations, concrete `Quantity` objects support scalar arithmetic.
Concrete `Tensor` and `QuantityTensor` objects deliberately do not support
general eager arithmetic or named reductions. Arithmetic raises
`NotImplementedError`, while symbolic-only tensor methods raise `TypeError`.
`@transform` is ordinary executable Python and must not use these as eager
numerical operations.

Python's type checkers cannot infer symbolic execution from `@equation`.
They can accept `value * 2.0` where `value` has the union type, **even
outside an equation**. This is a documented limitation of the static type
contract, not evidence that eager tensor multiplication is supported. A
future separate equation checker might detect this, but isn't part of the
quantity typing migration.

`Quantity(value, unit)` remains the concrete scalar constructor.
For dimensioned pointwise fields, use `Quantity | QuantityTensor`.
Strict scalar fields use `Quantity`, and tensor-only fields use `QuantityTensor`.
Dimensioned `Annotated[Tensor, DIM]` declarations are rejected at system
boundaries; scalar-only equations and transforms may also use concrete `Quantity`.

## Other limitations

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
