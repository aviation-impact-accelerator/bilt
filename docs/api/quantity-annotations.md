# Quantity annotations and typing

This is the reference for the annotations on Bilt system fields, equations and
transforms. For unit definitions, conversions and arithmetic, see the
[Units API](units.md); for a guided introduction, see [Units in models](../guide/units.md).
For named axes, reductions and selections, see the [Tensor API](tensors.md).

## Choose the annotation for the job

A port annotation answers two independent questions:

1. **Units:** does the value carry a unit? If so, `Annotated` also states its
   physical dimension.
2. **Shape:** may it have any shape, or must it be a tensor because something
   reduces or selects along one of its axes?

|              | Any shape  | Must be a tensor |
| ------------ | ---------- | ---------------- |
| **Units**    | `Quantity` | `QuantityTensor` |
| **No units** | `Number`   | `Tensor`         |

All four are imported from `bilt`. They describe what a value can do, not which
class it is: you never construct them by name. Create a quantity by multiplying a
number by a unit (`2.0 * kilogram`), and a tensor with `tensor(...)`.

An annotation means the same thing on a system field, an equation parameter and a
transform. `Quantity` accepts a scalar or a tensor everywhere: an equation or
transform written for one value is lifted by Bilt over the domain of a tensor
input. Reach for `QuantityTensor` only when the code reduces or selects along an
axis.

## One alias per dimension

Write the dimension once, in an alias, and use it on systems and equations alike:

```python exec="on" session="quantity-annotations" source="above"
from typing import Annotated

from bilt import Implementation, Interface, Quantity, dim, equation, field, system, tensor
from bilt.units import kilogram
from bilt.units.dimensions import MASS

type Mass = Annotated[Quantity, MASS]


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

assert scalar_result == 4.0 * kilogram
assert tensor_result.values.tolist() == [4.0, 6.0]
```

The same `doubling` equation serves a scalar mass and a mass that varies over
`time`. It is **pointwise**: it relates values at each point of the domain
independently, so Bilt lifts it over the tensor's domain.

An alias can be written with the `type` statement, as here, or as a plain
assignment (`Mass = Annotated[Quantity, MASS]`); Bilt reads both the same way.

## What each annotation accepts

| Annotation | Accepts | Type checkers know |
| --- | --- | --- |
| `Annotated[Quantity, DIM]` | A scalar quantity or a tensor of quantities with that dimension | Quantity arithmetic, `unit`, `value`, `convert_to`, `cast_to`, `strip_metadata`, comparisons |
| `Annotated[QuantityTensor, DIM]` | A tensor of quantities with that dimension | All of the above, plus `domain`, `values`, `sum`, `mean`, `min`, `max`, `prod`, `isel`, `sel` |
| `Number` | A plain number or a unitless tensor | Plain arithmetic |
| `Tensor` | A unitless tensor | Plain arithmetic and the named tensor methods |
| `float` | A plain number; Bilt also accepts a unitless tensor, but type checkers do not | Ordinary `float` |

Prefer `Number` to `float` for a plain port that may receive a tensor. Inside an
equation or transform the two behave alike, since Bilt lifts the relationship
over a tensor input either way.

`Number` and `Tensor` are for values **outside** the unit system: counts,
flags, inputs to external code. A coefficient that takes part in quantity
arithmetic, such as a gain multiplying a length in an equation, must be a
dimensionless quantity, `Annotated[Quantity, DIMENSIONLESS]`, supplied as
`0.5 * UNITLESS`.

A `QuantityTensor` is also a `Quantity`, so it can be passed wherever a
`Quantity` is expected. Neither quantity annotation accepts a unitless value,
and neither `Number` nor `Tensor` accepts a value with a unit: type checkers
report these mistakes, and Bilt rejects them at runtime too.

`Quantity` does not distinguish a scalar from a tensor. Its `value` is therefore
a `Number`: a `float` for a scalar, a unitless `Tensor` for a tensor. A transform
that returns a quantity's magnitude is annotated `-> Number`:

```python exec="off"
from bilt import Number, transform
from bilt.units import gram


@transform
def grams(mass: Mass) -> Number:
    return mass.convert_to(gram).value
```

To get a Python `float` from a result you know is a scalar, call `float(...)`
on its value; this also works for a tensor with no dimensions.

```python exec="on" session="quantity-annotations" source="above"
assert float(scalar_result.value) == 4.0
```

## Creating values

| Expression | Annotation it satisfies |
| --- | --- |
| `2.0 * kilogram` | `Quantity` |
| `tensor([1.0, 2.0], over=time, unit=kilogram)` | `QuantityTensor` |
| `tensor([1.0, 2.0], over=time)` | `Tensor` |
| `tensor([1.0, 2.0], over=time) * kilogram` | `QuantityTensor` |

If the `unit` argument of `tensor(...)` has type `Unit | None`, the static return
type is `Tensor | QuantityTensor`.

You can check what you were given with `isinstance`, against the same names:

```python exec="on" session="quantity-annotations" source="above"
from bilt import QuantityTensor, Tensor

assert isinstance(scalar_result, Quantity)
assert not isinstance(scalar_result, QuantityTensor)
assert isinstance(tensor_result, QuantityTensor)
assert not isinstance(tensor([1.0], over=time), Quantity)
assert isinstance(tensor([1.0], over=time), Tensor)
```

`isinstance` is looser than a port: it only checks that the methods exist, so a
`QuantityTensor` also passes `isinstance(value, Tensor)`, although a `Tensor`
port refuses it. To test for a tensor of quantities, check `QuantityTensor`.

## Named tensor operations in equations

Annotate an equation parameter `QuantityTensor` when the equation uses `sum`,
`mean`, `isel`, `sel` or another named tensor method:

```python exec="on" session="quantity-annotations" source="above"
type MassOverADomain = Annotated[QuantityTensor, MASS]


@equation
def total_mass(masses: MassOverADomain, total: Mass) -> bool:
    return total == masses.sum(time)
```

The result of `masses.sum(time)` is still a `QuantityTensor` to a type checker,
which a `Mass` (any shape) accepts. A tensor with domain `(scenario, time)`
reduced over `time` keeps `scenario`; Bilt determines the remaining domain and the
physical dimension when it compiles the model. The type checker does not track
which axes remain.

## Minimum domain requirements

A tensor quantity port may require one or more named axes:

```python exec="on" session="quantity-annotations" source="above"
from bilt import Domain

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

At a system boundary, Bilt checks supplied and computed values. Within an
implementation, Bilt checks the requirements on **selected** equations against the
inferred domains of the values those equations use or produce. Requirements on
equations that are not selected do not constrain that evaluation.

On an equation parameter, `Annotated[Quantity, DIM, Domain(...)]` is an **exact**
domain hint for structural and linear-block analysis, not a minimum-domain
requirement. It is not allowed on a system field. Domain constraints are checked
by Bilt, not by type checkers.

## Where Bilt reads an annotation

Bilt reads the dimension, `ABSOLUTE` and any `Domain` from `Annotated` at the top
of a port's annotation, directly or through a type alias without type parameters.
Anything else raises `BiltPortDeclarationError` rather than silently ignoring the
dimension:

- a union such as `Annotated[Quantity, MASS] | None`;
- a subscripted generic alias such as `Length[Quantity]`, where
  `type Length[Q] = Annotated[Q, LENGTH]`;
- a dimension on a unitless annotation, such as `Annotated[Tensor, MASS]`: use
  `QuantityTensor`.

## Symbolic and eager arithmetic

An `@equation` builds a symbolic expression: Bilt calls it with symbols, not
values, and lifts pointwise operations over compatible domains. A `@transform` body
is compiled for the selected backend. In both, a parameter annotated with one of
the four types supports the operations listed above.

Outside a model, a scalar quantity supports ordinary arithmetic and comparison.
Concrete tensors deliberately do not: arithmetic on them raises
`NotImplementedError`, and named tensor methods raise `TypeError`. Type checkers
cannot tell whether code runs inside a model, so they accept
`tensor_value * 2.0` anywhere; this is a limitation of static typing, not a sign
that eager tensor arithmetic is supported.

## Earlier spellings

Earlier versions of Bilt used `Quantity | QuantityTensor` for a quantity of any
shape and `float | Tensor` for a plain value of any shape, with separate aliases
such as `PointwiseMass` for system fields. `Quantity | QuantityTensor` still means
the same as `Quantity`, and `float | Tensor` is exactly `Number`, so existing
annotations keep working. Two things changed:

- `Annotated[Quantity, DIM]` on a system field used to accept only a scalar; it now
  accepts a tensor too. There is no scalar-only quantity annotation.
- `Quantity(value, unit)` no longer constructs a quantity; write `value * unit`.

## Limitations

- Type checkers do not prove physical dimensions, unit categories, named axis sets
  or Bilt's structural execution choices. Bilt checks these when it builds and
  evaluates a model.
- A `Number` must be a `float`, an `int` or a unitless tensor; Python's other
  numeric types are not accepted.
- Bilt does not ship per-dimension aliases such as `Length`; define the ones a model
  needs.

Use the [Domains and tensors guide](../guide/dimensions-domains-and-tensors.md)
for broadcasting, axis alignment and examples of domain inference.
