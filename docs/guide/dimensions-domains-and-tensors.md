## Domains and tensors

For constructors, reductions, indexing and their current limitations, see the [Tensor API reference](../api/tensors.md). For quantity-related type annotations, see [Quantity annotations and typing](../api/quantity-annotations.md).

So far, every value in our examples has been a scalar:

```python
speed: float
temperature: float
pressure: float
```

Real engineering models quickly need something more.

A temperature may vary with time. Demand may vary with location. A quantity may vary over both:

```text
temperature[time]

demand[location, time]
```

Bilt represents these values using **tensors** over named **dimensions**.

### Dimensions give axes names

In NumPy, an array has a shape:

```python
values.shape
```

which might give:

```text
(2, 3)
```

That tells us how many elements lie along each axis, but not what those axes mean.

Perhaps `(2, 3)` means:

```text
(location, time)
```

or:

```text
(engine, operating_point)
```

Those are very different things.

In Bilt, we can give dimensions explicit names:

```python
from bilt import dim


location = dim("location")
time = dim("time")
```

A dimension identifies one direction in which a value may vary.

We can then describe a domain containing those dimensions:

```text
(location, time)
```

rather than relying only on positional axes:

```text
(2, 3)
```

The distinction becomes important when different values vary over different sets of dimensions.

### Tensors carry values over a domain

A Bilt **tensor** combines concrete values with the dimensions they vary over.

For example:

```python
from bilt import tensor


demand = tensor(
    [
        [10.0, 12.0, 14.0],
        [8.0, 9.0, 11.0],
    ],
    over=(location, time),
)
```

The underlying values have shape:

```text
(2, 3)
```

and the domain is:

```text
(location, time)
```

So we can think of the tensor as:

```text
                 time
             ───────────►

location      10  12  14
   │           8   9  11
   ▼
```

The numbers still live in an ordinary array. The domain tells Bilt what each axis represents.

### Scalars have a domain too

A scalar is simply a value that does not vary over any dimensions.

Its domain is empty:

```text
()
```

So these values can be viewed consistently:

```text
efficiency                ()
price[time]               (time)
demand[location, time]    (location, time)
```

This gives Bilt one vocabulary for scalar and non-scalar values rather than treating arrays as a completely separate modelling system.

### Named dimensions make broadcasting explicit

Suppose demand varies by location and time:

```text
demand : (location, time)
```

but electricity price varies only with time:

```text
price : (time)
```

If we calculate:

```text
cost = demand × price
```

the price for each time can be applied across every location.

Conceptually:

```text
demand    (location, time)
price               (time)
                     │
                     ▼
cost      (location, time)
```

Bilt can align the values using the **names of their dimensions**, rather than requiring the user to arrange axes purely by position.

### Scalar or tensor values at system boundaries

When a system can accept either a plain scalar or a **unitless** tensor,
a union is sufficient. There is no need for an additional Bilt protocol:

```python
from bilt import Interface, Tensor, field, system

type Value = float | Tensor


@system
class Scale(metaclass=Interface):
    x: Value = field()
    doubled: Value = field()
```

Both `Scale(x=2.0)` and `Scale(x=tensor([1.0, 2.0], over=time))`
are accepted. An equation can still use `float` annotations and Bilt will
lift it over the input's named domain. This keeps the system's accepted
value types separate from the local operations of the equation.

**Limit of static typing:** The concrete `QuantityTensor` class currently
inherits `Tensor`. Python type checkers therefore also consider a
unit-bearing tensor valid wherever a `Tensor` is annotated. Bilt rejects
it **at runtime** at a plain, unitless port. A static guarantee that the
tensor has no unit would require a separate tensor typing design; a
`float | Tensor` union cannot provide that guarantee.

### Requiring a minimum domain

A system input can require a particular named dimension without fixing the
complete shape of the input. For a unit-bearing tensor quantity:

```python
import typing

from bilt import Interface, QuantityLike, Tensor, dim, field, system
from bilt.symbolic._domains import Domain
from bilt.units.dimensions import MASS

time = dim("time")
scenario = dim("scenario")
type TimeMass = typing.Annotated[QuantityLike[Tensor], MASS, Domain((time,))]


@system
class Demand(metaclass=Interface):
    mass: TimeMass = field()
```

A tensor over `(time,)` satisfies that requirement, as does one over
`(scenario, time)`. A tensor over only `(scenario,)` does not: it lacks
the required `time` dimension. The order of axes does not matter.

This is a **minimum** domain requirement, not an exact-domain restriction.
Bilt checks it when values enter a system and when a computed system output
has a declared minimum domain. Domain metadata is checked by Bilt at runtime;
Python type checkers do not interpret the dimensions in `Annotated`.

For now, this form applies to **new unit-bearing tensor annotations**
(`QuantityTensor` and `QuantityLike[Tensor]`). Bare
`QuantityLike` permits scalar values and therefore cannot also declare a
required tensor dimension. Existing unitless `Annotated[Tensor, Domain(...)]`
annotations and existing `Quantity` or `Tensor` annotations with Domain metadata
remain available as hints for structural domain inference; this new
minimum-domain check does not reinterpret their existing behaviour.
Equation-local minimum requirements are checked on the route Bilt selects,
as described below.

### Minimum domains on equation parameters

An equation can also require a dimension that is not mentioned on its
enclosing system field. This is useful when the system itself accepts many
possible shapes but one equation needs a specific axis:

```python
import typing

from bilt import Quantity, QuantityLike, QuantityTensor, equation
from bilt.symbolic._domains import Domain
from bilt.units.dimensions import MASS

type Mass = typing.Annotated[QuantityLike, MASS]
type ScalarMass = typing.Annotated[Quantity, MASS]
type TimeMass = typing.Annotated[QuantityTensor, MASS, Domain((time,))]


@equation
def copy_over_time(x: TimeMass, result: ScalarMass) -> bool:
    return result == x
```

When Bilt chooses this equation to compute an output, it checks the
**inferred domains** of the values the equation reads and produces.
The `time` dimension must be present, but additional axes such as
`scenario` are allowed. If the equation is not selected for a particular
evaluation, its domain requirement does not constrain that evaluation.

These are minimum-domain **requirements** rather than the existing exact
`Domain` **hints** on legacy `Quantity` and `Tensor` annotations. In
particular, they are not passed to the linear-block inference algorithm as
exact layouts. A mismatch rejects the selected evaluation route with an
explanation naming the equation, port and missing dimensions.

### Broadcasting rules

Bilt broadcasting follows dimension names rather than axis positions.

A few simple rules cover most cases:

```text
scalar                  ()
time value              (time)
location value          (location)
spatial time value      (location, time)
```

A scalar can be combined with any domain:

```text
() + (time)
    → (time)
```

If one value is missing a dimension, it is broadcast across that dimension:

```text
(time) + (location, time)
    → (location, time)
```

Partially overlapping domains are combined:

```text
(location, time) + (time, scenario)
    → (location, time, scenario)
```

Shared dimensions are aligned by name. Their original axis positions do not need to match.

There is one important restriction: if two values share a dimension, they must agree about that dimension during the evaluation. Bilt will not silently treat two incompatible definitions of `time`, for example, as though they were the same axis.

For most model code, this means thinking about **which dimensions a value varies over**, rather than manually arranging array axes to make broadcasting work.

<!-- TODO: Link to the detailed domain and broadcasting reference. -->

### The relationship can stay simple

One useful consequence is that the mathematical relationship does not necessarily need to know about every surrounding dimension.

Imagine a calculation:

```python
@transform
def cost(demand: float, price: float) -> float:
    return demand * price
```

The relationship itself is simple:

```text
cost = demand × price
```

If `demand` and `price` are evaluated over compatible domains, Bilt can handle the domain alignment around that calculation.

This lets the model express the engineering relationship without filling every transform with array-axis bookkeeping.

The dimensions belong to the values.

The transform describes what to do with them.

### Dimensions also have extents

Knowing that a tensor varies over `time` is not enough to store concrete data.

Bilt also needs to know how large that dimension is.

For example:

```text
time     → 3 values
location → 2 values
```

A tensor over:

```text
(location, time)
```

therefore has shape:

```text
(2, 3)
```

During an evaluation, Bilt keeps track of the concrete dimensions being used and checks that values which share a dimension agree about its extent.

If one value says:

```text
time → 3
```

while another says:

```text
time → 5
```

they cannot silently represent the same `time` dimension in that evaluation.

Catching that mismatch early is preferable to letting incompatible arrays make their way further into the model.

### Coordinates give dimensions concrete meaning

Sometimes an extent is all we need.

Often we also care which coordinate each position represents.

For a time dimension, for example, we might have:

```text
time
├── 2026-01-01
├── 2026-02-01
└── 2026-03-01
```

Bilt represents this with a **coordinate axis** associated with the dimension.

For example:

```python
from datetime import datetime

from bilt import coordinate_axis


time_axis = coordinate_axis(
    time,
    (
        datetime(2026, 1, 1),
        datetime(2026, 2, 1),
        datetime(2026, 3, 1),
    ),
)
```

The dimension still identifies the axis:

```text
time
```

while the coordinate axis tells us what the individual positions along it mean.

This separation is useful because many values can share the same dimension and coordinate system.

### Shared dimensions need to agree

Suppose two values both vary over `time`:

```text
temperature : (time)
pressure    : (time)
```

If they participate in the same evaluation, `time` should mean the same thing for both.

It would be dangerous if one secretly represented:

```text
January, February, March
```

while the other represented:

```text
1 s, 2 s, 3 s
```

simply because both arrays happened to contain three elements.

Bilt therefore tracks dimension information during evaluation so that values sharing a named dimension also share a compatible concrete definition of that dimension.

This is one of the main reasons Bilt uses named domains rather than treating tensors as bare arrays.

### Domains describe variation, not meaning

There is a useful boundary to keep in mind.

A domain tells Bilt **where a value varies**:

```text
(location, time)
```

It does not tell Bilt what the value itself means.

These two tensors might share exactly the same domain:

```text
temperature : (location, time)
pressure    : (location, time)
```

but they remain different model values.

Likewise, the same dimension may be shared by many values:

```text
temperature[time]
pressure[time]
mass_flow[time]
```

The domain describes their structure.

The system and its relationships describe their role in the model.

### Domains become part of planning

In the previous chapter, structural analysis asked whether a requested value could be reached from the values already known.

With tensors, there is another question:

> Are the values on that route compatible over their domains?

A structurally plausible calculation might require:

```text
a : (time)
b : (location)
```

and produce something over:

```text
(location, time)
```

That may be perfectly valid.

Another route might contain values that disagree about the concrete extent or coordinates of a shared dimension.

That route is not valid for the current evaluation.

So the path from model to execution now contains a little more information:

```text
model relationships
        │
        ▼
structural plan
        │
        ▼
domain alignment and validation
        │
        ▼
execution
```

The important point for model authors is simpler:

> Missing dimensions can broadcast. Shared dimensions align by name and must be compatible.

### From tensors to larger systems

We now have most of the basic pieces needed to describe individual Bilt models:

```text
systems          define model boundaries
values           live on those boundaries
transforms       describe directed calculations
equations        describe mathematical relationships
implementations  collect the relationships for a system
domains          describe where values vary
tensors          carry concrete values over those domains
```

The next step is to make models larger without turning them into one enormous system.

Bilt does that by allowing **systems to be composed from other systems**, while preserving the same boundaries and relationships we have been using throughout the guide.

That is the subject of the next chapter.

[← Models to computation](models-to-computation.md) · [Index](../index.md) · [Systems of systems →](systems-of-systems.md)
