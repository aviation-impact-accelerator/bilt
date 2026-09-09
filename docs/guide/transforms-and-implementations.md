## Transforms and implementations

In the previous chapter, we used an equation to describe a mathematical relationship:

```python
@equation
def kinematic_relation(distance: float, speed: float, time: float) -> bool:
    return distance == speed * time
```

Equations are useful when the relationship itself is important and Bilt may need to use it in different directions.

Not every part of a model works like that.

Sometimes we simply want to say:

> Given these values, here is how to calculate another one.

For that, Bilt provides **transforms**.

### Transforms describe calculations

Consider a small system:

```python
from bilt import Interface, field, system


@system
class Demo(metaclass=Interface):
    base: float = field()
    increment: float = field()
    total: float = field()
```

We can define a transform that calculates `total` from `base` and `increment`:

```python
from bilt import transform


@transform
def total(base: float, increment: float) -> float:
    return base + increment
```

This tells Bilt that `total` can be calculated from `base` and `increment`:

```text
base ──────┐
           ├──► total
increment ─┘
```

Unlike an equation, a transform has a fixed computational direction.

The parameter names identify the values it requires, while the function name identifies the value it provides.

So:

```python
@transform
def total(base: float, increment: float) -> float:
    return base + increment
```

means:

```text
requires: base, increment
provides: total
```

Bilt can use this transform to calculate `total` from `base` and `increment`. It cannot use the same transform to calculate `base` from `increment` and `total`.

### Transforms are ordinary calculations

A transform contains ordinary Python code.

For example:

```python
@transform
def magnitude(value: float) -> float:
    return abs(value)
```

Given `value`, we can calculate its magnitude. But knowing the magnitude does not tell us whether the original value was positive or negative.

There is no unique way to run the calculation backwards.

Transforms can also contain conditionals, lookups, calls to other software, or other computations that cannot sensibly be represented as an algebraic relationship.

This is the main distinction between the two relationship types we have seen:

> An equation describes a relationship between values.

> A transform describes how to calculate one value from others.

Use an equation when the mathematical relationship is the important thing. Use a transform when the computation is the important thing.

### Implementations collect relationships for a system

We used an implementation in the previous chapter to connect an equation to a system.

Now we can look at its role more closely.

```python
from bilt import Implementation


basic = Implementation(Demo, [total])
```

A system and its implementation have different jobs.

`Demo` describes the boundary:

```text
Demo
├── base
├── increment
└── total
```

while `basic` supplies relationships that can be used to obtain those values:

```text
base, increment ──► total
```

An implementation can contain equations, transforms, or a mixture of both.

Keeping the system boundary separate from its implementation also allows the same system to be realised in different ways.

### Evaluating with an implementation

Create an instance with the values we already know:

```python
demo = Demo(
    base=1.0,
    increment=2.0,
)
```

Then make the implementation active while requesting `total`:

```python
with basic:
    assert demo.total == 3.0
```

Bilt knows:

```text
base = 1.0
increment = 2.0
```

and has been asked for:

```text
total
```

The active implementation contains a transform that can produce `total` from the available values, so the request can be satisfied.

Conceptually:

```text
known

base, increment
      │
      ▼
total(base, increment)
      │
      ▼
    total
```

The important point is that we did not call the transform ourselves:

```python
total(1.0, 2.0)
```

Instead, we asked the system for:

```python
demo.total
```

Bilt found a relationship that could provide it.

### Implementations can contain many relationships

A real implementation will usually contain more than one relationship.

Let's extend our system:

```python
@system
class Demo(metaclass=Interface):
    base: float = field()
    increment: float = field()
    total: float = field()
    adjusted: float = field()
    result: float = field()
```

and define three transforms:

```python
@transform
def total(base: float, increment: float) -> float:
    return base + increment


@transform
def adjusted(base: float, total: float) -> float:
    return base + total


@transform
def result(total: float, adjusted: float) -> float:
    return total + adjusted
```

We can collect them into one implementation:

```python
basic = Implementation(Demo, [total, adjusted, result])
```

This gives Bilt three available relationships:

```text
base, increment ──► total
base, total ──► adjusted
total, adjusted ──► result
```

Now create an instance with just `base` and `increment` known:

```python
demo = Demo(
    base=1.0,
    increment=2.0,
)
```

and ask for `result`:

```python
with basic:
    assert demo.result == 7.0
```

To satisfy that request, Bilt needs to obtain:

```text
total = 3
adjusted = 4
result = 7
```

or, as a graph:

```text
base ──────┐
           ├──► total ─────┐
increment ─┘               ├──► result
base ─────────► adjusted ──┘
                 ▲
                 │
               total
```

We never wrote:

```python
total_value = total(base, increment)
adjusted_value = adjusted(base, total_value)
result_value = result(total_value, adjusted_value)
```

The implementation describes which relationships are available. It does not prescribe the order in which they must be evaluated.

### The model is not the execution order

Nor does the order in:

```python
Implementation(Demo, [total, adjusted, result])
```

mean:

```text
first total
then adjusted
then result
```

Those relationships describe the model.

The required execution order depends on which values are already known and which value we request.

If we request `total`, Bilt only needs:

```text
base, increment ──► total
```

If we request `result`, it needs a longer route:

```text
base, increment ──► total
base, total ──► adjusted
total, adjusted ──► result
```

So:

> The model describes the available relationships. The request determines the computation.

This applies equally when an implementation contains equations, transforms, or a mixture of both.

### Equations and transforms can work together

An implementation does not have to choose between equations and transforms.

For example, part of a model might be described by an equation:

```text
distance = speed × time
```

while another part is naturally a directed calculation:

```text
distance ──► journey_category
```

Bilt can use both kinds of relationship when working out how to obtain a requested value.

Equations give it relationships that may be usable in different directions.

Transforms give it relationships with a fixed direction.

Together, they form the set of possible routes through the model.

The next question is how Bilt chooses a route from the values that are known to the value that has been requested.

That is the subject of the next chapter: **from models to computation**.

[← Equations and acausal modelling](equations-and-acausal-modelling.md) · [Index](../index.md) · [Models to computation →](models-to-computation.md)
