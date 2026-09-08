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
    a: float = field()
    b: float = field()
    c: float = field()
```

We can define a transform that calculates `c` from `a` and `b`:

```python
from bilt import transform


@transform
def c(a: float, b: float) -> float:
    return a + b
```

This tells Bilt that `c` can be calculated from `a` and `b`:

```text
a ──┐
    ├──► c
b ──┘
```

Unlike an equation, a transform has a fixed computational direction.

The parameter names identify the values it requires, while the function name identifies the value it provides.

So:

```python
@transform
def c(a: float, b: float) -> float:
    return a + b
```

means:

```text
requires: a, b
provides: c
```

Bilt can use this transform to calculate `c` from `a` and `b`. It cannot use the same transform to calculate `a` from `b` and `c`.

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


basic = Implementation(Demo, [c])
```

A system and its implementation have different jobs.

`Demo` describes the boundary:

```text
Demo
├── a
├── b
└── c
```

while `basic` supplies relationships that can be used to obtain those values:

```text
a, b ──► c
```

An implementation can contain equations, transforms, or a mixture of both.

Keeping the system boundary separate from its implementation also allows the same system to be realised in different ways.

### Evaluating with an implementation

Create an instance with the values we already know:

```python
demo = Demo(
    a=1.0,
    b=2.0,
)
```

Then make the implementation active while requesting `c`:

```python
with basic:
    assert demo.c == 3.0
```

Bilt knows:

```text
a = 1.0
b = 2.0
```

and has been asked for:

```text
c
```

The active implementation contains a transform that can produce `c` from the available values, so the request can be satisfied.

Conceptually:

```text
known
a, b
 │
 ▼
c(a, b)
 │
 ▼
c
```

The important point is that we did not call the transform ourselves:

```python
c(1.0, 2.0)
```

Instead, we asked the system for:

```python
demo.c
```

Bilt found a relationship that could provide it.

### Implementations can contain many relationships

A real implementation will usually contain more than one relationship.

Let's extend our system:

```python
@system
class Demo(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()
    d: float = field()
    e: float = field()
```

and define three transforms:

```python
@transform
def c(a: float, b: float) -> float:
    return a + b


@transform
def d(a: float, c: float) -> float:
    return a + c


@transform
def e(c: float, d: float) -> float:
    return c + d
```

We can collect them into one implementation:

```python
basic = Implementation(Demo, [c, d, e])
```

This gives Bilt three available relationships:

```text
a, b ──► c
a, c ──► d
c, d ──► e
```

Now create an instance with just `a` and `b` known:

```python
demo = Demo(
    a=1.0,
    b=2.0,
)
```

and ask for `e`:

```python
with basic:
    assert demo.e == 7.0
```

To satisfy that request, Bilt needs to obtain:

```text
c = 3
d = 4
e = 7
```

or, as a graph:

```text
a ──┐
    ├──► c ──┐
b ──┘        ├──► e
a ──────► d ─┘
          ▲
          │
          c
```

We never wrote:

```python
c_value = c(a, b)
d_value = d(a, c_value)
e_value = e(c_value, d_value)
```

The implementation describes which relationships are available. It does not prescribe the order in which they must be evaluated.

### The model is not the execution order

Nor does the order in:

```python
Implementation(Demo, [c, d, e])
```

mean:

```text
first c
then d
then e
```

Those relationships describe the model.

The required execution order depends on which values are already known and which value we request.

If we request `c`, Bilt only needs:

```text
a, b ──► c
```

If we request `e`, it needs a longer route:

```text
a, b ──► c
a, c ──► d
c, d ──► e
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

[← Equations and acausal modelling](equations-and-acausal-modelling.md) · [Index](docs/index.md) · [Dimensions, domains and tensors →](dimensions-domaains-and-tensors.md)
