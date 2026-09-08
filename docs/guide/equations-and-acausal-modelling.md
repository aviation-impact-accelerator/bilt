## Equations and acausal models

In the previous chapter, we saw that named values lie on the boundary of a system. The system itself, however, does not describe how those values are related.

For that, Bilt provides **equations**.

### Equations describe relationships

Consider a simple relationship between distance, speed, and time:

```text
distance = speed × time
```

We can declare the values with a system:

```python
from bilt import Interface, equation, field, system


@system
class Motion(metaclass=Interface):
    distance: float = field()
    speed: float = field()
    time: float = field()
```

Then define the relationship:

```python
@equation
def kinematic_relation(distance: float, speed: float, time: float) -> bool:
    return distance == speed * time
```

The equation simply says:

```text
distance = speed × time
```

All three values participate in the relationship. None is permanently designated as the input or output.

There is one slightly unusual detail in the function signature:

```python
-> bool
```

In ordinary Python, the expression:

```python
distance == speed * time
```

would evaluate to either `True` or `False`.

An equation decorated with `@equation` is different. Bilt interprets the function symbolically, capturing the algebraic relationship rather than using it as an ordinary Boolean-valued Python function.

Conceptually, the Python function is a convenient way of writing:

```text
distance = speed × time
```

in a form that Bilt can inspect and reason about.

### Using an equation

Equations become part of a system through an **implementation**:

```python
from bilt import Implementation


basic = Implementation(Motion, [kinematic_relation])
```

An implementation supplies the relationships Bilt can use for a particular system. We'll see more of what implementations can contain in the next chapter.

Suppose we know the speed and time:

```python
journey = Motion(
    speed=20.0,
    time=5.0,
)
```

We can ask for the distance:

```python
with basic:
    assert journey.distance == 100.0
```

Bilt can use the equation as:

```text
distance = speed × time
```

But it is not restricted to this single form of the mathematical relationship.

### The same equation can be used in another direction

Suppose instead that we know the distance and time:

```python
journey = Motion(
    distance=100.0,
    time=5.0,
)
```

Now ask for the speed:

```python
with basic:
    assert journey.speed == 20.0
```

We never wrote a separate calculation for:

```text
speed = distance / time
```

Bilt can rearrange the equation to obtain it.

Likewise, if we know distance and speed:

```python
journey = Motion(
    distance=100.0,
    speed=20.0,
)
```

we can request time:

```python
with basic:
    assert journey.time == 5.0
```

using:

```text
time = distance / speed
```

The model still contains only one relationship:

```text
distance = speed × time
```

The direction is chosen when Bilt works out how to satisfy a particular request.

### This is what we mean by acausal

Bilt is often described as supporting **acausal modelling**.

That can sound more mysterious than it is.

An acausal relationship does not specify a permanent direction of computation:

```text
distance ─┐
          │
speed ────┼── distance = speed × time
          │
time ─────┘
```

Depending on what is known, Bilt may use that relationship as:

```text
speed, time ──► distance
```

or:

```text
distance, time ──► speed
```

or:

```text
distance, speed ──► time
```

The relationship is acausal.

The eventual calculation is not. Once Bilt decides how the equation is being used, there is still a concrete computation to perform.

### Equations are symbolic

An equation describes a mathematical expression that Bilt can inspect and reason about.

For:

```python
@equation
def kinematic_relation(distance: float, speed: float, time: float) -> bool:
    return distance == speed * time
```

Bilt conceptually sees a structure more like:

```text
        =
       / \
distance   ×
          / \
       speed time
```

This symbolic representation is what allows Bilt to reason about different ways of using the relationship.

It also means an equation is not simply an arbitrary Python function. The expression needs to consist of mathematical operations that Bilt can represent symbolically.

### Not every equation can simply be rearranged

Our kinematic equation is pleasantly straightforward:

```text
distance = speed × time
```

Each value can be isolated using ordinary algebra.

Real equations are not always so cooperative.

A value may appear more than once, several solutions may be possible, or the relationship may require solving several equations together.

So an equation does not mean:

> Bilt can always calculate any value from all the others.

It means:

> Bilt knows the mathematical relationship and can reason about the ways it may be used.

For simple relationships, that may mean rearranging the equation directly. More complicated systems may require several equations to be considered together or eventually passed to a numerical solver.

We'll return to those cases later.

### Not every relationship is an equation

Equations are a natural starting point for Bilt because many engineering models are written as mathematical relationships.

But not every piece of a model is best described this way.

Some operations have a natural direction of computation. Others may contain conditionals, lookups, calls to external software, or calculations that cannot be meaningfully rearranged.

For those cases, Bilt provides **transforms**.

A useful distinction is:

> An equation describes a relationship between values.

> A transform describes how to calculate one value from others.

In the next chapter, we'll introduce transforms and see how equations and transforms can live together in the same implementation.

[← Systems and values](systems-and-values.md) · [Index](../index.md) · [Transforms and implementations →](transforms-and-implementations.md)
