## Equations and acausal models

In the previous chapter, we saw that transforms are directional.

A transform such as:

```python
@transform
def c(a: float, b: float) -> float:
    return a + b
```

describes one calculation:

```text
a, b ──► c
```

Even though we can rearrange `c = a + b` ourselves, Bilt does not treat the body of an arbitrary Python function as an equation.

Sometimes, though, the relationship itself is exactly what we want to describe.

For that, Bilt provides **equations**.

### Equations describe relationships

Consider a simple relationship between distance, speed, and time:

```text
distance = speed × time
```

We can describe the values with a system:

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
def motion(distance: float, speed: float, time: float):
    return distance == speed * time
```

Unlike a transform, the parameters here are not inputs and the function name is not an output.

The equation simply says:

```text
distance = speed × time
```

All three values participate in the relationship.

### Using an equation

As with transforms, equations become part of a system through an implementation:

```python
from bilt import Implementation


basic = Implementation(Motion, [motion])
```

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

So far, this looks much like a transform.

The difference appears when we change what we know.

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

We never wrote a separate calculation:

```python
def speed(distance: float, time: float) -> float:
    return distance / time
```

Bilt can rearrange the equation to obtain:

```text
speed = distance / time
```

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

There is an important difference between this:

```python
@transform
def distance(speed: float, time: float) -> float:
    return speed * time
```

and this:

```python
@equation
def motion(distance: float, speed: float, time: float):
    return distance == speed * time
```

A transform contains ordinary Python code that Bilt can execute in the direction you wrote it.

An equation describes a mathematical expression that Bilt can inspect and reason about.

Conceptually, Bilt sees something more like:

```text
        =
       / \
distance   ×
          / \
       speed time
```

That symbolic structure is what allows Bilt to rearrange the relationship.

It also means equations are more restricted than arbitrary transforms. They need to be expressed using mathematical operations that Bilt understands symbolically.

### Not every equation can simply be reversed

Our motion equation is pleasantly straightforward:

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

For simple relationships, that may mean rearranging the equation directly. More complicated systems may require other forms of analysis or a numerical solver.

We'll return to those cases later.

### Transforms or equations?

Transforms and equations overlap in what they can represent, but they communicate different things.

A transform says:

> Given these values, here is how to calculate another value.

An equation says:

> These values obey this mathematical relationship.

For example:

```python
@transform
def magnitude(value: float) -> float:
    return abs(value)
```

is naturally a transform. It describes a calculation, and there is no unique way to recover `value` from its magnitude.

By contrast:

```python
@equation
def motion(distance: float, speed: float, time: float):
    return distance == speed * time
```

is naturally an equation because the relationship itself is useful.

A useful rule of thumb is:

> Use a transform when the computation is the important thing. Use an equation when the relationship is the important thing.

### The model now contains alternatives

With transforms alone, our model contained directed routes:

```text
a, b ──► c
a, c ──► d
c, d ──► e
```

Equations add another possibility. A single relationship may provide several possible routes depending on what is already known.

For `Motion`:

```text
             ┌── speed, time ─────► distance
motion ──────┼── distance, time ──► speed
             └── distance, speed ─► time
```

Bilt therefore has more than a list of calculations to execute.

It has a collection of possible relationships, some directed and some usable in several directions, and it needs to decide which ones can satisfy the request at hand.

That is the subject of the next chapter: **from models to computation**.
