## Transforms and implementations

In the previous chapter, we defined a system with three values:

```python
from bilt import Interface, field, system


@system
class Demo(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()
```

and created an instance with two known values:

```python
demo = Demo(a=1.0, b=2.0)
```

At this point, Bilt knows that `c` exists, but it still has no way to calculate it.

To give the system some behaviour, we need to describe a relationship between its values.

### Transforms describe calculations

A **transform** is a directed calculation from one or more values to another.

For our `Demo` system, we can define:

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

The function itself is ordinary Python. The parameter names identify the values it needs, while the function name identifies the value it provides.

So:

```python
@transform
def c(a: float, b: float) -> float:
    return a + b
```

means:

```text
given:    a, b
provides: c
```

A transform has a fixed computational direction. Bilt can use this transform to calculate `c` from `a` and `b`, but it cannot use it to calculate `a` from `b` and `c`.

We will introduce relationships that can be used in more than one direction in the next chapter.

### Implementations collect the relationships for a system

Defining the transform still doesn't attach it to `Demo`.

For that we create an **implementation**:

```python
from bilt import Implementation


basic = Implementation(Demo, [c])
```

The system and its implementation have different jobs.

`Demo` describes the boundary:

```text
Demo
├── a
├── b
└── c
```

while `basic` describes one way those values can be related:

```text
a, b ──► c
```

Keeping these separate means the same system can later have more than one implementation without changing the code that uses its interface.

### Evaluating with an implementation

To use an implementation, make it active while evaluating the system:

```python
with basic:
    result = demo.c
```

Here, Bilt knows:

```text
a = 1.0
b = 2.0
```

and we request:

```text
c
```

The active implementation contains a transform that can produce `c` from `a` and `b`, so Bilt can satisfy the request:

```python
with basic:
    assert demo.c == 3.0
```

Conceptually, the evaluation looks like this:

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

Instead, we asked the system for a value:

```python
demo.c
```

Bilt worked out that the transform provided a way to obtain it.

### The model is not the execution order

With one transform, this may not seem very different from calling a function directly.

The difference becomes clearer when we add a few more values:

```python
@system
class Demo(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()
    d: float = field()
    e: float = field()
```

and some more transforms:

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

We can collect them into an implementation:

```python
basic = Implementation(Demo, [c, d, e])
```

This describes the following relationships:

```text
a, b ──► c
a, c ──► d
c, d ──► e
```

Now create a system with just `a` and `b` known:

```python
demo = Demo(a=1.0, b=2.0)
```

and ask for `e`:

```python
with basic:
    assert demo.e == 7.0
```

To answer that request, Bilt needs to calculate:

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

Nor does the order of transforms in:

```python
Implementation(Demo, [c, d, e])
```

define an execution sequence.

The implementation describes the relationships that are available. When we request a value, Bilt works out which of those relationships are needed and in what order they must be evaluated.

> The model describes the available relationships. The request determines the computation.

That distinction becomes much more important as models grow.

### Transforms are directional

There is one limitation in the example above that is worth making explicit.

Suppose we know:

```text
a = 1
c = 3
```

and want `b`.

Mathematically, we can see from:

```text
c = a + b
```

that:

```text
b = c - a
```

But our transform does not describe that relationship. It describes only the calculation:

```text
a, b ──► c
```

Bilt will not automatically reverse an arbitrary Python function.

This is intentional. A transform is an ordinary directed calculation, and there may be no unique way to run it backwards.

For example:

```python
@transform
def magnitude(value: float) -> float:
    return abs(value)
```

Given `value`, we can always calculate its magnitude. But knowing the magnitude does not tell us whether the original value was positive or negative.

More generally, transforms can contain conditionals, lookups, calls to other software, or any other computation that Bilt cannot simply rearrange.

When a relationship really is mathematical and useful in more than one direction, Bilt has another way to express it.

Those are **equations**, and they are the subject of the next chapter.
