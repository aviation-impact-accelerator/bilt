## Equations and acausal models

In the previous chapter, we saw that named values lie on the boundary of a system. The system itself, however, does not describe how those values are related.

For that, Bilt provides **equations**.

### Equations describe relationships

Consider a simple relationship between distance, speed, and time:

```text
distance = speed × time
```

We can declare the values with a system:

```python exec="on" session="motion-equations"
from bilt import Interface, equation, field, system


@system
class Motion(metaclass=Interface):
    distance: float = field()
    speed: float = field()
    time: float = field()
```

Then define the relationship:

```python exec="on" session="motion-equations"
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

```python exec="off"
-> bool
```

In ordinary Python, the expression:

```python exec="off"
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

```python exec="on" session="motion-equations"
from bilt import Implementation


basic = Implementation(Motion, [kinematic_relation])
```

An implementation supplies the relationships Bilt can use for a particular system. We'll see more of what implementations can contain in the next chapter.

Suppose we know the speed and time:

```python exec="on" session="motion-equations"
journey = Motion(
    speed=20.0,
    time=5.0,
)
```

We can ask for the distance:

```python exec="on" session="motion-equations"
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

```python exec="on" session="motion-equations"
journey = Motion(
    distance=100.0,
    time=5.0,
)
```

Now ask for the speed:

```python exec="on" session="motion-equations"
with basic:
    assert journey.speed == 20.0
```

We never wrote a separate calculation for:

```text
speed = distance / time
```

Bilt can rearrange the equation to obtain it.

Likewise, if we know distance and speed:

```python exec="on" session="motion-equations"
journey = Motion(
    distance=100.0,
    speed=20.0,
)
```

we can request time:

```python exec="on" session="motion-equations"
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

```python exec="off"
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

### Writing equations

An equation should describe a mathematical relationship, rather than a series of instructions.

You can give parts of the calculation names to make it easier to read:

```python exec="off"
@equation
def adjusted_distance(distance: float, speed: float, time: float) -> bool:
    travelled = speed * time
    adjusted = travelled + 1.0
    return distance == adjusted
```

There are a few rules to follow:

- End with **one `return` statement** that gives a relationship, such as `distance == adjusted`.
- Assign each name **only once**. You cannot change the value of a name later in the function.
- Do not change the contents of objects, lists or dictionaries.
- Do not use loops or comprehensions.

For example, the following will not work:

```python exec="off"
@equation
def invalid_relation(distance: float, speed: float, time: float) -> bool:
    travelled = speed * time
    travelled += 1.0
    return distance == travelled
```

Here, `travelled` is changed after its first assignment. Instead, give the new value a different name, as in the previous example.

You can call other functions, provided Bilt understands the mathematical operations they produce. Avoid functions that change files or other values outside the equation.

Bilt also needs to read your function's original Python code. For this reason, define equations in Python files. Functions created dynamically, such as with `exec`, will not work, and some notebook environments may have difficulty providing the source code.

### Using `if/else` in an equation

Sometimes a relationship depends on a condition. For example, we might want to square positive values but change the sign of negative ones:

```python exec="on" session="motion-equations"
@system
class Piecewise(metaclass=Interface):
    x: float = field()
    result: float = field()


@equation
def piecewise_relation(x: float, result: float) -> bool:
    if x > 0:
        selected = x**2
    else:
        selected = -x
    return result == selected


piecewise_model = Implementation(Piecewise, [piecewise_relation])
```

Both branches use the name `selected`. Bilt understands both possibilities, so it can choose the right one when a value is requested:

```python exec="on" session="motion-equations"
with piecewise_model:
    assert Piecewise(x=3.0).result == 9.0
    assert Piecewise(x=-3.0).result == 3.0
```

The same equation works with the JAX backend:

```python exec="on" session="motion-equations"
from bilt import JAX_BACKEND

with JAX_BACKEND, piecewise_model:
    assert Piecewise(x=-3.0).result == 3.0
```

You can also use `elif` and nested `if/else` blocks:

```python exec="on" session="motion-equations"
@equation
def multiple_ranges(x: float, result: float) -> bool:
    if x > 10:
        selected = x * 3
    elif x > 0:
        if x > 5:
            selected = x * 2
        else:
            selected = x
    else:
        selected = -x
    return result == selected


multi_range_model = Implementation(Piecewise, [multiple_ranges])
with multi_range_model:
    assert Piecewise(x=7.0).result == 14.0
    assert Piecewise(x=-3.0).result == 3.0

with JAX_BACKEND, multi_range_model:
    assert Piecewise(x=12.0).result == 36.0
```

Each condition must test a single comparison such as `x > 0`, and every path
must assign the same name exactly once. Each chain needs an `else`, and the
single `return` comes afterwards. Multiple separate `if` statements are not
supported yet. Conditions and selected values must be **single numbers, not
tensors (arrays of values)**. Inputs used in any branch must be available,
even if only one branch is selected.

Bilt can calculate `result` from `x` in this example. It cannot currently work backwards through the `if/else` to calculate `x` from `result`.

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

Some operations have a natural direction of computation. Others involve more complicated instructions or calculations that Bilt cannot treat as equations.

For those cases, Bilt provides **transforms**. Equations can include supported `if/elif/else` and nested conditionals, but transforms are better suited to calculations that do not describe a mathematical relationship.

A useful distinction is:

> An equation describes a relationship between values.

> A transform describes how to calculate one value from others.

In the next chapter, we'll introduce transforms and see how equations and transforms can live together in the same implementation.

[← Systems and values](systems-and-values.md) · [Index](../index.md) · [Transforms and implementations →](transforms-and-implementations.md)
