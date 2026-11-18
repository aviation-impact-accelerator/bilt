# Your first Bilt model

In this tutorial, you'll build a small motion model and use the **same equation** to calculate either distance or speed. You'll learn how to declare a system, describe a relationship, create an implementation and request a result.

You only need basic Python. You do not need to understand symbolic algebra or solvers to get started.

## 1. Set up a project

Bilt requires Python 3.15 or newer, [uv](https://docs.astral.sh/uv/) and a [Rust toolchain](https://www.rust-lang.org/tools/install). Create a project and install Bilt from GitHub:

```sh
uv init --python 3.15 my-bilt-project
cd my-bilt-project
uv add "git+https://github.com/aviation-impact-accelerator/bilt.git"
```

Open the generated `main.py`. Replace its contents with the Python blocks below, **in order**, then run the file at the end.

## 2. Declare the values in a system

A system names the quantities that belong to a model. For a simple journey, those are distance, speed and time:

```python exec="on" session="first-model"
from bilt import Implementation, Interface, equation, field, system


@system
class Motion(metaclass=Interface):
    distance: float = field()
    speed: float = field()
    time: float = field()
```

`@system` and `metaclass=Interface` tell Bilt that `Motion` is a system interface. Each `field()` declares a value. Nothing has been calculated yet, and none of these values is permanently an input or output.

## 3. Describe the relationship

We know that distance equals speed multiplied by time. Add an equation to describe that relationship:

```python exec="on" session="first-model"
@equation
def motion_relation(distance: float, speed: float, time: float) -> bool:
    return distance == speed * time
```

This looks like a normal Python function, but `@equation` makes Bilt interpret its body symbolically. It describes the relationship rather than a specific sequence of calculations.

Bilt reads the equation's source code when it is defined, so put it in a Python source file such as `main.py`.

## 4. Create an implementation and ask for distance

An implementation connects the system interface to the relationships it can use:

```python exec="on" session="first-model"
model = Implementation(Motion, [motion_relation])

with model:
    journey = Motion(speed=20.0, time=5.0)
    distance = journey.distance

assert distance == 100.0
print("Distance:", distance)
```

The instance supplies two known values. Reading `journey.distance` asks Bilt to calculate the missing value using the active implementation.

## 5. Ask the same model a different question

Now suppose we know distance and time but need speed:

```python exec="on" session="first-model"
with model:
    another_journey = Motion(distance=100.0, time=5.0)
    speed = another_journey.speed

assert speed == 20.0
print("Speed:", speed)
```

We have not changed the equation or written a separate formula for speed. Bilt works out a supported way to use the relationship from the known values and requested result.

Run the complete file:

```sh
uv run python main.py
```

You should see:

```text
Distance: 100.0
Speed: 20.0
```

## 6. Try it yourself

Change the known values, or create a fresh `Motion` instance with distance and speed to request its `time`. Try adding an assertion for the answer you expect.

This small model illustrates the central idea: **the model describes relationships, while the request determines the calculation**.

In real engineering models, you can also give values physical units, connect reusable subsystems and work with named tensors. Bilt supports many, but not all, possible rearrangements or coupled solves.

## Where next?

- [Equations and acausal modelling](../guide/equations-and-acausal-modelling.md) explains how Bilt represents relationships and what kinds of equations it supports.
- [Units in models](../guide/units.md) adds physical dimensions and unit conversions.
- [Two springs in series](../worked_examples/two-springs-in-series.md) extends the same approach to reusable subsystems and coupled equations.
- [Systems and implementations API](../api/systems.md) documents the interfaces used in this tutorial.

[Back to documentation home](../index.md)
