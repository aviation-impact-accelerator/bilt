## Using different backends

Bilt separates the model from the way its computations are executed.

Systems, equations, and transforms describe the model. A **backend** executes the computation Bilt derives from it.

Bilt currently provides Python and JAX backends.

## Python backend

The Python backend is available with the standard Bilt installation:

```console
uv add bilt
```

or:

```console
pip install bilt
```

For normal use, no additional dependencies are required.

## Install the JAX backend

JAX is an optional dependency.

With `uv`:

```console
uv add "bilt[jax]"
```

or with `pip`:

```console
pip install "bilt[jax]"
```

This installs Bilt together with its supported JAX dependency.

## Use the JAX backend

A Bilt model does not need to be rewritten for JAX.

For example:

```python
from bilt import Implementation, Interface, equation, field, system


@system
class Motion(metaclass=Interface):
    distance: float = field()
    speed: float = field()
    time: float = field()


@equation
def kinematic_relation(
    distance: float,
    speed: float,
    time: float,
) -> bool:
    return distance == speed * time


motion_model = Implementation(
    Motion,
    [kinematic_relation],
)
```

The model is backend-independent.

To execute it using JAX, select the JAX backend around the evaluation:

```python
from bilt import JAX_BACKEND


motion = Motion(
    speed=20.0,
    time=5.0,
)

with JAX_BACKEND, motion_model:
    assert motion.distance == 100.0
```

Changing the backend does not change the system, equations, or implementation.

## What the backend does

When a value is requested, Bilt first determines how it can be calculated.

The resulting computation is then handed to the selected backend:

```text
model
  ↓
structural analysis
  ↓
computation plan
  ↓
Python AST
  ↓
backend
  ├── Python
  └── JAX
  ↓
result
```

The Python backend produces ordinary Python execution.

The JAX backend transpiles the generated computation into JAX operations.

## Writing JAX-compatible relationships

Simple arithmetic works naturally with the JAX backend:

```python
@equation
def kinetic_energy_relation(
    kinetic_energy: float,
    mass: float,
    velocity: float,
) -> bool:
    return kinetic_energy == 0.5 * mass * velocity**2
```

JAX operations can also be used where required:

```python
import jax.numpy as jnp

from bilt import transform


@transform
def root(value: float) -> float:
    return jnp.sqrt(value)
```

The JAX backend supports a restricted subset of Python syntax.

For example, Python `if` statements are not currently supported:

```python
@transform
def absolute(value: float) -> float:
    if value > 0:
        return value

    return -value
```

A model intended for JAX execution should therefore use operations supported by the JAX backend.

## Lists and arrays

With the JAX backend, list literals are converted to JAX arrays.

For example:

```python
@transform
def values() -> list[float]:
    return [1.0, 2.0, 3.0]
```

produces a JAX array when executed with the JAX backend.

Nested lists similarly become multidimensional JAX arrays.

## Choosing a backend

Use the Python backend for straightforward execution and development.

Use the JAX backend when the generated computation needs to participate in JAX-based numerical workflows.

In either case, the model stays the same:

```text
system
equations
transforms
implementation
```

Only the execution backend changes.

## What to remember

Install ordinary Bilt with:

```console
uv add bilt
```

Install JAX support with:

```console
uv add "bilt[jax]"
```

Then select JAX when evaluating:

```python
from bilt import JAX_BACKEND

with JAX_BACKEND, implementation:
    result = model.value
```

The model describes **what** to compute.

The backend determines **how** that computation runs.
