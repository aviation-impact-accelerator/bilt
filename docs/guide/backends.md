# Using different backends

Bilt separates the model from the way its computations are executed.

Systems, equations, and transforms describe the model. A **backend** determines
how Bilt runs the calculation.

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

The Python backend is used by default, so for normal use you do not need to
select a backend explicitly:

```python
with motion_model:
    assert motion.distance == 100.0
```

You can also select it explicitly when required:

```python
from bilt import PYTHON_BACKEND

with PYTHON_BACKEND, motion_model:
    assert motion.distance == 100.0
```

No additional dependencies are required.

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

By default, Bilt uses the Python backend. To use JAX instead, select the JAX
backend around the evaluation:

```python
from bilt import JAX_BACKEND

motion = Motion(
    speed=20.0,
    time=5.0,
)

with JAX_BACKEND, motion_model:
    assert motion.distance == 100.0
```

Changing the backend does not change the system, equations, transforms, or
implementation.

## Python and JAX

For many models, the Python backend is the simplest choice. It has no
compilation step and works naturally with ordinary Python and NumPy.

The JAX backend is useful when a substantial numerical calculation is evaluated
repeatedly. JAX compiles the calculation, which has an initial cost but can make
later evaluations faster. Compilation can also combine numerical operations and
reduce some intermediate array work.

A useful distinction is:

```text
Python      run the calculation directly

JAX cold    compile and run the calculation

JAX warm    reuse the compiled calculation
```

JAX is therefore most attractive for workloads such as large parameter sweeps,
uncertainty propagation, optimisation, calibration, or other repeated numerical
evaluations.

It is not automatically faster. For small or one-off calculations, compilation
may cost more than it saves.

The JAX worked example compares Python, cold JAX, and warm JAX on a large array
calculation.

## Writing JAX-compatible relationships

Code evaluated with the JAX backend must use operations supported by Bilt's JAX
backend.

Simple numerical expressions work naturally:

```python
@equation
def kinetic_energy_relation(
    kinetic_energy: float,
    mass: float,
    velocity: float,
) -> bool:
    return kinetic_energy == 0.5 * mass * velocity**2
```

Supported JAX operations can also be used where required:

```python
import jax.numpy as jnp

from bilt import transform


@transform
def root(value: float) -> float:
    return jnp.sqrt(value)
```

The JAX backend is well suited to array-based numerical calculations built from
supported arithmetic and JAX operations.

Some ordinary Python behaviour is not suitable for JAX execution. Examples
include file or network access, Python side effects, and calls to libraries that
do not provide JAX-compatible operations.

Bilt's current JAX backend also supports only a subset of Python syntax. For
example, ordinary Python `if` statements are not currently supported:

```python
@transform
def absolute(value: float) -> float:
    if value > 0:
        return value
    return -value
```

Additional syntax and JAX capabilities may be supported by Bilt in the future.
For now, relationships intended for JAX execution should use the operations and
syntax supported by the current backend.

## Lists and arrays

With the JAX backend, list literals are converted to JAX arrays.

For example:

```python
@transform
def values() -> list[float]:
    return [1.0, 2.0, 3.0]
```

produces a JAX array when evaluated with the JAX backend.

Nested lists similarly become multidimensional JAX arrays.

## Nested backend evaluation

The preferred approach is to evaluate a Bilt model with a single backend.

This keeps the complete chain of relationships used to produce an output
visible to Bilt. If the model is JAX-compatible, selecting the JAX backend
around the whole implementation preserves that transparency while allowing JAX
to compile the calculation.

There is currently an escape hatch for cases where this is not possible. A
transform running with the Python backend can explicitly evaluate another Bilt
implementation using JAX.

For example, a JAX-compatible subproblem can be defined normally:

```python
from bilt import Implementation, Interface, field, system, transform


@system
class JaxSubproblem(metaclass=Interface):
    value: float = field()
    result: float = field()


@transform
def result(value: float) -> float:
    return value**2 + 2.0 * value + 1.0


jax_subproblem = Implementation(
    JaxSubproblem,
    [result],
)
```

A Python-backed transform can then delegate to it:

```python
@transform
def output_value(input_value: float) -> float:
    from bilt import JAX_BACKEND
    from my_model import JaxSubproblem, jax_subproblem

    value = abs(input_value)

    with JAX_BACKEND, jax_subproblem:
        return JaxSubproblem(value=value).result
```

This can be useful when an unavoidable Python-only operation must happen before
a substantial JAX-compatible numerical calculation.

However, the nested evaluation creates a boundary in the model. The outer Bilt
evaluation sees the transform as one operation; it cannot inspect the chain of
relationships used inside the nested JAX evaluation. This means the complete
calculation is no longer represented as one transparent Bilt analysis.

For that reason, nested backend evaluation should be used **judiciously** and
treated as a workaround rather than the preferred way to compose a model.

Future Bilt support may make it possible to represent backend boundaries while
preserving transparency across the complete calculation. Until then, prefer a
single backend over the whole model whenever practical.

## Choosing a backend

For most use, start with the Python backend.

| Use Python when | Use JAX when |
| --- | --- |
| the calculation is small or one-off | the numerical calculation is substantial |
| ordinary Python compatibility matters | the calculation is JAX-compatible |
| Python-only libraries or dynamic behaviour are required | the same calculation is evaluated repeatedly |
| compilation would not pay off | the whole relevant calculation can remain visible to Bilt under JAX |

Select Python explicitly when required:

```python
from bilt import PYTHON_BACKEND

with PYTHON_BACKEND, implementation:
    result = model.value
```

or select JAX:

```python
from bilt import JAX_BACKEND

with JAX_BACKEND, implementation:
    result = model.value
```

The model describes **what** to compute. The backend determines **how** that
calculation runs.

[← Systems of systems](systems-of-systems.md) · [Index](../index.md) · [Units →](units.md)
