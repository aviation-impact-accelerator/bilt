---
name: bilt
description: Build, run, check and debug scientific and engineering models using the Bilt Python framework. Use when writing Bilt systems, acausal equations, transforms, subsystems, physical units or named tensors in an application that depends on Bilt. Not for changing Bilt's implementation.
license: MIT
---

# Develop models with Bilt

Use this skill when writing or reviewing models that **use** Bilt. For changes to Bilt itself, use the `bilt-development` skill instead.

Bilt is equation-oriented: define relationships, provide known inputs, and request outputs. The framework plans the calculation. Start from the user's scientific assumptions, not from a procedural calculation order.

## Typical workflow

1. Identify system boundaries, named values, units and relationships.
2. Declare an interface using `@system`, `Interface` and `field()`.
3. Prefer `@equation` for mathematical relationships; use `@transform` for genuinely directed operations.
4. Create an `Implementation` from the relationships, supplying child implementations when required.
5. Provide known values, activate the implementation and request outputs.
6. Check dimensions and validate numeric results against physical expectations.

## Minimal example

```python
from bilt import Implementation, Interface, equation, field, system


@system
class Motion(metaclass=Interface):
    distance: float = field()
    speed: float = field()
    time: float = field()


@equation
def motion_relation(distance: float, speed: float, time: float) -> bool:
    return distance == speed * time


motion = Implementation(Motion, [motion_relation])
with motion:
    assert Motion(speed=20.0, time=5.0).distance == 100.0
    assert Motion(distance=100.0, time=5.0).speed == 20.0
```

## Modelling rules

- Equations specify relationships, not arbitrary Python programs. Keep them within Bilt's supported symbolic syntax: single-assignment locals, a final relationship return, and supported `if/elif/else` conditionals. Do not invent loop, mutation or tensor-predicate support in equations.
- Avoid manually rearranging relationships or hard-coding their direction. Bilt can solve supported coupled **linear** blocks, including blocks crossing subsystem boundaries; do not assume arbitrary nonlinear coupled blocks work.
- Use `Annotated[Quantity | QuantityTensor, DIMENSION]` for unit-aware system ports that may take scalar or named-tensor values. For scalar equations use `Annotated[Quantity, DIMENSION]` where appropriate. Prefer Bilt's named tensor operations to manual positional broadcasting.
- Remember that `Implementation` selects equations and transforms; registering one alone does not activate it. Results are cached on a system instance. Use fresh instances when changing inputs or implementations.
- Use the optional JAX backend only with supported operations and compatible transforms; test behaviour on the intended backend.
- Do not invent or rely on internal Bilt APIs. Check the installed version against the current documentation and examples.

## Checking models

- Run `uv run bilt check .` from a model project where Bilt is installed. It validates discoverable model definitions, not numerical solvability or every possible runtime path.
- Run an example with known outputs and add numerical and dimensional tests; model checking is not a substitute for evaluation.
- When stuck, reduce to a minimal system and inspect the specific error before adding constraints, changing units or reordering equations.

## Current reference

- [User guide](https://aviation-impact-accelerator.github.io/bilt/guide/modelling-philosophy/)
- [Systems and implementations](https://aviation-impact-accelerator.github.io/bilt/api/systems/)
- [Equations and acausal modelling](https://aviation-impact-accelerator.github.io/bilt/guide/equations-and-acausal-modelling/)
- [Coupled systems](https://aviation-impact-accelerator.github.io/bilt/guide/coupled-systems/)
- [Units and tensors](https://aviation-impact-accelerator.github.io/bilt/guide/units/)
- [Model checker](https://aviation-impact-accelerator.github.io/bilt/check/)
