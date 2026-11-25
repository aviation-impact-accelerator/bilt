---
name: bilt
description: Build, run, check and debug scientific and engineering models using the Bilt Python framework. Use when writing Bilt systems, acausal equations, transforms, subsystems, physical units or named tensors in an application that depends on Bilt. Not for changing Bilt's implementation.
license: MIT
---

# Develop models with Bilt

Use this skill when writing or reviewing models that **use** Bilt. For changes to Bilt itself, use the `bilt-development` skill instead.

Bilt is equation-oriented: define relationships, provide known inputs, and request outputs. The framework plans the calculation. Start from the user's scientific assumptions, not from a procedural calculation order. Check the Bilt version pinned by the project before using an API: the published documentation follows `main`, which may be newer than the installed version.

## Typical workflow

1. Identify system boundaries, named values, physical dimensions and relationships. Prefer unit-bearing quantities for physical values; keep categorical values and genuinely unitless interfaces plain.
2. Declare an interface using `@system`, `Interface` and `field()`.
3. Prefer `@equation` for mathematical relationships; use `@transform` for genuinely directed operations.
4. Create an `Implementation` from the relationships, supplying child implementations when required.
5. Provide known values, activate the implementation and request outputs.
6. Check dimensions and validate numeric results against physical expectations.

## Minimal example

Use the same quantity alias for a system field and a pointwise equation. The model works for scalar quantities and named tensors without changing the equation.

```python
from typing import Annotated

from bilt import Implementation, Interface, Quantity, QuantityTensor, dim, equation, field, system, tensor
from bilt.units import meter, second
from bilt.units.dimensions import LENGTH, TIME

Length = Annotated[Quantity | QuantityTensor, LENGTH]
Speed = Annotated[Quantity | QuantityTensor, LENGTH / TIME]
Duration = Annotated[Quantity | QuantityTensor, TIME]


@system
class Motion(metaclass=Interface):
    distance: Length = field()
    speed: Speed = field()
    duration: Duration = field()


@equation
def motion_relation(distance: Length, speed: Speed, duration: Duration) -> bool:
    return distance == speed * duration


motion = Implementation(Motion, [motion_relation])
scenario = dim("scenario")

with motion:
    assert Motion(speed=20.0 * meter / second, duration=5.0 * second).distance == 100.0 * meter
    assert Motion(distance=100.0 * meter, duration=5.0 * second).speed == 20.0 * meter / second

    distances = Motion(
        speed=tensor([10.0, 20.0], over=scenario, unit=meter / second),
        duration=5.0 * second,
    ).distance

assert distances.unit == meter
assert distances.domain.dims == (scenario,)
assert distances.values.tolist() == [50.0, 100.0]
```

## Modelling rules

### Relationships and implementations

- Prefer equations for mathematical relationships, even if Bilt cannot invert them in every direction. Use transforms for empirical lookups, external routines or other genuinely directed calculations. A transform's name identifies its output; its parameters name its inputs. Define both decorators in real Python source files so Bilt can inspect them. Transform globals and closures are not automatically available to generated code; consult the backend rules.
- Equations use single-assignment locals and one final relationship return. Supported conditionals include one top-level `if/elif/else` or numeric-literal `match/case` (with final `case _`), including nested branches. Every branch assigns the same result name. All branch inputs are required; tensor masks use `&`, `|` and `~` and evaluate both branches. Do not assume loops, mutation, enum-member patterns or inversion through a conditional are supported.
- Do not manually rearrange relationships just to fix a calculation order. Bilt can isolate supported equations and solve coupled linear blocks across subsystem boundaries; it does not provide a general nonlinear coupled solver on this version.
- Equations in one implementation express simultaneous constraints, not competing recipes for the same unknown. Use separate implementations for alternative modelling assumptions. Bind child implementations with `internal_implementations={"child": child_model}` and connect them through equation parameters annotated with the child interface, referencing `child.port`.
- Activate an implementation with `with model:` and request a field on an instance. The first calculated lookup selects the implementation; outputs and intermediates are cached on that instance. Structural planning uses the original supplied inputs. Use fresh instances when changing inputs, implementation or backend.

### Quantities and domains

- Declare physical dimensions, not units, in annotations. Use a shared alias such as `Mass = Annotated[Quantity | QuantityTensor, MASS]` on fields and pointwise relationships. Use concrete `Quantity` or `QuantityTensor` when scalar-only or named tensor operations require it; do not create paired aliases solely for system fields. Annotations must be available at runtime.
- Supply quantities with explicit compatible units. Calculated outputs use coherent units; convert to the desired display unit. Dimensionless quantities still need a unit, e.g. `UNITLESS`; plain floats are outside the unit system. Distinguish absolute readings from relative changes and preserve metadata tags when relevant.
- In transforms, extract scalar numbers with `q.convert_to(unit).value`, not bare `q.value`, and construct quantities with `Quantity(value, unit)` or a float times a unit. Transforms do not prove arbitrary Python dimensionally correct.
- Use `dim` and `tensor(..., over=axis, unit=...)` to name axes. Shared axes align by name and must have matching lengths; independent axes broadcast. Do not add loops or positional reshaping for pointwise model calculations.
- Use named reductions and selections in supported equations; use `QuantityTensor` (or unitless `Tensor`) for tensor-method parameters. Transforms support a restricted reduction form. Concrete tensors outside model execution do not implement general eager arithmetic or reductions; use their numerical arrays with explicit units/domains when post-processing. Public coordinate binding is not yet available; resolve labels to positions and use `isel` instead. Do not use private APIs as a workaround.

### Backends and performance

- Python is the default; use `with JAX_BACKEND, model:` with the optional JAX extra for compatible calculations. Equation conditionals and ordinary Python control flow in transforms have different restrictions. Check the backend reference and test the target backend rather than assuming all Python runs under JAX.
- Distinguish structural planning, lowering, JIT compilation, numerical execution and instance-value caching. Benchmark cold and warm calls on fresh instances using the same implementation; synchronise JAX results before stopping the timer.
- Use only the documented public API for the installed Bilt version. Import from `bilt`, `bilt.units`, `bilt.units.dimensions` and `bilt.errors` as appropriate. Do not import private modules, access reserved model metadata, manipulate caches or call internal methods, even if they have no leading underscore. Evaluate models through declared fields on system instances within the active implementation and backend contexts. Expose child inputs and outputs on the parent interface through equations rather than evaluating child ports directly. If the public API cannot express a requirement, report the limitation instead of working around it through internals.

## Checking models

- Run `uv run bilt check .` from a model project where Bilt is installed. It discovers and imports candidate modules (so module-level code runs). It checks constructed definitions and statically recognisable private API misuse, not numerical solvability or every possible runtime path. Add explicit modules when discovery cannot find them.
- Run an example with known outputs and add numerical and dimensional tests; model checking is not a substitute for evaluation.
- When stuck, reduce to a minimal system and inspect the specific error before adding constraints, changing units or reordering equations.

## Current reference

Use the documentation for details rather than duplicating the entire API in this skill.

- [User guide](https://aviation-impact-accelerator.github.io/bilt/guide/modelling-philosophy/)
- [Systems and implementations](https://aviation-impact-accelerator.github.io/bilt/api/systems/)
- [Equations and acausal modelling](https://aviation-impact-accelerator.github.io/bilt/guide/equations-and-acausal-modelling/)
- [Coupled systems](https://aviation-impact-accelerator.github.io/bilt/guide/coupled-systems/)
- [Units](https://aviation-impact-accelerator.github.io/bilt/guide/units/), [quantity annotations](https://aviation-impact-accelerator.github.io/bilt/api/quantity-annotations/)
- [Dimensions, domains and tensors](https://aviation-impact-accelerator.github.io/bilt/guide/dimensions-domains-and-tensors/), [tensor API](https://aviation-impact-accelerator.github.io/bilt/api/tensors/)
- [Equations and transforms API](https://aviation-impact-accelerator.github.io/bilt/api/relationships/), [backends](https://aviation-impact-accelerator.github.io/bilt/api/backends/)
- [Worked examples](https://aviation-impact-accelerator.github.io/bilt/#worked-examples)
- [Model checker](https://aviation-impact-accelerator.github.io/bilt/check/)
