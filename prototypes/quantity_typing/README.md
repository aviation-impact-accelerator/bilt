# Quantity dimension typing prototype

This experimental package tests nominal physical dimensions and scalar/tensor
representations against **Bilt's actual system constructors, equation compiler,
structural analysis and Python evaluator**. It is an adapter, not a migration of
the production `bilt.units.Quantity` or tensor implementation.

## Run

From the repository root (Python 3.15, matching Bilt):

```sh
uv venv --python 3.15 .venv-prototype
uv pip install --python .venv-prototype/bin/python -r prototypes/quantity_typing/requirements.txt
.venv-prototype/bin/python prototypes/quantity_typing/check.py
PYTHONPATH=src:prototypes/quantity_typing .venv-prototype/bin/python prototypes/quantity_typing/examples.py
```

The checker runner verifies generated files, runs 17 runtime tests, checks the
positive model examples and public stubs with mypy, Pyrefly and ty, and verifies
that **every marked negative example is rejected by each checker**. The model
examples have no casts, Any annotations or ignores. Mypy's `--follow-imports=silent`
retains imported types while suppressing diagnostics in unmodified Bilt internals;
this is not a claim that all production internals pass mypy.

The prototype is excluded from the production checker sweep because it has its
own check scope and intentionally invalid fixtures. Its adapters use dynamic
reflection internally and publish ordinary `.pyi` signatures. This demonstrates
the public typing contract; it does not prove the adapter implementation against
those stubs. Runtime tests cover the supported paths.

## Public API

```python
from bilt_typing import (
    Implementation,
    Interface,
    Tensor,
    equation,
    field,
    system,
    tensor,
)
from bilt_typing.quantities import Mass
from bilt_typing.units import kilogram


@equation
def doubling(m: Mass[float], doubled: Mass[float]) -> bool:
    return doubled == 2.0 * m


@system
class Scale(metaclass=Interface):
    m: Mass[Tensor[float]] = field()
    doubled: Mass[Tensor[float]] = field()
```

(The executable, complete examples are in `examples.py`.) The fields describe
public values; scalar equations describe reusable scalar relations. The adapter
normalizes annotations for Bilt, unwraps supplied quantities into current native
values, and wraps public attribute results back into the declared quantity types.
It rejects scalar/tensor representation mismatches on input and output.

The following also execute through Bilt in `examples.py`:

- An `AbsoluteThermometer` equality equation with Celsius/Fahrenheit conversion.
- A scalar equation reused with tensor mass and scalar acceleration.
- A tensor reduction retaining year while reducing aircraft.
- A whole-tensor transform performing the same reduction.
- A custom mass-per-length dimension used to calculate cable mass.

`Implementation.evaluate` remains Bilt's low-level native-value interface; use
system attributes for wrapped public quantities. Subsystems and configured views
are outside this adapter's tested scope.

## Physical types and generated declarations

```python
from bilt.units.dimensions import MASS
from bilt_typing import dimension_types as dt
from bilt_typing import Quantity, Float

assert dt.MASS.runtime is MASS
type Mass[R: Float] = Quantity[dt.MASS, R]
```

`generate.py` uses existing dimension arithmetic to generate marker identities,
quantity aliases, typed unit wrappers and finite multiplication/division overloads.
The four scalar/tensor combinations are generated for each supported physical
relationship. Dimension parameters are invariant. Compatible units retain the
same dimension type; absolute temperature has a separate marker linking back to
`TEMPERATURE` with the absolute flag.

Generation runs during development, never on import or evaluation:

```sh
PYTHONPATH=src .venv-prototype/bin/python prototypes/quantity_typing/generate.py
PYTHONPATH=src .venv-prototype/bin/python prototypes/quantity_typing/generate.py --check
```

Generated files are committed. Identical dimension compositions reuse the same
marker. The catalogue includes `MASS_PER_LENGTH = MASS / LENGTH` as a concrete
custom-dimension example: it supports mass/length and density*length both in type
inference and model evaluation.

To add another dimension in this experiment, extend `CATALOGUE` (and unit wrappers
as needed) and regenerate the **whole experimental typing layer**. This is a
project-specific catalogue. It does not append overloads to an already installed
Bilt class or provide an arbitrary-plugin extension protocol. Arithmetic outside
the finite catalogue is deliberately not given an `Any` fallback.

## Tensor domains and relations

```python
from typing import Annotated
from bilt_typing import Float, Requires, dim, equation, equal
from bilt_typing.quantities import Mass

AIRCRAFT = dim("aircraft")
type AircraftMass = Annotated[Mass[Tensor[float]], Requires(AIRCRAFT)]


@equation
def total_mass(masses: AircraftMass, total: Mass[Float]) -> bool:
    return equal(total, masses.sum(AIRCRAFT))
```

`Requires` is checked at system input and public output boundaries and on supplied
equation/transform inputs. The native reduction compiler computes remaining axes.
The adapter does not implement general propagation of arbitrary `Requires`
metadata through internal intermediate ports; that is a production integration
requirement. Missing reduction axes are also rejected by native domain analysis.

Standard checkers ignore named-axis metadata. They do distinguish mass from length
and scalar from tensor. `sum` returns `Mass[float | Tensor[float]]` because the
checker cannot know whether axes remain. Ordinary `==` intentionally retains
Python's bool signature and allows unrelated operands. The generic `equal` helper
requires a shared invariant physical dimension and is rejected statically for
mass versus length. It returns bool for concrete values and a symbolic relation
while Bilt traces; that substitution remains internal to the adapter.

## Pointwise transforms

The unary `@pointwise` wrapper accepts a scalar quantity function and preserves
its physical dimension and scalar/tensor representation. It executes the scalar
kernel per coordinate, converts each result to the input unit, and supports empty
tensors without executing a synthetic sample. The kernel must preserve dimension;
this prototype does not support binary callable lifting or dimension-changing
pointwise wrappers. Native symbolic equations already demonstrate mixed
scalar/tensor lifting separately.

## Limits and next steps

- Units live on the experimental outer `Quantity`; Bilt's production runtime still
  uses its existing representation behind the adapter. This avoids conflating
  public API evidence with a completed production migration.
- The generated catalogue is finite, including only selected dimensions and units.
- Direct tensor arithmetic requires matching domains and shapes or a scalar
  operand. Bilt's compiler handles model alignment. Direct coordinate alignment
  is not implemented in this wrapper.
- Absolute temperature conversion/equality is tested. Absolute scaling and summing
  are rejected at runtime; the generic stubs do not yet encode every absolute-unit
  restriction. The arithmetic catalogue is not a complete units type system.
- No JAX validation, subsystem validation, tensor product reduction, generic
  shape typing or full unit-operator typing is claimed.
- In a production implementation, declaration parsing, domain constraints and
  boundary wrapping should move into Bilt's own compiler/evaluation APIs. The
  adapter demonstrates the behavior without changing those production modules.
