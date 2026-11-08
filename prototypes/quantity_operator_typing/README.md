# Quantity operator typing: simplifying the PR4 experiment

PR4 ([#396](https://github.com/aviation-impact-accelerator/bilt/pull/396))
experimented with distinct `ScalarQuantityLike` and
`TensorQuantityLike` protocols. PR6 replaces those with the existing
concrete types for **equation operations**, while retaining a single minimal
`QuantityLike[M]` protocol for **system fields**.

```python
from typing import Annotated

from bilt import Quantity, QuantityLike, QuantityTensor
from bilt.units.dimensions import MASS

type Mass = Annotated[QuantityLike, MASS]
type ScalarMass = Annotated[Quantity, MASS]
type TensorMass = Annotated[QuantityTensor, MASS]
```

- `QuantityLike` says that a system input can be a scalar or tensor with a
  physical unit. Its structural properties are `.value` and `.unit`.
- `Quantity` offers existing scalar arithmetic (including `2.0 * mass`)
  when authoring a scalar-local equation. Bilt still automatically lifts
  these equations over additional tensor dimensions.
- `QuantityTensor` offers the concrete tensor API, including named
  reductions and selections. These operations are symbolic inside equations;
  eager tensor arithmetic remains unsupported.
- `QuantityLike[float]` and `QuantityLike[Tensor]` remain available for
  strict, typed system boundaries. Their common interface intentionally
  does **not** expose scalar arithmetic or tensor reductions.

Existing annotations `Annotated[Quantity, DIM]` and
`Annotated[Tensor, DIM]` remain supported; this PR does not change legacy
`Quantity` boundary semantics.

## Validation

The positive tests in `tests/api/test_quantity_operator_typing.py`
exercise Bilt's actual `@equation`, `@system`, and `Implementation`.
They verify scalar arithmetic, lifting over tensor inputs, and explicit
tensor reductions over a named axis without static suppressions.

The negative checker cases in `tests/typing/` verify mismatches involving
unitless tensors, scalar and tensor quantities, concrete system constructors,
and operations outside the minimal generic contract. Both ty and Pyrefly must
reject these cases at the expected source lines.

## Outstanding choices

This is still a draft awaiting team review. In particular:

- Python's type system cannot represent all domain and unit-dimension
  constraints; Bilt must validate these itself.
- Using a concrete `Quantity` annotation on an equation intentionally
  differs from using `QuantityLike` on the enclosing system field.
- The user-facing names `QuantityLike` and `QuantityTensor` have not yet
  been approved as the final public API.
- The original PR4 extra protocols are not required for the tested equation
  operations, but more advanced operations may lead to further decisions.
