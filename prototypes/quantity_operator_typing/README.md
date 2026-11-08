# Quantity operator typing: PR4

This is stacked on [#395](https://github.com/aviation-impact-accelerator/bilt/pull/395) and remains a **draft for team review**.

The previous PR introduces `QuantityLike[M]` as a minimal, honest common
interface. That contract cannot promise concrete arithmetic on tensors:
`QuantityTensor` deliberately inherits Bilt's `Tensor` API, where arithmetic
is dispatched during symbolic evaluation, and concrete tensor arithmetic
outside a model raises `NotImplementedError`.

For PR4, test two *narrower structural capability protocols* rather than
giving bare `QuantityLike` operators that its tensor implementation doesn't
satisfy:

- `ScalarQuantityLike`: read-only `float` magnitude plus basic scalar
  quantity operations (`+`, `-`, scalar multiplication/division and negation).
  The existing concrete `Quantity` should satisfy this structurally.
- `TensorQuantityLike`: `Tensor` magnitude and named reduction/selection
  operations. The existing concrete `QuantityTensor` should satisfy this;
  its reductions are **symbolic** in equations, not eager NumPy operations.

Normal system fields can still use `Annotated[QuantityLike, MASS]`.
Equations can request `Annotated[ScalarQuantityLike, MASS]` for scalar-local
arithmetic (which Bilt lifts) or `Annotated[TensorQuantityLike, MASS]`
when a named tensor operation is actually needed.

This makes the distinction between **typed capabilities** and **concrete
operations** explicit, without changing how the compiler executes equations.

## Current limitations

- `QuantityLike[float]` and `QuantityLike[Tensor]` still expose only the
  shared value/unit contract. Arithmetic and reduction methods require
  the more specific capability protocols for now. We should decide if this
  extra public API surface is preferable to a larger generic protocol after
  reviewing the typing results.
- The tensor reduction methods return `Tensor`, matching today's concrete
  `Tensor` method signatures. The compiler—not Python typing—tracks which
  dimensions remain and which unit the output has.
- This experiment does not make multiplication of two generic quantities or
  general tensor arithmetic work outside an active Bilt backend.
- It also does not attempt dimensional arithmetic in ty or Pyrefly. Bilt
  validates `Annotated` dimensions and equation consistency.
- `ScalarQuantityLike` can only guarantee that a value supports the
  stated operations, not that a type checker can prove two quantities have
  the same dimensions; Bilt keeps that validation.
- Negative type-checking examples are below, but must be added as
  expected-failure checker fixtures before this can be considered complete.

Examples of calls **the static checkers should reject**:

```python
def scalar_only(mass: ScalarQuantityLike) -> None:
    mass.sum(time)  # No tensor reduction on scalar quantity

def tensor_only(mass: TensorQuantityLike) -> None:
    2.0 * mass  # No concrete scalar multiplication guaranteed on tensors

def generic_only(mass: QuantityLike) -> None:
    mass.sum(time)  # Unavailable without narrowing to tensor capability
```

The positive fixture is run by normal repo-wide ty and Pyrefly checks:
`tests/api/test_quantity_operator_typing.py`.
