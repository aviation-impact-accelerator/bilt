# Prototype: scalar-annotated equations and tensor lifting

Follow-up to [#367](https://github.com/aviation-impact-accelerator/bilt/issues/367) and [#390](https://github.com/aviation-impact-accelerator/bilt/pull/390).

**Question:** Can `Quantity[float]` express the *local scalar operation* inside an
equation while Bilt still lifts it across tensor-valued system inputs? This
tests the opposite of interpreting `Quantity[float]` as a blanket prohibition
on tensor inputs at every site.

This prototype uses **Bilt's real `@system`, `@equation`, `Implementation`,
symbolic analysis, unit conversion and lifting**. All implementation code is
isolated in `tests/core/test_quantity_lifting_prototype.py`. No public API,
production module or interpreter behaviour is changed.

## Proposed meanings

| Annotation | In an equation | At a system boundary |
| --- | --- | --- |
| `Quantity` | Generic quantity; ordinary symbolic expression | Scalar or unit-bearing tensor |
| `Quantity[float]` | Scalar-local expression; **liftable** over outer tensor axes | Strictly scalar |
| `Quantity[Tensor]` | Tensor expression; named reductions/indexing available | Unit-bearing tensor required |

Dimensional metadata remains in `typing.Annotated`, as in Bilt today:

```python
type Mass = Annotated[Quantity, MASS]
type ScalarMass = Annotated[Quantity[float], MASS]
type TensorMass = Annotated[Quantity[Tensor], MASS]
```

The distinction is about the interpretation of a function's *local*
signature versus the concrete type of its system boundary.

## What is tested

The experiment defines `prototype_equation` and `prototype_system`,
which translate these annotations to Bilt's **current** runtime ones when
decorating a function or class:

- `Quantity` and `Quantity[float]` become `bilt.Quantity`. The real Bilt
  compiler creates ordinary `Expr` symbols that can be lifted.
- `Quantity[Tensor]` becomes `bilt.Tensor`. The real compiler creates
  `TensorExpr` symbols with named reduction support.
- `Annotated` metadata is preserved for Bilt's actual dimensional checking.
- The decorators restore the original annotations after Bilt has compiled the
  equation/system, so the proposed signatures remain inspectable.

The adapter does not implement a new evaluator. Tests exercise the real
`Implementation` and verify:

1. A scalar-annotated mass equation gives a scalar answer for a scalar input
   and a unit-bearing tensor answer for a tensor input.
2. One scalar-annotated equation combines a tensor and scalar quantity, with
   unit conversion and lifting.
3. Two tensor inputs over disjoint axes broadcast to their union.
4. A tensor-annotated mass reduction consumes an inner `time` axis while
   lifting over an outer `scenario` axis.
5. The proposed strict-scalar and tensor-only boundary restrictions can be
   applied through `prototype_instantiate` as an **opt-in** check.

## Limitations and questions

The `Quantity[M]` type in the test file is **an annotation-only mock**,
not the implementation from #390. It supplies arithmetic signatures for
ty/Pyrefly but is never constructed or called by Bilt at runtime.

Bilt currently constructs native scalar `Quantity` or `Tensor` values, not
the proposed `QuantityTensor`. Accordingly, passing native tensors to a
system field annotated with the experimental `Quantity` requires targeted
static typing ignores in the tests. This is a genuine unresolved boundary
typing problem, not something the adapter solves.

`prototype_instantiate` does *not* alter the real system constructor. The
tests therefore establish proposed scalar-only and tensor-only restrictions,
but enforcing them in production would require a real boundary validation
policy, including decisions about lifting output ports.

The adapter also does not enforce declared tensor *minimum domains*. That
would require Bilt's domain-aware annotation metadata checks, independently
of whether the generic is covariant.

These tests can tell us if the semantics are compatible with **Bilt's current
symbolic execution**, but cannot prove the design will work with Python type
checkers at system call sites. The latter requires the quantity hierarchy,
`Tensor` implementation and public system constructor typing work from #390.

## Run

```sh
uv run pytest -q tests/core/test_quantity_lifting_prototype.py
uv run ty check tests/core/test_quantity_lifting_prototype.py
uv run pyrefly check tests/core/test_quantity_lifting_prototype.py
```

Nothing in this PR should be merged as a production implementation; it is a
comparison experiment before choosing annotation semantics.
