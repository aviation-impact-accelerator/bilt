# Quantity arithmetic typing experiment

This is a **typing probe**, stacked on PR #403. It doesn't change Bilt's public API, equations, or runtime behaviour.

## What we're checking

PR #403 adds operator suppressions because flexible system attributes expose the read-only `QuantityLike` protocol. It has no arithmetic methods. The earlier suggestion of simply adding `__add__(object)` and `__mul__(object)` would be misleading.

The positive fixture checks that a *narrow* scalar arithmetic protocol matches the existing concrete `Quantity` methods, and that a separately declared **scalar** subsystem view can express the original thermal equation without suppressions.

The negative fixture checks three limitations with both ty and Pyrefly:

- A protocol accepting **any** operand is not implemented by `Quantity.__add__(Quantity)`. Method parameters are not safely widened by a structural protocol.
- A scalar arithmetic protocol must not accept `QuantityTensor` as a scalar.
- A flexible `QuantityLike` field doesn't automatically gain a scalar arithmetic interface, even if its runtime value happens to be a `Quantity`.

There is an additional asymmetry: `QuantityTensor` inherits `Tensor` operators, many of which statically return `Tensor` with an optional unit. These do not satisfy a general arithmetic protocol promising a unit-bearing return. In normal Python, concrete tensor arithmetic is intentionally unsupported outside Bilt equations.

## Decision this informs

**A single arithmetic protocol is not, by itself, a replacement for `QuantityLike` at system boundaries.** It either won't be implemented by the concrete classes or will claim operations valid that aren't supported outside Bilt's symbolic equations. Broadening operator signatures just to pass the checkers would weaken typing.

The remaining design problem is different: the same system annotation is used as both an **input boundary contract** and the **statically visible expression type** of a nested attribute. If we want flexible inputs, sound arithmetic typing, and just one physical-dimension alias, we need to solve that distinction explicitly. Options to explore next are a compiler-aware equation view or an explicit symbolic expression API; neither is established by this experiment.

## How to verify

From the repository root, with the project's dev dependencies:

```sh
uv run ty check
uv run pyrefly check
uv run pytest tests/typing/test_negative_quantity_checkers.py tests/typing/test_quantity_arithmetic_prototype.py
```

The normal positive checks must remain clean and the negative fixture must continue to produce *only* the expected diagnostics. This prototype does not remove any #403 suppressions.
