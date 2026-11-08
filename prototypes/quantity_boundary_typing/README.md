# Quantity system-boundary typing prototype

Follow-up to [#367](https://github.com/aviation-impact-accelerator/bilt/issues/367),
[#390](https://github.com/aviation-impact-accelerator/bilt/pull/390) and
[#391](https://github.com/aviation-impact-accelerator/bilt/pull/391).

**Question:** Can both ty and Pyrefly accept a *real Bilt* system constructor
with scalar and tensor quantity inputs through one generic quantity annotation,
without `cast`, `Any`, or static-checker suppressions?

The experiment lives in `tests/core/test_quantity_boundary_typing_prototype.py`.
It uses a covariant structural `QuantityLike[M]` protocol with read-only
`value` and `unit` properties. Native Bilt `Quantity` already satisfies it.
A small `QuantityTensor(Tensor)` subtype adds `value: Tensor` and a
non-optional `unit: Unit`. It is returned by an explicitly typed factory.

This is deliberately different from the **nominal inheritance** experiments
in #390: Bilt's current scalar `Quantity` is marked `@final`, so a
structural protocol is an interesting compatibility alternative.

The prototype translates generic field annotations to existing native
declarations *only while running Bilt's real* `@system` decorator, restoring
the declared annotations afterwards. The real `Interface` metaclass already
uses `@dataclass_transform`, so the synthesized constructor type is inferred
from the original fields. The actual system construction, compilation,
equation lifting and evaluator are Bilt's production machinery.

The positive examples deliberately use **no type suppressions**, including
`MassScale(mass=NativeQuantity(...))` and
`MassScale(mass=quantity_tensor(...))` with a shared `Mass` annotation.
The equation uses native scalar Bilt quantity annotations because #391
already tests their experimental translation; the concern here is **system
input typing**, not another equation compiler adapter.

## Expected checker behaviour

| Value | Field typed as `QuantityLike` | `QuantityLike[float]` | `QuantityLike[Tensor]` |
| --- | --- | --- | --- |
| Native scalar Quantity | Accept | Accept | Reject |
| Unit-bearing QuantityTensor | Accept | Reject | Accept |
| Unitless Tensor | Reject | Reject | Reject |

The positive cases are checked as part of normal repository-wide ty/Pyrefly
CI, and executed through Bilt's actual system. Negative examples are documented below but intentionally not
included in passing type-check fixtures.

### Limits

- The factory is a separate `quantity_tensor` function, rather than an
  overloaded production `tensor`; the latter should be investigated as a
  follow-up without making incorrect promises about actual runtime return types.
- The resulting evaluated field is still **statically** `QuantityLike`, even
  if the real value is a tensor. Propagating inferred tensor domains from
  model inputs through arbitrary equations is a separate problem.
- A structural protocol only proves the *minimal shared value/unit interface*.
  It does not yet offer Bilt's complete quantity arithmetic interface, tensor
  reduction methods, or the nominal `isinstance(x, Quantity)` behaviour.
- No production `Quantity`, `Tensor`, `Unit`, `Interface` or decorators
  are changed. This is an exploratory API, not a proposed production migration.

Run:

```sh
uv run pytest -q tests/core/test_quantity_boundary_typing_prototype.py
uv run ty check tests/core/test_quantity_boundary_typing_prototype.py
uv run pyrefly check tests/core/test_quantity_boundary_typing_prototype.py
```

## Negative typing examples

These statements should be rejected by a type checker when placed in a
separate failing-check fixture (they are not executed by CI):

```python
unitless = tensor([1.0], over=time)
scalar = NativeQuantity(1.0, kilogram)
labelled = quantity_tensor([1.0], over=time, unit=kilogram)

accepts_mass(unitless)  # Reject: unit may be None and no value property
accepts_scalar(labelled)  # Reject: tensor magnitude is not float
accepts_tensor(scalar)  # Reject: scalar magnitude is not Tensor
```

The next step is to wire those up as expected-failure fixture checks so a
regression cannot silently make them valid.
