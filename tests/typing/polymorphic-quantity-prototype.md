# Scalar-or-tensor annotations: union prototype

**Experimental, stacked on #403.** This tries the direct alternative to adding a new expression protocol.

Instead of defining an arithmetic-enabled \u0060QuantityLike\u0060 that concrete classes do not implement, describe a pointwise physical quantity as the union of the two existing concrete representations:

\u0060\u0060\u0060python
Temperature = Annotated[Quantity | QuantityTensor, TEMPERATURE]
\u0060\u0060\u0060

Use the **same alias** on system fields and pointwise equation parameters. In equations, Bilt already replaces parameters with symbolic \u0060Expr\u0060 objects, so the concrete annotation is not the object actually passed during symbolic construction. A union declaration still yields a pointwise symbolic expression, lifted over named tensor domains as necessary.

A separate \u0060Annotated[QuantityTensor, DIM]\u0060 declaration is retained where an equation explicitly needs \u0060.sum(dim)\u0060, \u0060.isel(dim, i)\u0060 or other tensor operations.

## Checks

- One alias used on both systems and equations; scalar and tensor runtime inputs both accepted.
- Nested \u0060thermal_supply\u0060 expression exactly as in #403, **without suppressions**.
- Tensor reduction stays explicitly typed as \u0060QuantityTensor\u0060.
- Negative typing rejects unitless \u0060Tensor\u0060 as a unit-bearing quantity and rejects \u0060.sum()\u0060 on a generic pointwise quantity.
- Existing negative typing cases should remain unchanged apart from the additional assertions.

Run \u0060uv run ty check\u0060, \u0060uv run pyrefly check\u0060 and \u0060uv run pytest tests/typing\u0060.

## Deliberate limitations

The union is **not** a new eager arithmetic API. Concrete tensor arithmetic still raises outside Bilt equations. A static checker that accepts mixed \u0060Quantity\u0060/\u0060QuantityTensor\u0060 operators doesn't prove those operations can run eagerly, or that their physical dimensions match. Bilt already checks the latter when constructing the symbolic tree. If type checkers accept these operations too broadly, this design may not meet the original goal of rigorous static operation checking.

The port parser accepts only the **exact** two-type union, not arbitrary unions. This test is for pointwise equations, not \u0060@transform\u0060 or domain-constrained tensor fields. It is not yet a suggested production migration.


## Domain hints

On a pointwise equation, `Annotated[Quantity | QuantityTensor, DIM, Domain(...)]`
is an **exact** linear solver hint, as the previous scalar `Quantity` form
was. A tensor-specific `Annotated[QuantityTensor, DIM, Domain(...)]`
still means a **minimum** required set of axes on a selected equation.
Generic pointwise system fields don't accept a minimum-domain constraint.

## Eager arithmetic caveat (confirmed)

The same union can be used in regular Python, and both type checkers accept
`value * 2.0` even when `value` may be a `QuantityTensor`.
Eager `Tensor` arithmetic raises `NotImplementedError` by design.
This is not statically sound for eager use: Python typing has no way to
change permitted operators solely because the expression occurs inside an
`@equation` function. We shouldn't claim otherwise. To enforce this
statically would need an equation-only type plus a separate boundary type,
a checker plugin, or eager tensor arithmetic with well-defined semantics.
The new runtime test preserves the explicit rejection rather than
silently treating the union as a general eager arithmetic API.
