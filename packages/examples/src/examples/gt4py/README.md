# Bilt to GT4Py field mapping prototype

This experiment keeps GT4Py outside Bilt core. `CartesianFieldMapping`
maps Bilt named dimensions to GT4Py Next field dimensions and reorders arrays
when the source domain differs from the kernel's axis order.

Install a compatible GT4Py Next release separately (GT4Py is **not** a Bilt
dependency), then run:

```sh
python -m examples.gt4py.diffusion_example
```

The example supplies a Bilt tensor ordered `(y, x)` to a GT4Py kernel
expecting `(I, J)`, and restores the result to Bilt's evaluation-wide order.
It computes the interior of a five-point stencil with zero-filled outer cells;
the latter are placeholders, **not** physical boundary conditions.

Direct conversion is also possible using
`mapping.to_field(bilt_tensor)` and
`mapping.to_tensor(gt4py_field, domain=bilt_tensor.domain)`.
For raw arrays, supply the actual Bilt domain explicitly.

The `to_runtime_field` and `to_runtime_array` helpers use Bilt's **private**
evaluation context to recover its canonical axis order. They are a temporary
proof of concept, not a stable public integration API.

Current limitations: strict one-to-one dimension mapping; no halo handling,
batch dimensions, staggered grids, unit conversion, zero-copy guarantees,
JAX tracing or differentiation. All field axes need to be explicitly mapped.
