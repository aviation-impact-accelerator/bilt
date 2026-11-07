# Bilt to GT4Py: Cartesian grid subsystem prototype

This example keeps GT4Py outside Bilt core. It uses ordinary Bilt
`@system`, `@equation`, `@transform`, and `Implementation` APIs.

The grid is a real Bilt subsystem:

```python
@system
class CartesianGrid2D(metaclass=Interface):
    dx: float = field()
    dy: float = field()
    inv_dx2: float = field()
    inv_dy2: float = field()
```

Its implementation computes `inv_dx2 = 1/dx**2` and
`inv_dy2 = 1/dy**2` with ordinary transforms. The `Diffusion` parent
connects its spacing ports and the grid's derived coefficients with
ordinary equations, and registers the child using
`internal_implementations={"grid": grid_implementation}`.

The diffusion transform receives the aligned input array and derived
coefficients, uses `CartesianFieldMapping` to map `(x, y)` to GT4Py
`(I, J)`, pads a periodic one-cell halo temporarily, evaluates a
five-point Laplacian on every physical cell and strips the halo before
returning data to Bilt.

Run with a compatible GT4Py Next installation:

```sh
python -m examples.gt4py.diffusion_example
```

The example deliberately supplies a temperature tensor in reversed
`(y, x)` order to exercise the adapter.

The grid subsystem defines physical grid spacing. Periodic boundary
handling and the one-cell halo remain **implementation choices**, not
Bilt ports. Halos never change the Bilt tensor's physical extents.
A future interface could expose boundary data as actual model values,
if different model instances need different physical conditions.

`to_runtime_field` and `to_runtime_array` currently use Bilt's
**private** evaluation context to recover canonical axis order. This
is a temporary prototype detail rather than a stable public API.

Current limitations: only exact two-dimensional field mapping in the
diffusion example; no staggered grids, batch dimensions, unit-aware grid
spacing, zero-copy guarantees, JAX tracing or differentiation. The
separate axis-mapping tests do not require GT4Py; the integration tests
require a working GT4Py installation.
