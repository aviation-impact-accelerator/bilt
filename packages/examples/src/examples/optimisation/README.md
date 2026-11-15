# Bilt + asdex + IPOPT cruise-speed optimisation

An experimental integration in `packages/examples/src/examples/optimisation/asdex_ipopt_cruise.py`.

This example optimises eight nondimensional aircraft cruise legs. Each leg has
an acausal Bilt model with drag, energy and journey-time equations:

- `drag = parasite_coefficient * speed**2 + induced_coefficient / speed**2`
- `energy = drag * distance`
- `distance = speed * time`

We minimise the sum of energies subject to each leg's maximum permitted time.
The first derivative comes from JAX; sparse constraint Jacobians and
Lagrangian Hessians come from **asdex**, then are passed to **IPOPT** using
`cyipopt.Problem`.

## Running

Bilt requires Python 3.15+. From the repo root, after `uv sync --extra jax`,
install the two *optional* optimisation dependencies into the development
environment:

```bash
uv pip install --python .venv/bin/python asdex cyipopt
uv run --no-sync python -m examples.optimisation.asdex_ipopt_cruise
uv run --no-sync pytest packages/examples/tests/test_asdex_ipopt_cruise.py
```

`cyipopt` also requires the system IPOPT library and build metadata. Where
they are unavailable, use an environment containing IPOPT, for instance
conda-forge's `ipopt` and `cyipopt` packages. We deliberately do not add
these dependencies to Bilt's default installation.

## Expected solution

With both drag coefficients equal to 1, drag is `v**2 + v**-2`,
which is minimised at `v = 1`. Since `time <= distance / required_speed`,
the constrained solution for each leg is

```python
speeds = np.maximum(1.0, np.asarray(REQUIRED_SPEEDS))
```

The sparse Jacobian of the per-leg time constraints has eight entries (one
per speed), and the lower-triangular Hessian of the Lagrangian has eight
entries. The test checks the solver against the analytic solution.

## Important limitation

The differentiated numerical function **does evaluate Bilt**, but it uses
`jax.vmap` over scalar Bilt leg evaluations. We do not pass traced arrays into
Bilt's `tensor()`: Bilt's `Tensor` currently uses `np.asarray`, which
cannot accept a JAX tracer during differentiation. Likewise, Bilt's
unit-carrying scalar output lifting calls `float()`. This is why the first
prototype is nondimensional and uses plain `float`-annotated fields.

For a more native optimiser we need a public interface to Bilt's compiled
numerical function and its input/output metadata, with JAX arrays kept inside
the differentiable graph. This is an integration exploration, not yet an
optimisation backend or a claim of end-to-end named-tensor differentiation.
