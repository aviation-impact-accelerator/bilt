# Bilt + asdex + IPOPT cruise-speed optimisation

Experimental integration in `asdex_ipopt_cruise.py` using a named `leg`
dimension. The Bilt model describes acausal drag, energy, and travel time:

- `drag = parasite_coefficient * speed**2 + induced_coefficient / speed**2`
- `energy = drag * distance`
- `distance = speed * time`

Eight independent cruise legs minimise total nondimensional drag energy, subject
to maximum travel time per leg. Since `drag = speed**2 + speed**-2`, the
analytic constrained solution is `maximum(1, required_speed)` per leg.

## Integration boundary

The point of this prototype is to keep **structural analysis out of the
differentiated function**:

1. Prepare concrete Bilt tensors with the named `leg` dimension.
2. Bind the domains and ask Bilt for a backend-specific compiled evaluation for
   each output with `Implementation.get_evaluation`.
3. Lower numerical inputs and align named dimensions once.
4. Use only the compiled `BackendCallable` objects with JAX array arguments in
   objective and constraint functions.
5. Differentiate the resulting JAX function using JAX and asdex, and pass sparse
   Jacobian/Hessian callbacks to IPOPT through cyipopt.

No `Implementation.evaluate`, structural analysis, `Quantity` lifting or
`Tensor` construction takes place inside JAX's traced objective/constraints.
This prototype uses Bilt's internal compilation API and plain
nondimensional numerical fields; it is *not* proposing direct differentiation of
Bilt's Python object model. The integration suggests exposing a stable public
compile-and-bind API in Bilt.

## Run

Bilt currently requires Python 3.15+. Install the optional JAX backend and
asdex/cyipopt in an environment containing the system IPOPT library:

```bash
uv sync --extra jax
uv pip install --python .venv/bin/python asdex cyipopt
uv run --no-sync python -m examples.optimisation.asdex_ipopt_cruise
uv run --no-sync pytest packages/examples/tests/test_asdex_ipopt_cruise.py
```

CI runs the JAX compilation/gradient test without requiring IPOPT, and only
runs the IPOPT integration test when `asdex` and `cyipopt` are installed.

## Open questions

- Verify asdex's sparse coordinate ordering stays fixed between IPOPT
  structure/value callbacks.
- Compare its sparse derivatives with dense JAX derivatives.
- Consider how multiple outputs should be compiled as one shared numerical
  graph without reconstructing independent evaluations.
- Expose fixed unit scales and domain metadata in a future stable public
  numerical compilation interface.
