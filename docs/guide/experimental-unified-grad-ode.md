# Prototype: one `grad` interface for expressions and ODE states

This is an **experimental proof of concept**, stacked on the symbolic
expression derivatives in PR #437. It is deliberately narrower than the
archived November 2026 ODE prototype, which already supported more advanced
SCC-based causalisation and shared RHS planning.

## One syntax, two paths

Within ordinary Bilt equations:

```python exec="off"
grad(expression, wrt=x)
```

is still evaluated by the symbolic forward-mode JVP rules. For instance,
`grad(x**3, wrt=x)` produces an explicit algebraic expression.

A different pattern, where a **named tensor state** is differentiated with
respect to a **distinct tensor coordinate**, stays unresolved:

```python exec="off"
grad(position, wrt=time)
```

That is represented as a first-class `DEFERRED_GRAD(position, time)` tree,
rather than being incorrectly simplified to zero. Together with a matching
`at(position, wrt=time, where=t0)` initial condition and an explicit
rate equation, it can be classified as an ODE by the surrounding
`Implementation`.

**This is a limited recognition rule, not yet complete inference from all
physical dependencies.** It deliberately supports named, first-order states;
arbitrary derivatives of composite time-dependent expressions must not
silently become zero and are rejected if unsupported.

## Worked example: spring–mass oscillator

These are the **runnable public operators from this prototype**.
`grad` describes both an explicit energy derivative and formal time rates:

```python exec="off"
import numpy as np
from bilt import (
    Implementation,
    Interface,
    Tensor,
    at,
    dim,
    equation,
    field,
    grad,
    system,
    tensor,
)

TIME = dim("time")


@system
class Oscillator(metaclass=Interface):
    t: Tensor = field()
    x: Tensor = field()
    v: Tensor = field()
    stiffness: float = field()
    mass: float = field()
    t0: float = field()
    x0: float = field()
    v0: float = field()


@equation
def position_rate(x: Tensor, v: Tensor, t: Tensor) -> bool:
    return grad(x, wrt=t) == v


@equation
def velocity_rate(x: Tensor, v: Tensor, t: Tensor, stiffness: float, mass: float) -> bool:
    potential = 0.5 * stiffness * x * x
    force = -grad(potential, wrt=x)
    return grad(v, wrt=t) == force / mass


@equation
def initial_x(x: Tensor, t: Tensor, t0: float, x0: float) -> bool:
    return at(x, wrt=t, where=t0) == x0


@equation
def initial_v(v: Tensor, t: Tensor, t0: float, v0: float) -> bool:
    return at(v, wrt=t, where=t0) == v0


implementation = Implementation(Oscillator, [position_rate, velocity_rate, initial_x, initial_v])

with implementation:
    oscillator = Oscillator(
        t=tensor(np.linspace(0, np.pi, 101), over=TIME),
        stiffness=4.0,
        mass=1.0,
        t0=0.0,
        x0=1.0,
        v0=0.0,
    )
    position = oscillator.x
    velocity = oscillator.v
```

The first requested state triggers one common SciPy `solve_ivp` call
for **both states**; the instance publishes and caches both complete
trajectories over the named time domain. No solver callback or state
packing is written by the modeller.

SciPy is imported lazily and is required to execute trajectories. Only
the Python backend is supported for integration; the ordinary expression
JVP implementation still works independently in Python and JAX.

## How this relates to the archived ODE prototype

The archived `bilt-ode-prototype.zip` demonstrated:

- An explicit differential/point-condition IR (`derivative.total` and
  `at`), and extraction of named rates.
- Consistent **full-scope** structural validation of differential and
  algebraic constraints, so DAEs cannot be accidentally accepted.
- Shared multi-output SCC planning of the instantaneous RHS with
  algebraic intermediates and coupled linear blocks.
- A semantic `ODESolve` region, isolated from SciPy numerical execution.
- One integrated solve for coupled states, and instance-level result caching.

This spike keeps the **shared solve and caching** but implements only direct
explicit RHS expressions. It does **not** port the archived SCC planner,
constraint solver, full-scope validation, integration options, or qualified
subsystems. It actively rejects additional algebraic state constraints
rather than ignoring them.

## Representation and semantics

```text
@equation trace
    |
    +-- grad(explicit_expression, wrt=its_symbol)
    |       -> ordinary symbolic JVP
    |
    +-- grad(named_tensor_state, wrt=separate_tensor_coordinate)
    |       -> DEFERRED_GRAD(state, coordinate)
    |
    +-- at(state, wrt=coordinate, where=start)
            -> AT(state, coordinate, start)
    |
Implementation-level classification:
    group all formal rates + initial conditions
    ensure one shared coordinate and one condition per state
    reject conflicting / unsupported additional state constraints
    |
    v
One compiled, explicit, scalar instantaneous RHS
    |
    v
SciPy RK45 for all coupled states together
    |
    v
Bilt Tensor trajectories with the named time domain
```

There is no spatial differentiation or PDE support in this prototype.

### Important limitations

1. Only **local, named first-order tensor-state rates** and one independent
   coordinate are supported. The instantaneous state values and all model
   parameters must be scalar. The output times form a one-dimensional
   increasing tensor.
2. Rate equations must already be in explicit form
   `grad(state, wrt=time) == rhs`. Algebraic coupling and extra constraints
   involving evolving states are rejected; restoring those capabilities
   requires the archived full-scope acausal planner.
3. Arbitrary derivatives of composite time-dependent expressions such as
   `grad(x*x, wrt=t)` are preserved as formal expressions but are
   **not yet lowered using the chain rule**. Such rates are rejected
   rather than returned as zero. Mixed explicit/time-dependent expressions
   need an explicit dependency-classification pass before production.
4. Bare numeric scalar parameters only: physical units on ODE rates,
   `QuantityTensor` trajectories, named tensor-state axes, sampled forcing
   and user-configurable solver tolerances are future work.
5. Python/SciPy only for solving trajectories. JAX has no ODE execution
   integration in this PR. Instance-level parameter derivatives through
   the entire ODE trajectory are also not yet supported.
6. This is a proof of concept for the public notation, **not** a proposal
   to merge the narrow ODE solver instead of the archived mature SCC
   integration. The larger solver should be reused or adapted.

## Tests

`tests/api/test_deferred_grad_ode_prototype.py` covers numerical decay,
the oscillator with an energy-derived restoring force, Lorenz's three-state
RHS, one-solve caching, backend restrictions, missing initial conditions,
invalid sampling and rejection of unsupported extra constraints.

The next production PR should transplant the formal derivative/point
constraint nodes into the current structural analysis pipeline, port
the archived explicit ODE validation and multi-output RHS planner, and
replace this dedicated narrow execution path with a semantic `ODESolve`
region. Do not special-case `grad` by variable spelling or silently
differentiate an unknown trajectory as if it were an independent symbol.
