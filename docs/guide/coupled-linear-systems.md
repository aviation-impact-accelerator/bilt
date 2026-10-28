## Coupled linear systems

So far, our computation plans have been chains of calculations.

One relationship produces a value. Another relationship uses it. Each step can run once its inputs are known.

But some equations need to be solved together.

Consider:

\[
x+y=p
\]

\[
x-y=q
\]

If `p` and `q` are known, neither equation can calculate `x` without `y`, or `y` without `x`.

Together, however, they determine both values.

Bilt calls this group a **coupled block**. When the block is linear in its unknowns, Bilt can assemble and solve a linear system for it.

The surrounding model can still contain ordinary equations and transforms. Only the coupled part needs a simultaneous solve.

### Availability on the feature branch

This chapter describes the SCC analyser on the coupled-solve feature branch.

The branch still uses the exhaustive analyser by default. Selecting `PYTHON_BACKEND` or `JAX_BACKEND` chooses how calculations run; it does not select the structural analyser. There is currently no public `analyser_type` option.

For this chapter's executable examples, use the following temporary development setup. It selects SCC only inside the context and restores the previous selection afterwards. These internal imports are a feature-branch workaround, not a supported public selection API.

```python exec="on" source="above" session="linear-systems"
from unittest.mock import patch

import numpy as np

from bilt import Implementation, Interface, PYTHON_BACKEND, field, system, equation
from bilt.core._structural_analysis._causalization_block_solver._backend import (
    CausalizationStructuralAnalyzer,
)
from bilt.core._system import _implementation as implementation_module


def coupled_analysis():
    return patch.object(
        implementation_module,
        "ExhaustiveStructuralAnalyser",
        CausalizationStructuralAnalyzer,
    )
```

The model definitions themselves use the ordinary Bilt API.

### Start with two unknowns

We declare four values and two equations:

```python exec="on" source="above" session="linear-systems"
@system
class Pair(metaclass=Interface):
    x: float = field()
    y: float = field()
    p: float = field()
    q: float = field()


@equation
def total(x: float, y: float, p: float) -> bool:
    return x + y == p


@equation
def difference(x: float, y: float, q: float) -> bool:
    return x - y == q


pair_model = Implementation(Pair, [total, difference])

with coupled_analysis():
    scalar_inputs = {"p": 10.0, "q": 2.0}
    x = pair_model.evaluate(scalar_inputs, "x", PYTHON_BACKEND)
    y = pair_model.evaluate(scalar_inputs, "y", PYTHON_BACKEND)

assert np.isclose(x, 6.0)
assert np.isclose(y, 4.0)
```

We did not rearrange the equations or declare a matrix.

Bilt identifies that `x` and `y` depend on one another, extracts their coefficients, and solves the block.

Although we requested one value at a time, each request can require both unknowns to be solved together.

We use the same original input dictionary for these separate requests. This avoids treating a previously calculated output as an extra known value in the next request; see the [current instance-caching restriction](../api/linear-systems.md#requesting-several-block-outputs).

### A different request can give a different plan

Now supply `x` and `q`, and ask for `p`:

```python exec="on" source="above" session="linear-systems"
with coupled_analysis(), PYTHON_BACKEND, pair_model:
    inverse = Pair(x=6.0, q=2.0)
    assert np.isclose(inverse.p, 10.0)
```

For this request, Bilt can calculate `y=x-q`, then `p=x+y`.

There is no need for a coupled solve.

The equations did not change. The values supplied changed what remained unknown, and therefore how the model could be evaluated.

### Linear in which unknowns?

A block is linear **in the values being solved together**.

For example:

\[
a x+y=p
\]

is linear in `x` and `y` when `a` is supplied.

The coefficient `a` might itself have been calculated by a nonlinear relationship earlier in the plan. That does not make this block nonlinear in `x` and `y`.

But if both `a` and `x` must be solved inside the block, their product is nonlinear. The current linear solver cannot handle that formulation.

This distinction matters in engineering models. A compressor pressure-ratio expression can be nonlinear while a later temperature and work balance is linear in its remaining unknowns.

### Independent solves over a domain

Suppose `p` varies over operating points, while `q` is the same at each point:

```python exec="on" source="above" session="linear-systems"
from bilt import dim, domain, tensor

operating_point = dim("operating_point")

with coupled_analysis():
    pointwise = pair_model.evaluate(
        {"p": tensor([10.0, 14.0, 18.0], over=operating_point), "q": 2.0},
        "x",
        PYTHON_BACKEND,
    )

assert pointwise.domain == domain(operating_point)
np.testing.assert_allclose(pointwise.values, [6.0, 8.0, 10.0])
```

There is a separate pair of unknowns at each operating point.

Bilt can solve the same two-by-two system in a batch. No equation connects one operating point to another.

Broadcasting `q` means using its supplied value in every calculation. It does not introduce a new shared unknown.

### One unknown can be shared

Now consider a different pair of relationships:

\[
x_t+y=p_t
\]

\[
\tfrac12\sum_t x_t-y=q
\]

Here `x` varies over time, while the same unknown `y` appears at every time point.

The sum connects all those points. We cannot solve each point independently.

```python exec="on" source="above" session="linear-systems"
from bilt import Tensor

time = dim("time")


@equation
def shared_balance(x: Tensor, y: float, p: Tensor) -> bool:
    return x + y == p


@equation
def aggregate(x: Tensor, y: float, q: float) -> bool:
    return 0.5 * x.sum(time) - y == q


shared_model = Implementation(Pair, [shared_balance, aggregate])
shared_inputs = {"p": tensor([2.0, 4.0, 6.0], over=time), "q": 1.0}

with coupled_analysis():
    varying = shared_model.evaluate(shared_inputs, "x", PYTHON_BACKEND)
    shared = shared_model.evaluate(shared_inputs, "y", PYTHON_BACKEND)

assert varying.domain == domain(time)
np.testing.assert_allclose(varying.values, [0.0, 2.0, 4.0])
assert np.isclose(shared, 2.0)
```

There are four scalar unknowns: three values of `x` and one value of `y`.

Bilt assembles one four-by-four system. It does not duplicate the shared unknown at every coordinate.

The same idea applies when a shared value varies over only some axes. For example, `x(site,time)` can be coupled to `y(site)` by a reduction over time.

### How does Bilt know what varies?

In the previous example, we did not provide `Domain` hints for the unknowns.

Bilt combines the known input domains with the equations and reduction axes. It looks for a consistent layout and, where possible, chooses the least layout: one that introduces no variation beyond what every other supported layout requires.

Here, `x.sum(time)` requires `x` to contain `time`. The remaining relationships allow `y` to be shared. That gives the inferred result above.

A Python `float` annotation alone is not an exact scalar-domain constraint. Use `Domain.scalar()` if the unknown must be shared.

Inference does not invent a dimension's size. The named axes and their concrete extents must already be established by known block inputs.

### When to add a hint

Sometimes the equations allow different layouts without a single least choice.

For example, a balance involving `x.sum(site)` and `y.sum(time)` can admit different choices about the other axes of `x` and `y`.

In that case Bilt asks for a domain hint rather than choosing arbitrarily.

A hint states which dimensions an unknown has:

```python exec="on" source="above" session="linear-systems"
from typing import Annotated

from bilt import Domain

OverTime = Annotated[Tensor, domain(time)]
Shared = Annotated[float, Domain.scalar()]


@equation
def hinted_aggregate(x: OverTime, y: Shared, q: float) -> bool:
    return 0.5 * x.sum(time) - y == q


hinted_model = Implementation(Pair, [shared_balance, hinted_aggregate])

with coupled_analysis():
    hinted = hinted_model.evaluate(shared_inputs, "y", PYTHON_BACKEND)

assert np.isclose(hinted, shared)
```

This replaces the unannotated aggregate equation; we do not add a second copy of the same constraint.

Put hints on primitive equation parameters. They specify named axes, not coordinate counts, numerical guesses or physical units. `Domain.scalar()` specifies no axes.

Bilt preserves explicit hints as constraints. A hint that conflicts with the equations is an error. One hint may be enough to resolve an ambiguity; every unknown does not need an annotation.

### Coupling can cross a subsystem boundary

A child can describe local physics while its parent describes connections and balances.

If those relationships depend on one another, Bilt can solve their primitive equations in the same block. The subsystem boundary does not require the child to be evaluated independently first.

The [gas turbine example](../worked_examples/two-shaft-gas-turbine.md#14-close-the-loop-with-recuperation) demonstrates this with an exhaust heat recuperator. The exhaust temperature affects preheating, which affects the turbine inlet temperature and therefore the exhaust temperature again.

The model stays composed of reusable child systems. The simultaneous solve crosses their boundaries where the request requires it.

### What if the solve fails?

There are two different stages to consider.

**Before numerical execution**, Bilt may find that an input is missing, a domain layout is ambiguous, a block is nonlinear in its unknowns, or a required operation is unsupported. Planning can try alternative supported formulations. If none works, the request fails with an analysis error and its underlying cause.

**During numerical execution**, the supplied coefficients may make the matrix singular or otherwise invalid. A structurally square block does not guarantee a unique numerical answer.

For example, these equations become dependent when `a=1`:

\[
a x+y=p,\qquad x+y=q
\]

A numerical failure depends on the values supplied for that evaluation. It does not permanently invalidate the model or blacklist the same route for other coefficient values.

Neither a successful solve nor matching units establishes that an operating point is physically sensible. A model can still predict a negative flow or an efficiency above one. Those require engineering checks.

For supported operations, layout restrictions and exception handling, see the [Linear systems reference](../api/linear-systems.md).

[← Systems of systems](systems-of-systems.md) · [Index](../index.md) · [Backends →](backends.md)
