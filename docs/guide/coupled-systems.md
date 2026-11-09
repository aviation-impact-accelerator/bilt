## Coupled systems

Some equations must be solved together because their unknowns depend on one another. Bilt identifies these **coupled blocks** and currently supports solving blocks that are **linear in their unknowns**. Other calculations in the model may still be nonlinear.

Coupled equations can span systems and subsystems. You do not need to flatten the model or rearrange the equations into matrix form.

### Start with two unknowns

Consider:

\[
x+y=p,\qquad x-y=q
\]

If `p` and `q` are known, neither equation alone determines `x` or `y`. Together, they determine both.

```python exec="on" source="above" session="linear-systems"
import numpy as np

from bilt import Implementation, Interface, equation, field, system


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


pair_implementation = Implementation(Pair, [total, difference])
pair_system = Pair(p=10.0, q=2.0)

with pair_implementation:
    x = pair_system.x
    y = pair_system.y

assert np.isclose(x, 6.0)
assert np.isclose(y, 4.0)
```

`Implementation` associates equations with the `Pair` interface class. An instance supplies known values; reading an output inside the implementation context requests its calculation.

Bilt’s default causalization (SCC) structural analyser identifies the coupled block. Bilt then extracts its coefficients and solves the resulting linear system. The Python or JAX backend determines how the calculations execute.

### Different inputs can give a different plan

Supply `x` and `q` instead, and ask for `p`:

```python exec="on" source="above" session="linear-systems"
inverse_system = Pair(x=6.0, q=2.0)

with pair_implementation:
    inverse_p = inverse_system.p

assert np.isclose(inverse_p, 10.0)
```

Bilt can now calculate `y=x-q`, then `p=x+y`. No coupled solve is needed.

The equations are unchanged. The supplied values determine what remains unknown and how the model can be evaluated.

### Linear in which unknowns?

A block must be linear **in the values being solved together**.

For example:

\[
a x+y=p
\]

is linear in `x` and `y` when `a` is known. The coefficient `a` may be supplied directly or calculated by a nonlinear relationship earlier in the plan.

If both `a` and `x` are unknowns in the same block, their product is nonlinear. Bilt currently supports only linear coupled blocks; if no supported alternative formulation is available, analysis fails.

The restriction therefore applies to the coupled block, not to every relationship in the model.

### Independent solves over a domain

A tensor holds values over named dimensions. Suppose `p` varies over operating points while the known value `q` is shared:

```python exec="on" source="above" session="linear-systems"
from bilt import dim, domain, tensor

operating_point = dim("operating_point")

pointwise_system = Pair(
    p=tensor([10.0, 14.0, 18.0], over=operating_point),
    q=2.0,
)

with pair_implementation:
    pointwise = pointwise_system.x

assert pointwise.domain == domain(operating_point)
np.testing.assert_allclose(pointwise.values, [6.0, 8.0, 10.0])
```

Each operating point has its own unknown `x` and `y`. Bilt broadcasts the known `q` across the points and can solve the independent blocks as a batch.

Sharing a **known value** does not connect the unknowns at different points.

### A shared unknown connects the points

Now consider:

\[
x_t+y=p_t,\qquad \tfrac12\sum_t x_t-y=q
\]

Here, `x` varies over time, but there is one unknown `y` for the whole set. The reduction `x.sum(time)` connects all the time points through a single equation.

```python exec="on" source="above" session="linear-systems"
from bilt import Tensor

time = dim("time")


@equation
def shared_balance(x: Tensor, y: float, p: Tensor) -> bool:
    return x + y == p


@equation
def aggregate(x: Tensor, y: float, q: float) -> bool:
    return 0.5 * x.sum(time) - y == q


shared_implementation = Implementation(Pair, [shared_balance, aggregate])
shared_system = Pair(
    p=tensor([2.0, 4.0, 6.0], over=time),
    q=1.0,
)

with shared_implementation:
    varying = shared_system.x
    shared = shared_system.y

assert varying.domain == domain(time)
np.testing.assert_allclose(varying.values, [0.0, 2.0, 4.0])
assert np.isclose(shared, 2.0)
```

There are four unknown values: three elements of `x` and one shared `y`. The three point balances and the aggregate equation form one four-by-four linear system.

Bilt infers the unknowns’ domains from the known inputs and relationships. When inference is ambiguous, add domain hints to equation parameters: for example, `Annotated[Tensor, domain(time)]` or `Annotated[float, Domain.scalar()]`. A plain `float` annotation does not require a scalar domain. Dimension sizes must already be established by known inputs.

See [Domains and tensors](dimensions-domains-and-tensors.md) for inference and hint details.

### Units and subsystem boundaries

Each output retains its own physical units and inferred or explicitly hinted domain. Solving values together does not require them to have matching units or domains. See [Units in models](units.md) for physical declarations and conversions.

Coupling can also cross subsystem boundaries. In the [two springs in series](../worked_examples/two-springs-in-series.md#9-a-request-that-requires-a-coupled-solve), each child supplies its spring law, while the parent connects their forces and adds their extensions.

Given the total extension and both stiffnesses, Bilt solves the common force and both extensions together. The model remains composed of reusable child systems. The same principle applies through nested subsystems; calculations outside the coupled block remain separate steps.

### What if the solve fails?

Failures can occur at two stages:

- **During analysis:** inputs may be missing, domain layouts ambiguous, or a required block nonlinear or unsupported. Bilt can try alternative supported formulations; the request fails if none works.
- **During numerical execution:** the supplied coefficients may produce a singular or otherwise invalid matrix. A structurally square block does not guarantee a unique answer.

For example, these equations become dependent when `a=1`:

\[
a x+y=p,\qquad x+y=q
\]

A numerical failure for one set of coefficients does not permanently invalidate the model.

A successful solve also does not establish physical validity. For example, negative flows or efficiencies above one still require engineering checks.

[← Systems of systems](systems-of-systems.md) · [Index](../index.md) · [Backends →](backends.md)
