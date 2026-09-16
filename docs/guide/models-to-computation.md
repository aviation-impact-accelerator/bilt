## From models to computation

So far, we have given Bilt two kinds of relationship.

A transform describes a directed calculation:

```text
speed, time ──► distance
```

An equation describes a mathematical relationship that may be usable in several directions:

```text
distance = speed × time
```

Neither tells Bilt which calculations should run for a particular analysis.

Together, they describe computations that *might* be possible.

A transform contributes one fixed direction.

An equation may contribute several possible directions.

When we ask a system for a value, Bilt uses the values already known to work out which of those possibilities can satisfy the request.

We will start with transforms because their direction is already fixed. This makes the structural analysis easier to see.

### Start with what is known

Consider this abstract system:

```python
@system
class Demo(metaclass=Interface):
    alpha: float = field()
    beta: float = field()
    gamma: float = field()
    delta: float = field()
    epsilon: float = field()
```

with these transforms:

```python
@transform
def gamma(alpha: float, beta: float) -> float:
    return alpha + beta


@transform
def delta(alpha: float, gamma: float) -> float:
    return alpha + gamma


@transform
def epsilon(gamma: float, delta: float) -> float:
    return gamma + delta
```

and an implementation:

```python
basic = Implementation(Demo, [gamma, delta, epsilon])
```

Now create an instance:

```python
demo = Demo(alpha=1.0, beta=2.0)
```

At this point, Bilt has two known values:

```text
alpha = 1
beta = 2
```

and three values that have not yet been calculated:

```text
gamma
delta
epsilon
```

Nothing happens just because we created the system.

It is only when we request a value:

```python
with basic:
    result = demo.epsilon
```

that Bilt has a question to answer:

> Given `alpha` and `beta`, can this implementation produce `epsilon`?

### Finding a route

Bilt can look at the available relationships and work backwards from the requested value.

To obtain `epsilon`, it needs:

```text
gamma, delta ──► epsilon
```

`gamma` is not known, but it can be obtained from:

```text
alpha, beta ──► gamma
```

`delta` is also not known, but it can be obtained from:

```text
alpha, gamma ──► delta
```

Since `alpha` and `beta` are already available, Bilt can assemble a complete route:

```text
alpha ──┐
        ├──► gamma ──┐
beta  ──┘            │
                     ├──► epsilon
alpha ───────────────┤
                     │
gamma ─────► delta ──┘
```

That gives a computation plan:

```text
1. gamma from alpha and beta
2. delta from alpha and gamma
3. epsilon from gamma and delta
```

Bilt can then execute that plan and return:

```python
with basic:
    assert demo.epsilon == 7.0
```

This process of working out which relationships can satisfy a request is **structural analysis**.

### Structural analysis is about relationships

At this stage, Bilt does not need to know that:

```python
return alpha + beta
```

happens to perform addition.

For structural analysis, the important information about this transform is:

```text
alpha, beta ──► gamma
```

or, equivalently:

```text
requires: alpha, beta
provides: gamma
```

The numerical calculation matters later, when the plan is executed.

This lets Bilt reason about the structure of a model before carrying out the calculations themselves.

For transforms, the direction is already fixed.

For equations, Bilt may first need to decide which value the relationship can provide from the values available.

Once a particular direction has been chosen, it contributes the same kind of dependency information to the computation plan.

The overall process is:

```text
model relationships
        │
        ▼
request + known values
        │
        ▼
structural analysis
        │
        ▼
computation plan
        │
        ▼
execution
```

First decide **what needs to be done**.

Then do it.

### Bilt only needs a route to the requested value

Suppose we add another value and transform:

```python
@system
class Demo(metaclass=Interface):
    alpha: float = field()
    beta: float = field()
    gamma: float = field()
    delta: float = field()
    epsilon: float = field()
    zeta: float = field()


@transform
def zeta(alpha: float) -> float:
    return alpha * 10
```

and include it in the implementation:

```python
basic = Implementation(Demo, [gamma, delta, epsilon, zeta])
```

If we ask for `epsilon`, there is no reason to calculate `zeta`.

```text
alpha ─────────────────────► zeta

alpha ──┐
        ├──► gamma ──┐
beta  ──┘            │
                     ├──► epsilon
alpha ───────────────┤
                     │
gamma ─────► delta ──┘
```

The route to `zeta` exists, but it is irrelevant to the request.

Bilt can restrict the computation to the part of the model needed to produce `epsilon`.

This is one reason the requested value matters.

An implementation describes everything that *could* be calculated.

A particular evaluation only needs the relationships that lead to what was requested.

### Sometimes there is no route

Now consider a smaller system:

```python
@system
class Demo(metaclass=Interface):
    alpha: float = field()
    beta: float = field()
    gamma: float = field()
    delta: float = field()
```

with only this transform:

```python
@transform
def delta(alpha: float, beta: float) -> float:
    return alpha + beta
```

and an implementation:

```python
basic = Implementation(Demo, [delta])
```

We can construct:

```python
demo = Demo(alpha=1.0, beta=2.0)
```

but if we ask for `gamma`:

```python
with basic:
    _ = demo.gamma
```

Bilt has no relationship that can produce it.

The implementation knows how to calculate `delta`, but that does not help us reach `gamma`.

Conceptually:

```text
alpha, beta ──► delta

gamma ?
```

There is no route.

In that case Bilt raises a `BiltSystemAnalysisError` rather than beginning an evaluation that cannot satisfy the request.

```python
from bilt import BiltSystemAnalysisError

with basic, pytest.raises(BiltSystemAnalysisError):
    _ = demo.gamma
```

This is useful because the failure happens at the level of the model structure.

Bilt can tell that the requested result is unavailable without first stumbling through a partially completed calculation.

### Equations add possible routes

Transforms are the simplest case because they already have a direction.

Equations make structural analysis more interesting.

Recall our motion equation:

```python
@equation
def motion(distance: float, speed: float, time: float):
    return distance == speed * time
```

The equation itself is not directional.

It simply states:

```text
distance = speed × time
```

But that relationship may support several possible calculations:

```text
speed, time ─────► distance

distance, time ──► speed

distance, speed ─► time
```

Suppose we create:

```python
journey = Motion(distance=100.0, time=5.0)
```

and request:

```python
with motion_model:
    speed = journey.speed
```

Bilt can see that `distance` and `time` are already available.

For this request, it can use the equation in the direction:

```text
distance, time ──► speed
```

If instead `speed` and `time` were known and `distance` were requested, the same equation could contribute:

```text
speed, time ──► distance
```

The relationship has not changed.

What changes is how Bilt uses it in the computation plan.

This is an important distinction.

The model contains relationships.

The computation plan contains particular directed uses of those relationships.

### Transforms and equations can work together

A computation plan does not need to use only one kind of relationship.

Suppose an implementation contains a transform:

```text
alpha, beta ──► gamma
```

and an equation:

```text
gamma = delta × epsilon
```

Now suppose `alpha`, `beta`, and `epsilon` are known, and we ask for `delta`.

Bilt can first use the transform:

```text
alpha, beta ──► gamma
```

Once `gamma` is available, it can use the equation in the direction:

```text
gamma, epsilon ──► delta
```

Together, those relationships give a computation plan:

```text
1. gamma from alpha and beta
2. delta from gamma and epsilon
```

or, written more like a conventional program:

```text
gamma = gamma(alpha, beta)
delta = gamma / epsilon
```

One step came from a transform.

The other came from an equation.

From the point of view of structural analysis, they are simply relationships that can contribute steps towards the requested value.

The resulting computation plan can freely combine both.

### A model can contain more than one route

As models grow, there may be several possible ways to obtain the same value.

For example, imagine an implementation that contains:

```text
alpha, beta ─────────► gamma

alpha ──► delta ──┐
                  ├──► gamma
beta ─────────────┘
```

Both routes eventually provide `gamma`.

That means finding *a* valid route and deciding which route to use are separate questions.

The same can happen when equations introduce alternative directions, or when several parts of a composed model can provide the same value.

For now, the important point is simply that an implementation is not necessarily one predetermined chain of calculations.

It can contain alternatives.

Structural analysis turns those possibilities into a concrete plan for a particular request.

### The plan is the program Bilt actually runs

There is a useful way to think about the layers we have built so far.

The model contains relationships:

```text
alpha, beta ──► gamma

alpha, gamma ──► delta

gamma, delta ──► epsilon
```

The request adds a goal:

```text
known: alpha, beta

want:  epsilon
```

Structural analysis turns the two into something much closer to a conventional program:

```text
gamma = gamma(alpha, beta)

delta = delta(alpha, gamma)

epsilon = epsilon(gamma, delta)
```

That plan is directional, ordered, and executable.

An equation can participate in exactly the same kind of plan once Bilt has chosen how to use it.

For example:

```text
speed = motion(distance, time)
```

is not the equation itself.

It is one directed use of the equation selected for this particular analysis.

A plan may also combine transform calls and directed uses of equations, as we saw above.

So although the model does not prescribe a single execution order, the result of analysing it does.

You can think of Bilt as moving through increasingly concrete descriptions:

```text
relationships
     │
     ▼
request + known values
     │
     ▼
computation plan
     │
     ▼
executed values
```

The public model stays concerned with relationships.

Bilt takes responsibility for turning those relationships into a computation.

### Why separate the model from the plan?

For our five-value example, writing the calculation directly would certainly be easier.

```python
gamma = alpha + beta
delta = alpha + gamma
epsilon = gamma + delta
```

Bilt becomes useful when the answer to "what should I calculate?" depends on the problem being asked.

Different evaluations may provide different known values.

Equations may be usable in different directions.

Transforms and equations may contribute to the same analysis.

Large implementations may contain many relationships of which only a few are relevant.

There may be several possible routes to the same result.

And composed models may offer routes through several constituent systems.

Separating the **model** from the **computation plan** lets the same model support those different analyses.

> You describe the relationships once. Bilt works out how to use them for the problem at hand.

So far, all of our values have been simple scalars.

Real engineering models quickly need values that vary over time, location, operating point, or other dimensions.

The next chapter introduces **dimensions, domains and tensors**, which let Bilt carry these same modelling ideas beyond scalar values.

[← Transforms and implementations](transforms-and-implementations.md) · [Index](../index.md) · [Dimensions, domains and tensors →](dimensions-domains-and-tensors.md)
