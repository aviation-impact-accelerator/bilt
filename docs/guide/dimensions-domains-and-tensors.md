# Dimensions, domains and tensors

For constructors, reductions, indexing and their current limitations, see the [Tensor API reference](../api/tensors.md). For quantity-related type annotations, see [Quantity annotations and typing](../api/quantity-annotations.md).

So far, most of the values in our examples have been scalars: a speed, a temperature or a pressure.

Real engineering models quickly need something more.

A temperature may vary with time. Demand may vary with location. A quantity may vary over both:

```text
temperature[time]

demand[location, time]
```

Bilt represents these values using **tensors** over named **dimensions**. We will use a simple aircraft energy model to see how a scalar equation can be used across a range of routes and technologies.

## Dimensions give axes names

In NumPy, an array has a shape, such as `(3, 2)`. That tells us how many elements lie along each axis, but not what those axes mean.

Perhaps the shape describes three routes and two technologies. Or it could describe three engines and two operating points. Those are very different things.

In Bilt, we can give dimensions explicit names:

```python exec="on" source="above" session="domains-tensors"
from bilt import dim, domain, tensor

route = dim("route")
technology = dim("technology")
```

A dimension identifies one direction in which a value may vary. A domain is an ordered collection of dimensions:

```python exec="on" source="above" session="domains-tensors"
route_technology = domain(route, technology)
assert route_technology.dims == (route, technology)
```

The distinction becomes important when different values vary over different sets of dimensions.

## Tensors carry values over a domain

A Bilt **tensor** combines concrete values with the dimensions they vary over. Suppose we want to consider three route distances and two aircraft technologies.

```python exec="on" source="above" session="domains-tensors"
distances = tensor([500.0, 1000.0, 2000.0], over=route)
intensities = tensor([0.5, 0.8], over=technology)

assert distances.domain == domain(route)
assert distances.shape == (3,)
assert intensities.domain == domain(technology)
assert intensities.shape == (2,)
```

The route distances are in kilometres, and the energy intensities in kWh per kilometre. These are illustrative numbers, not aircraft performance data. For now we use plain numbers with an agreed unit convention; [Units in models](units.md) explains how to declare physical dimensions and let Bilt check them.

The numbers still live in ordinary arrays. The domain tells Bilt what each axis represents.

## Scalars have a domain too

A scalar is simply a value that does not vary over any dimensions. Its domain is empty: `()`.

We can start with an ordinary relationship:

```text
energy = distance × energy intensity
```

```python exec="on" source="above" session="domains-tensors"
import numpy as np

from bilt import Implementation, Interface, Tensor, equation, field, system


@system
class FlightEnergy(metaclass=Interface):
    distance: float | Tensor = field()
    intensity: float | Tensor = field()
    energy: float | Tensor = field()


@equation
def energy_relation(distance: float, intensity: float, energy: float) -> bool:
    return energy == distance * intensity


energy_model = Implementation(FlightEnergy, [energy_relation])

with energy_model:
    scalar_energy = FlightEnergy(distance=500.0, intensity=0.5).energy

assert scalar_energy == 250.0
```

The system fields accept a plain scalar or a unitless tensor. The equation uses scalar annotations because the relationship is pointwise: it does not need to know which axes the values vary over.

The same equation works with a tensor of route distances and a scalar intensity:

```python exec="on" source="above" session="domains-tensors"
with energy_model:
    route_energy = FlightEnergy(distance=distances, intensity=0.5).energy

assert route_energy.domain == domain(route)
assert route_energy.values.tolist() == [250.0, 500.0, 1000.0]
```

The scalar intensity is applied to each route. The result varies over the route dimension.

## Named dimensions make broadcasting explicit

Now supply both tensors:

```python exec="on" source="above" session="domains-tensors"
with energy_model:
    all_energy = FlightEnergy(distance=distances, intensity=intensities).energy

assert all_energy.domain == domain(route, technology)
assert all_energy.shape == (3, 2)
np.testing.assert_allclose(all_energy.values, [[250.0, 400.0], [500.0, 800.0], [1000.0, 1600.0]])
```

Conceptually:

```text
distance         (route)
intensity               (technology)
                    │
                    ▼
energy           (route, technology)
```

| Distance | Technology 1 (0.5 kWh/km) | Technology 2 (0.8 kWh/km) |
| --- | ---: | ---: |
| 500 km | 250 kWh | 400 kWh |
| 1,000 km | 500 kWh | 800 kWh |
| 2,000 km | 1,000 kWh | 1,600 kWh |

Bilt evaluates every combination of route and technology. We did not need to write nested loops or reshape either input.

## Scalar or tensor values at system boundaries

For plain values, use `float | Tensor` when a system field accepts either a scalar or a unitless tensor. For quantities with physical units, use the corresponding dimensioned `Quantity | QuantityTensor` annotation.

Sometimes a relationship needs a particular axis, for example when reducing over time. Bilt supports minimum domain requirements on dimensioned tensor fields and equation parameters. Exact domain hints can also help disambiguate coupled solves.

See [Quantity annotations and typing](../api/quantity-annotations.md#minimum-domain-requirements) and [Tensor API](../api/tensors.md) for the declaration rules and static typing limitations.

## Broadcasting rules

Bilt broadcasting follows dimension names rather than axis positions.

A few simple rules cover most cases:

```text
scalar                  ()
time value              (time)
location value          (location)
spatial time value      (location, time)
```

A scalar can be combined with any domain:

```text
() + (time)
    → (time)
```

If one value is missing a dimension, it is broadcast across that dimension:

```text
(time) + (location, time)
    → (location, time)
```

Partially overlapping domains are combined:

```text
(location, time) + (time, scenario)
    → (location, time, scenario)
```

Shared dimensions are aligned by name. Their original axis positions do not need to match.

There is one important restriction: if two values share a dimension, they must agree about that dimension during the evaluation. Bilt will not silently treat two incompatible definitions of `time`, for example, as though they were the same axis.

For most model code, this means thinking about **which dimensions a value varies over**, rather than manually arranging array axes to make broadcasting work.

The result axes follow the evaluation's dimension binding order. Inspect the result's `domain` rather than assuming an axis position. See the [Tensor API](../api/tensors.md#construction-and-inspection).

## The relationship can stay simple

The same energy equation has now answered three questions: one flight, three routes for one technology, and every combination of routes and technologies.

The dimensions belong to the values. The equation describes how those values are related.

This is useful when the same engineering model is used for a single design point, a parameter sweep or an uncertainty analysis. Additional named dimensions do not require changes to a pointwise equation.

## Reduce over a named dimension

Sometimes we want a summary rather than every combination. For example, we can calculate the mean energy across the three routes, retaining one value for each technology.

We will use the calculated energy tensor as an input to a small reporting system:

```python exec="on" source="above" session="domains-tensors"
@system
class EnergyReport(metaclass=Interface):
    energy: Tensor = field()
    mean_energy: Tensor = field()
    second_route_energy: Tensor = field()
    selected_route_energy: Tensor = field()


@equation
def mean_over_routes(energy: Tensor, mean_energy: Tensor) -> bool:
    return mean_energy == energy.mean(route)


mean_model = Implementation(EnergyReport, [mean_over_routes])

with mean_model:
    average = EnergyReport(energy=all_energy).mean_energy

assert average.domain == domain(technology)
np.testing.assert_allclose(average.values, [1750.0 / 3.0, 2800.0 / 3.0])
```

A reduction removes the named dimension. Here the input domain is `(route, technology)` and the result domain is `(technology,)`.

This is an unweighted mean over the three example routes, not a fleet-weighted energy estimate. Use the appropriate aggregation for the engineering question.

## Select by position

We can also select the second route (1,000 km) while keeping both technologies:

```python exec="on" source="above" session="domains-tensors"
@equation
def second_route(energy: Tensor, second_route_energy: Tensor) -> bool:
    return second_route_energy == energy.isel(route, 1)


position_model = Implementation(EnergyReport, [second_route])

with position_model:
    selected = EnergyReport(energy=all_energy).second_route_energy

assert selected.domain == domain(technology)
assert selected.values.tolist() == [500.0, 800.0]
```

The index is zero-based. Like a reduction, selection removes one dimension. We use `Tensor` parameters in these equations because they call named tensor methods, rather than just pointwise arithmetic.

These methods are symbolic operations in Bilt equations. Outside a model, use the returned tensor's `values` array for ordinary numerical operations.

## Dimensions also have extents

Knowing that a tensor varies over `time` is not enough to store concrete data.

Bilt also needs to know how large that dimension is.

For example:

```text
time     → 3 values
location → 2 values
```

A tensor over:

```text
(location, time)
```

therefore has shape:

```text
(2, 3)
```

During an evaluation, Bilt keeps track of the concrete dimensions being used and checks that values which share a dimension agree about its extent.

If one value says:

```text
time → 3
```

while another says:

```text
time → 5
```

they cannot silently represent the same `time` dimension in that evaluation.

Catching that mismatch early is preferable to letting incompatible arrays make their way further into the model.

## Coordinates give dimensions concrete meaning

Sometimes an extent is all we need. Often we also care which coordinate each position represents.

For our routes, we can use labels:

```python exec="on" source="above" session="domains-tensors"
from bilt import coordinate_axis

route_axis = coordinate_axis(route, ("short", "medium", "long"))
assert route_axis.coordinates == ("short", "medium", "long")
```

The dimension still identifies the axis, while the coordinate axis tells us what the individual positions along it mean.

We can then select by label rather than by position:

```python exec="on" source="above" session="domains-tensors"
@equation
def select_route(energy: Tensor, selected_route_energy: Tensor) -> bool:
    return selected_route_energy == energy.sel(route, "medium")


coordinate_model = Implementation(EnergyReport, [select_route])
```

Creating the coordinate axis does not bind it to an evaluation. At present, binding requires the internal evaluation context. This is the current mechanism, not a stable public coordinate-binding API:

```python exec="on" source="above" session="domains-tensors"
from bilt import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext

with EvaluationContext() as evaluation:
    evaluation.domains.bind(route_axis)
    medium = coordinate_model.evaluate(
        {"energy": all_energy},
        "selected_route_energy",
        PYTHON_BACKEND,
    )

assert medium.domain == domain(technology)
assert medium.values.tolist() == [500.0, 800.0]
```

The result is the same as positional selection, but the equation refers to the route label rather than its position. Coordinate labels can also be numbers or dates. See [Coordinate axes](../api/tensors.md#coordinate-axes) for the supported types and binding rules.

## Shared dimensions need to agree

Suppose two values both vary over `time`:

```text
temperature : (time)
pressure    : (time)
```

If they participate in the same evaluation, `time` should mean the same thing for both.

It would be dangerous if one secretly represented:

```text
January, February, March
```

while the other represented:

```text
1 s, 2 s, 3 s
```

simply because both arrays happened to contain three elements.

Bilt tracks the extents and any explicitly bound coordinate axes during evaluation. It cannot infer the labels of an unlabelled array: the caller must ensure that the values use the same coordinate order when sharing a dimension.

This is one of the main reasons Bilt uses named domains rather than treating tensors as bare arrays.

## Domains describe variation, not meaning

There is a useful boundary to keep in mind.

A domain tells Bilt **where a value varies**:

```text
(location, time)
```

It does not tell Bilt what the value itself means.

These two tensors might share exactly the same domain:

```text
temperature : (location, time)
pressure    : (location, time)
```

but they remain different model values.

Likewise, the same dimension may be shared by many values:

```text
temperature[time]
pressure[time]
mass_flow[time]
```

The domain describes their structure.

The system and its relationships describe their role in the model.

## Domains become part of planning

In the previous chapter, structural analysis asked whether the known values and relationships admit a supported plan for the requested result.

With tensors, there is another question:

> Are the values on that route compatible over their domains?

A structurally plausible calculation might require:

```text
a : (time)
b : (location)
```

and produce something over:

```text
(location, time)
```

That may be perfectly valid.

Another route might contain values that disagree about the concrete extent or coordinates of a shared dimension.

That route is not valid for the current evaluation.

So the path from model to execution now contains a little more information:

```text
model relationships
        │
        ▼
structural plan
        │
        ▼
domain alignment and validation
        │
        ▼
execution
```

The important point for model authors is simpler:

> Missing dimensions can broadcast. Shared dimensions align by name and must be compatible.

## From tensors to larger systems

We now have most of the basic pieces needed to describe individual Bilt models:

```text
systems          define model boundaries
values           live on those boundaries
transforms       describe directed calculations
equations        describe mathematical relationships
implementations  collect the relationships for a system
domains          describe where values vary
tensors          carry concrete values over those domains
```

The next step is to make models larger without turning them into one enormous system.

Bilt does that by allowing **systems to be composed from other systems**, while preserving the same boundaries and relationships we have been using throughout the guide.

That is the subject of the next chapter. For a larger example of named domains and broadcasting, see the [battery-electric mission model](../worked_examples/battery-electric-mission-model-with-quasi-monte-carlo-broadcasting.md).

[← Models to computation](models-to-computation.md) · [Index](../index.md) · [Systems of systems →](systems-of-systems.md)
