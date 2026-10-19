## Units in models

So far, the values in our models have mostly been plain numbers:

```python
distance: float = field()
```

But what does a distance of `5.0` mean?

Five metres? Five kilometres? Five miles?

We can agree on a convention and remember it everywhere. That works until a dataset, a colleague, or our future selves uses a different convention.

Bilt lets us put that information into the model. We will start with the motion model from [Equations and acausal modelling](equations-and-acausal-modelling.md), give its values units, and then add a transform that works alongside its equation.

### A number, a unit, and a dimension

First, let's make a quantity:

```python
from bilt import Quantity
from bilt.units import kilometer, meter


distance = 5.0 * kilometer
```

A **quantity** combines a numerical value with a unit. We can also construct one explicitly:

```python
distance = Quantity(5.0, kilometer)
```

Both expressions describe the same distance.

To express it in another unit, use `convert_to()`:

```python
in_metres = distance.convert_to(meter)

assert in_metres.value == 5000.0
assert in_metres.unit == meter
```

The distance has not changed. Its numerical representation has.

A **dimension** describes what kind of quantity this is. Metres and kilometres are different units of the same dimension: length.

These three pieces have different jobs:

| Piece | Example | What it tells us |
| --- | --- | --- |
| Value | `5.0` | How much, in the chosen unit |
| Unit | `kilometer` | How the quantity is represented |
| Dimension | `LENGTH` | What kind of quantity it is |

This distinction matters when we declare a model. We want a distance to accept any unit of length, rather than requiring every caller to supply kilometres.

### Give the system dimensions

Let's declare our motion system again:

```python
from typing import Annotated

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.units.dimensions import LENGTH, TIME


Length = Annotated[Quantity, LENGTH]
Duration = Annotated[Quantity, TIME]
Speed = Annotated[Quantity, LENGTH / TIME]


@system
class Motion(metaclass=Interface):
    distance: Length = field()
    speed: Speed = field()
    duration: Duration = field()
```

There is one new piece of Python here:

```python
Annotated[Quantity, LENGTH]
```

`Annotated` lets us attach extra information to a type. Here, the type is `Quantity` and the extra information is its dimension.

Bilt reads that annotation when `@system` runs. The `distance` field now says:

> This value is a quantity of length.

It does not prescribe metres or kilometres.

We have given the annotations short names so that we can reuse them in our equations and transforms. The aliases are a convenience; writing `Annotated[Quantity, LENGTH]` directly has the same meaning.

Dimensions can be combined using multiplication, division and powers. Speed is length divided by time. Acceleration would be `LENGTH / TIME**2`.

Use a dimension in the annotation, rather than a unit:

```python
Length = Annotated[Quantity, LENGTH]  # A valid declaration.
```

`Annotated[Quantity, meter]` raises `BiltPortDeclarationError`. So does a bare `Quantity` annotation: it says that a value carries units, but leaves its dimension unspecified.

### Give the equation the same dimensions

The mathematical relationship is unchanged:

```python
@equation
def kinematic_relation(
    distance: Length,
    speed: Speed,
    duration: Duration,
) -> bool:
    return distance == speed * duration


motion_model = Implementation(Motion, [kinematic_relation])
```

We have changed the annotations, but the equation still says that distance equals speed multiplied by duration.

Bilt checks the dimensions when `@equation` runs. On the right, speed multiplied by duration has dimension:

\[
\frac{\text{length}}{\text{time}} \times \text{time} = \text{length}.
\]

That agrees with the left side.

If we accidentally wrote `distance == speed + duration`, the equation would raise `BiltEquationDimensionalityError` when decorated. Adding speed to time cannot produce a meaningful quantity.

This check happens when the model is defined. We do not need to wait for a particular set of inputs to discover the mistake.

### Supply quantities and ask for an answer

Now we can use whichever compatible units are convenient:

```python
from bilt.units import hour, second


with motion_model:
    journey = Motion(
        speed=36.0 * kilometer / hour,
        duration=5.0 * second,
    )
    travelled = journey.distance

assert travelled == 50.0 * meter
```

Thirty-six kilometres per hour is ten metres per second. After five seconds, we have travelled fifty metres.

Bilt converts supplied quantities to **coherent SI units** for computation. In this example, that means metres, seconds and metres per second. A calculated quantity comes back in the coherent unit of its declared dimension.

For display, we can choose another unit:

```python
assert travelled.convert_to(kilometer).value == 0.05
```

Units also work when the equation is used in another direction:

```python
with motion_model:
    journey = Motion(
        distance=1.0 * kilometer,
        duration=100.0 * second,
    )
    required_speed = journey.speed

assert required_speed == 10.0 * meter / second
assert required_speed.convert_to(kilometer / hour).value == 36.0
```

We still have one equation. Bilt rearranges it as needed, and the requested value keeps its own declared dimension.

### The boundary checks what we supply

A declared field requires a quantity of the right dimension:

```python
Motion(distance=5.0 * kilometer)  # A length: accepted.
```

Supplying `Motion(distance=5.0)` raises `BiltInputDimensionalityError`. Bilt does not assume that an unlabelled number means metres.

Supplying `Motion(distance=5.0 * second)` raises the same error. This time the value has a unit, but it measures time rather than length.

Values declared on the system interface are checked at instantiation, even if we have not yet requested an output. Values accepted through an active implementation are also checked against that implementation's declarations.

Calling `Implementation.evaluate()` directly applies the same input checks, including declarations on intermediate ports that are not exposed by the system interface.

This gives callers a clear contract: choose any compatible unit, but say which unit the value uses.

### A transform can use quantity constants

Suppose we want to include a small allowance in our journey distance. This calculation has a natural direction, so we will use a transform.

Extend the system with one field:

```python
@system
class Motion(metaclass=Interface):
    distance: Length = field()
    speed: Speed = field()
    duration: Duration = field()
    allowance: Length = field()


@transform
def allowance(distance: Length) -> Length:
    return distance + 500.0 * meter


motion_model = Implementation(Motion, [kinematic_relation, allowance])
```

As with any transform, the function name identifies its output. Its return annotation declares the output's dimension.

Now ask for the allowance:

```python
with motion_model:
    result = Motion(distance=1.0 * kilometer).allowance

assert result == 1500.0 * meter
```

The constant means five hundred metres regardless of the unit used by the caller.

Quantity constructors work too:

```python
@transform
def allowance(distance: Length) -> Length:
    return distance + Quantity(0.5, kilometer)
```

This describes the same calculation.

Bilt translates the unit operations in a transform into numerical operations for the backend. The backend computes with coherent values, while the model keeps its declarations at the boundary.

That is also why the output comes back in metres. Returning a quantity expressed in kilometres does not select a permanent output unit for the system.

### Choose a unit before extracting a number

Sometimes another library wants a plain number in a particular unit. We can make that conversion explicit:

```python
@transform
def distance_km(distance: Length) -> float:
    return distance.convert_to(kilometer).value
```

For a distance of one thousand metres, this produces `1.0`.

Add `distance_km: float = field()` to the interface and include the transform in its implementation to expose that result through a system.

The opposite bridge is just as explicit:

```python
@transform
def distance(distance_km: float) -> Length:
    return Quantity(distance_km, kilometer)
```

This interprets a plain number as kilometres and produces a declared length. These two transforms illustrate opposite directions; use the one needed by the surrounding model.

Inside a transform, a bare `distance.value` is refused with `BiltDefinitionError`. Its meaning would depend on the unit in which the quantity arrived. Write `distance.convert_to(kilometer).value` to state the numerical convention you need.

Likewise, avoid branching on `distance.unit` or relabelling a quantity with `cast_to()` inside a transform. Model behaviour should depend on the quantity, rather than the caller's choice of representation.

Outside a transform, reading a returned quantity's `.value` is ordinary Python. For a numerical interface that expects a specific unit, converting first still makes the convention clear.

These restrictions apply to quantity operations. Ordinary Python objects may also have attributes named `.value` or methods named `convert_to()`. Bilt preserves these operations where it recognises the object as plain, including unit objects passed as arguments. Recognition of locally constructed objects is currently limited.

### Declarations must agree throughout an implementation

A field and every relationship that names it must agree about its dimension.

For example, this transform leaves `distance` plain:

```python
@transform
def allowance(distance: float) -> float:
    return distance + 500.0
```

It cannot be used with the unit-aware `Motion` interface above. Creating the implementation raises `BiltImplementationDimensionalityError`: the interface declares lengths, while the transform leaves those same values outside the unit system.

Bilt does not copy missing declarations from the interface into a relationship. Declare the dimension everywhere the value is named.

The same rule applies to intermediate values shared by relationships, even when those values are absent from the public interface. It also applies to subsystem connections: a child field retains its declaration when a parent equation references it.

Equations and transforms provide different kinds of checking. Bilt analyses an equation's symbolic expression for dimensional consistency. For a transform, it checks declarations and supported unit operations, including contradictory conversions on unchanged parameters. It does not prove that every operation in an arbitrary Python body is dimensionally correct. Test the calculation itself, especially where it bridges to plain numbers or external software.

### Plain values and dimensionless quantities

A model can still contain plain numbers, flags and other values:

```python
enabled: bool = field()
```

These values stand outside the unit system. Adding declared quantities elsewhere does not automatically give them dimensions.

A dimensionless quantity is a different thing:

```python
from bilt.units import UNITLESS, percent
from bilt.units.dimensions import DIMENSIONLESS


Ratio = Annotated[Quantity, DIMENSIONLESS]

half = 50.0 * percent
assert half.convert_to(UNITLESS).value == 0.5
```

A field annotated `Ratio` requires a dimensionless quantity, such as `Quantity(0.5, UNITLESS)`. A bare `0.5` is still refused.

`DIMENSIONLESS` is the dimension used in declarations. `UNITLESS` is a unit used to construct or convert quantities.

In equations, plain ports cannot participate in the same quantity arithmetic as declared ports. If a gain multiplies a length, declare the gain as `Ratio` too. Numerical literals such as the `2.0` in `distance * 2.0` can still act as ordinary scaling constants.

Transforms can bridge between plain values and quantities explicitly, as we did with `distance_km`.

### Constants in equations carry units too

Units are useful for fixed physical constants as well as inputs:

```python
from bilt.units.dimensions import FORCE, MASS


Force = Annotated[Quantity, FORCE]
Mass = Annotated[Quantity, MASS]


@equation
def weight(force: Force, mass: Mass) -> bool:
    return force == mass * (9.81 * meter / second**2)
```

The constant declares an acceleration. Multiplying it by a mass produces a force.

Writing only `9.81` would lose that information: a bare number scales the mass but cannot turn it into a force.

A constant can also be defined outside the equation and referred to by name. Its unit and dimension remain part of the relationship.

### Temperature readings and temperature changes

Temperature introduces another distinction.

A reading of twenty degrees Celsius is a position on a temperature scale. A rise of twenty degrees Celsius is a difference between two positions.

Bilt represents these as **absolute** and **relative** quantities:

```python
from bilt.units import absolute_celsius, absolute_kelvin, celsius, kelvin
from bilt.units.dimensions import ABSOLUTE, TEMPERATURE


AbsoluteTemperature = Annotated[Quantity, TEMPERATURE, ABSOLUTE]
TemperatureChange = Annotated[Quantity, TEMPERATURE]


@system
class Heater(metaclass=Interface):
    inlet: AbsoluteTemperature = field()
    rise: TemperatureChange = field()
    outlet: AbsoluteTemperature = field()


@equation
def heating(
    outlet: AbsoluteTemperature,
    inlet: AbsoluteTemperature,
    rise: TemperatureChange,
) -> bool:
    return outlet == inlet + rise


with Implementation(Heater, [heating]):
    outlet = Heater(
        inlet=Quantity(20.0, absolute_celsius),
        rise=10.0 * celsius,
    ).outlet

assert outlet.unit == absolute_kelvin
assert abs(outlet.convert_to(absolute_celsius).value - 30.0) < 1e-10
```

The inlet is an absolute reading. The rise is a relative change. Adding them produces another absolute reading.

The result uses absolute kelvin internally, including the Celsius offset. Converting it back gives thirty degrees Celsius.

For additive arithmetic, keep these rules in mind:

| Operation | Result |
| --- | --- |
| Absolute reading + relative change | Absolute reading |
| Absolute reading − relative change | Absolute reading |
| Absolute reading − absolute reading | Relative change |
| Absolute reading + absolute reading | Refused |
| Relative change − absolute reading | Refused |

An equation for the midpoint between two readings can use:

```python
return midpoint == cold + (hot - cold) / 2.0
```

The difference is relative; adding half of it to `cold` produces an absolute reading.

In a transform that receives a plain Celsius number, construct the reading explicitly with `Quantity(degrees, absolute_celsius)`. To extract Celsius degrees from an absolute quantity, use `reading.convert_to(absolute_celsius).value`.

`celsius` denotes a change; `absolute_celsius` denotes a reading. They cannot be substituted for each other at a declared port.

### Quantities can vary over a domain

Units also work with the tensors introduced in [Domains and tensors](dimensions-domains-and-tensors.md):

```python
from bilt import Tensor, dim, tensor


LengthSeries = Annotated[Tensor, LENGTH]
sample = dim("sample")


@system
class Survey(metaclass=Interface):
    distance: LengthSeries = field()
    doubled: LengthSeries = field()


@equation
def doubling(doubled: LengthSeries, distance: LengthSeries) -> bool:
    return doubled == distance * 2.0


with Implementation(Survey, [doubling]):
    doubled = Survey(
        distance=tensor([1.0, 2.0], over=sample, unit=kilometer),
    ).doubled

assert doubled.unit == meter
assert list(doubled.values) == [2000.0, 4000.0]
assert list(doubled.convert_to(kilometer).values) == [2.0, 4.0]
```

The domain says where the value varies. The unit says what its numbers represent.

These are independent: `dim("sample")` names an array axis, while `LENGTH` describes a physical dimension.

Use `Annotated[Tensor, LENGTH]` when the interface requires a tensor. It rejects a scalar quantity. A `Quantity` declaration can also accept a unit-carrying tensor at runtime, allowing a scalar relationship to be reused over a domain. The same port may be declared over `Tensor` on the interface and over `Quantity` in a relationship, provided its dimension and absolute marker agree.

A tensor with no domain axes—a rank-zero tensor—is still a tensor. For example, `tensor(1.0, unit=kilometer)` is accepted by a Tensor-declared length port. Calculated outputs from Tensor-declared ports remain tensors even when their domain is scalar. They can therefore be reused as inputs without changing their declared type.

### Tags preserve what a quantity measures

Sometimes physical dimensions alone are insufficient. A kilogram of fuel and a kilogram of carbon dioxide are both masses, but confusing them would change the meaning of a model.

Bilt's metadata tags can make that distinction part of the declaration:

```python
from bilt.units import Metadata, kilogram


fuel = Metadata("fuel")
co2 = Metadata("co2")

FuelMass = Annotated[Quantity, MASS * fuel]
Co2Mass = Annotated[Quantity, MASS * co2]


@system
class Burn(metaclass=Interface):
    burned: FuelMass = field()
    emitted: Co2Mass = field()


@equation
def emissions(emitted: Co2Mass, burned: FuelMass) -> bool:
    return emitted == burned * (3.16 * (kilogram * co2) / (kilogram * fuel))


with Implementation(Burn, [emissions]):
    emitted = Burn(burned=2.0 * kilogram * fuel).emitted

assert emitted.unit == kilogram * co2
assert abs(emitted.value - 6.32) < 1e-10
```

The emission factor carries both tags. Multiplication cancels the fuel tag and leaves a carbon dioxide mass.

An untagged mass, or one carrying the wrong tag, is refused at `burned`. Tags also need to agree across an implementation.

Put a tag into the dimension with `MASS * fuel`; writing `Annotated[Quantity, MASS, fuel]` is refused.

Transforms can deliberately remove tags with `quantity.strip_metadata()` and attach them with `quantity * tag`. Equations express these changes through tagged unit factors, as above; bare tag multiplication and `strip_metadata()` are not supported in equations.

### When something disagrees

The errors identify the stage at which Bilt finds a disagreement:

| Error | What to check |
| --- | --- |
| `BiltPortDeclarationError` | Use exactly one dimension on `Quantity` or `Tensor`; use dimensions rather than units in annotations. |
| `BiltEquationDimensionalityError` | Check the equation's operations, both sides of the relation, and absolute versus relative quantities. |
| `BiltImplementationDimensionalityError` | Give every occurrence of a shared port the same declaration. |
| `BiltInputDimensionalityError` | Supply a quantity or tensor with the required dimension, tags and absolute or relative unit. |
| `BiltDefinitionError` | Use supported transform operations and name a unit before extracting a numerical value. |
| `BiltUnresolvedAnnotationError` | Define or import annotation names before the decorator runs, outside `if TYPE_CHECKING:`. |

Bilt reads annotations at runtime, so their imports must remain available at runtime too. If Ruff asks to move them into a type-checking block, see [Configuration](../config.md#ruff).

We can now write the same engineering relationships as before, while stating what their values measure. Callers choose convenient units. The declarations keep those choices compatible with the model.

---

[← Backends](backends.md) · [Index](../index.md)
