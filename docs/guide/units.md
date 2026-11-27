# Units in models

For a catalogue of built-in units, helpers and errors, see the [Units API reference](../api/units.md).
For flexible unit-bearing scalar/tensor fields and equation annotations, see
[Quantity annotations and typing](../api/quantity-annotations.md).

So far, the values in our models have mostly been plain numbers, declared with annotations such as `distance: float = field()`.

But what does a distance of `5.0` mean?

Five metres? Five kilometres? Five miles?

We can agree on a convention and remember it everywhere. That works until a dataset, a colleague, or our future selves uses a different convention.

Bilt lets us put that information into the model. We will start with the motion model from [Equations and acausal modelling](equations-and-acausal-modelling.md), give its values units, and then see how to exchange plain numbers with other software.

## A number, a unit, and a dimension

First, let's make a quantity:

```python exec="on" source="above" session="units-models"
from bilt import Quantity
from bilt.units import kilometer, meter


distance = 5.0 * kilometer
assert distance.value == 5.0
assert distance.unit == kilometer
```

A **quantity** combines a numerical value with a unit. Both construction forms describe the same distance.

To express it in another unit, use `convert_to()`:

```python exec="on" source="above" session="units-models"
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

## Give the system dimensions

Let's declare our motion system again:

```python exec="on" source="above" session="units-models"
from typing import Annotated

from bilt import Implementation, Interface, Quantity, QuantityTensor, equation, field, system
from bilt.units.dimensions import LENGTH, TIME


Length = Annotated[Quantity | QuantityTensor, LENGTH]
Duration = Annotated[Quantity | QuantityTensor, TIME]
Speed = Annotated[Quantity | QuantityTensor, LENGTH / TIME]


@system
class Motion(metaclass=Interface):
    distance: Length = field()
    speed: Speed = field()
    duration: Duration = field()
```

`Annotated[Quantity | QuantityTensor, LENGTH]` says that a value may be a scalar quantity or a tensor quantity, and must have the physical dimension of length.

Bilt reads the annotation when `@system` runs. It does not prescribe metres or kilometres.

Dimensions can be combined using multiplication, division and powers. Speed is length divided by time. Acceleration would be `LENGTH / TIME**2`.

Use a **dimension**, such as `LENGTH`, in an annotation, not a unit such as `meter`.

These are the three useful quantity declarations:

| Annotation | When to use it |
| --- | --- |
| `Annotated[Quantity \\| QuantityTensor, LENGTH]` | Scalar or tensor; pointwise relationships. |
| `Annotated[Quantity, LENGTH]` | Scalar-only field; also usable for pointwise equation parameters. |
| `Annotated[QuantityTensor, LENGTH]` | Tensor-only field or named tensor operations in an equation. |

For specialised domain requirements and typing rules, see [Quantity annotations and typing](../api/quantity-annotations.md).

## Give the equation the same dimensions

The mathematical relationship is unchanged. We can use the same pointwise annotations on the equation and system:

```python exec="on" source="above" session="units-models"
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

## Supply quantities and ask for an answer

Now we can use whichever compatible units are convenient:

```python exec="on" source="above" session="units-models"
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

Bilt converts supplied quantities to **coherent units** for computation (SI units for ordinary physical dimensions). In this example, that means metres, seconds and metres per second. A calculated quantity comes back in the coherent unit of its declared dimension.

For display, choose another unit with `travelled.convert_to(kilometer)`, which represents the same distance as 0.05 km.

Units also work when the equation is used in another direction:

```python exec="on" source="above" session="units-models"
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

## The boundary checks what we supply

A declared field requires a quantity of the right dimension:

```python exec="on" source="above" session="units-models"
Motion(distance=5.0 * kilometer)  # A length: accepted.
```

Supplying `Motion(distance=5.0)` raises `BiltInputDimensionalityError`. Bilt does not assume that an unlabelled number means metres.

Supplying `Motion(distance=5.0 * second)` raises the same error. This time the value has a unit, but it measures time rather than length.

Values declared on the system interface are checked at instantiation, even if we have not yet requested an output. Values accepted through an active implementation are also checked against that implementation's declarations.

This gives callers a clear contract: choose any compatible unit, but say which unit the value uses.

## Choose a unit before extracting a number

Sometimes another library wants a plain number in a particular unit. This is a useful case for a transform: it makes the boundary between the unit-aware model and the numerical interface explicit.

```python exec="off"
from typing import Annotated

from bilt import Quantity, transform
from bilt.units import kilometer
from bilt.units.dimensions import LENGTH

ScalarLength = Annotated[Quantity, LENGTH]


@transform
def distance_km(distance: ScalarLength) -> float:
    return distance.convert_to(kilometer).value


@transform
def distance(distance_km: float) -> ScalarLength:
    return (distance_km * kilometer)
```

<!-- markdown-exec: hidden setup -->

```python exec="on" session="units-models"
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

# Execute the displayed transform definitions from a real Python source file.
guide = Path("docs/guide/units.md").read_text(encoding="utf-8")
source = next(
    block.split("```", 1)[0] for block in guide.split('```python exec="off"\n')[1:] if block.startswith("from typing")
)
with TemporaryDirectory() as directory:
    module = Path(directory) / "unit_bridge.py"
    module.write_text(source, encoding="utf-8")
    definitions = run_path(str(module))
distance_km = definitions["distance_km"]
distance = definitions["distance"]
```

The first transform extracts the number in kilometres. The second interprets a plain number as kilometres. Use the direction appropriate for the surrounding model:

```python exec="on" source="above" session="units-models"
@system
class DistanceBridge(metaclass=Interface):
    distance: Length = field()
    distance_km: float = field()


to_km = Implementation(DistanceBridge, [distance_km])
from_km = Implementation(DistanceBridge, [distance])

with to_km:
    assert DistanceBridge(distance=1000.0 * meter).distance_km == 1.0

with from_km:
    assert DistanceBridge(distance_km=2.0).distance == 2000.0 * meter
```

Bilt translates supported unit operations inside transforms into calculations on coherent numerical values. A bare `distance.value` is refused inside a transform, because it does not state which unit the number represents.

Outside a model, reading a returned quantity's `.value` is ordinary Python. Converting first still makes the numerical convention clear.

Quantity constants can also be used in transforms, for example `distance + 500.0 * meter` or `distance + (0.5 * kilometer)`. See [Units API](../api/units.md#quantities) for the supported operations and restrictions.

## Declarations must agree throughout an implementation

A field and every relationship that names it must agree about its physical dimension, tags and absolute or relative kind.

Bilt does not copy missing declarations from the interface into a relationship. For example, declaring `distance: float` in a transform that uses the unit-aware `Motion.distance` field raises `BiltImplementationDimensionalityError` when the implementation is created.

This also applies to intermediate values and subsystem connections. Bilt checks the symbolic expression of an equation for dimensional consistency. For a transform, it checks declarations and supported unit operations; it does not prove that arbitrary Python is dimensionally correct.

Test the calculation itself, especially where it bridges to plain numbers or external software.

## Plain values and dimensionless quantities

A model can still contain plain numbers, flags and other values, such as `enabled: bool = field()`. These stand outside the unit system.

A dimensionless quantity is a different thing:

```python exec="on" source="above" session="units-models"
from bilt.units import UNITLESS, percent
from bilt.units.dimensions import DIMENSIONLESS


Ratio = Annotated[Quantity | QuantityTensor, DIMENSIONLESS]

half = 50.0 * percent
assert half.convert_to(UNITLESS).value == 0.5
```

A dimensionless system field still requires a quantity, such as `(0.5 * UNITLESS)`; a bare `0.5` is refused.

`DIMENSIONLESS` is the dimension used in declarations. `UNITLESS` is a unit used to construct or convert quantities.

In equations, plain ports cannot participate in the same quantity arithmetic as declared ports. If a gain multiplies a length, declare the gain as `Ratio` too. Numerical literals such as the `2.0` in `distance * 2.0` can still act as ordinary scaling constants.

Transforms can bridge between plain values and quantities explicitly, as we did with `distance_km`.

## Constants in equations carry units too

Units are useful for fixed physical constants as well as inputs:

```python exec="on" source="above" session="units-models"
from bilt.units import kilogram, newton
from bilt.units.dimensions import FORCE, MASS


Force = Annotated[Quantity | QuantityTensor, FORCE]
Mass = Annotated[Quantity | QuantityTensor, MASS]


@equation
def weight(force: Force, mass: Mass) -> bool:
    return force == mass * (9.81 * meter / second**2)


@system
class Weight(metaclass=Interface):
    force: Force = field()
    mass: Mass = field()


with Implementation(Weight, [weight]):
    force = Weight(mass=2.0 * kilogram).force

assert force.convert_to(newton).value == 19.62
```

The constant declares an acceleration. Multiplying it by a mass produces a force.

Writing only `9.81` would lose that information: a bare number scales the mass but cannot turn it into a force.

A constant can also be defined outside the equation and referred to by name. Its unit and dimension remain part of the relationship.

## Temperature readings and temperature changes

Temperature introduces another distinction.

A reading of twenty degrees Celsius is a position on a temperature scale. A rise of twenty degrees Celsius is a difference between two positions.

Bilt represents these as **absolute** and **relative** quantities:

```python exec="on" source="above" session="units-models"
from bilt.units import absolute_celsius, absolute_kelvin, celsius
from bilt.units.dimensions import ABSOLUTE, TEMPERATURE


AbsoluteTemperature = Annotated[Quantity | QuantityTensor, TEMPERATURE, ABSOLUTE]
TemperatureChange = Annotated[Quantity | QuantityTensor, TEMPERATURE]


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
        inlet=(20.0 * absolute_celsius),
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

An equation for a midpoint can use `midpoint == cold + (hot - cold) / 2.0`: the difference is relative, so adding half of it to `cold` produces an absolute reading.

`celsius` denotes a change; `absolute_celsius` denotes a reading. They cannot be substituted for each other at a declared port.

## Quantities can vary over a domain

Units also work with the tensors introduced in [Domains and tensors](dimensions-domains-and-tensors.md). Our original motion system already accepts them, and its equation does not need to change.

```python exec="on" source="above" session="units-models"
from bilt import dim, domain, tensor

sample = dim("sample")

with motion_model:
    travelled = Motion(
        speed=tensor([36.0, 72.0], over=sample, unit=kilometer / hour),
        duration=5.0 * second,
    ).distance

assert travelled.domain == domain(sample)
assert travelled.unit == meter
assert list(travelled.values) == [50.0, 100.0]
```

The domain says where the value varies. The unit says what its numbers represent.

These are independent: `dim("sample")` names an array axis, while `LENGTH` describes a physical dimension.

The same pointwise annotations work for both scalar and tensor quantities. Use a tensor-only annotation when the field requires a tensor or the equation uses named methods such as `sum` or `isel`. See [Quantity annotations and typing](../api/quantity-annotations.md#named-tensor-operations-in-equations) for more detail.

## Tags preserve what a quantity measures

Sometimes physical dimensions alone are insufficient. A kilogram of fuel and a kilogram of carbon dioxide are both masses, but confusing them would change the meaning of a model.

Bilt's metadata tags can make that distinction part of the declaration:

```python exec="on" source="above" session="units-models"
from bilt.units import Metadata, kilogram


fuel = Metadata("fuel")
co2 = Metadata("co2")

FuelMass = Annotated[Quantity | QuantityTensor, MASS * fuel]
Co2Mass = Annotated[Quantity | QuantityTensor, MASS * co2]


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

See [Dimensions and metadata](../api/units.md#dimensions-and-metadata) for tag conversions and supported operations.

## When something disagrees

The errors identify the stage at which Bilt finds a disagreement:

| Error | What to check |
| --- | --- |
| `BiltPortDeclarationError` | Use exactly one physical dimension on a quantity declaration. For new code, use `Quantity`, `QuantityTensor`, or their union, and annotate with a dimension rather than a unit. |
| `BiltEquationDimensionalityError` | Check the equation's operations, both sides of the relation, and absolute versus relative quantities. |
| `BiltImplementationDimensionalityError` | Give every occurrence of a shared port the same declaration. |
| `BiltInputDimensionalityError` | Supply a quantity or tensor with the required dimension, tags and absolute or relative unit. |
| `BiltDefinitionError` | Use supported transform operations and name a unit before extracting a numerical value. |
| `BiltUnresolvedAnnotationError` | Define or import annotation names before the decorator runs, outside `if TYPE_CHECKING:`. |

Bilt reads annotations at runtime, so their imports must remain available at runtime too. If Ruff asks to move them into a type-checking block, see [Configuration](../config.md#ruff).

We can now write the same engineering relationships as before, while stating what their values measure. Callers choose convenient units. The declarations keep those choices compatible with the model.

Continue with the [worked examples](../index.md#worked-examples), which combine units with equations, transforms, named tensors and subsystems.

---

[← Backends](backends.md) · [Index](../index.md)
