# Units API

Reference for `bilt.units`, `bilt.units.dimensions` and unit-related exceptions in
`bilt.errors`. For model declarations and a guided example, see [Units in models](../guide/units.md).

## Imports and declarations

```python
from typing import Annotated

from bilt import Quantity, Tensor, tensor
from bilt.units import Metadata, Prefix, Unit, kilometer, meter
from bilt.units.dimensions import ABSOLUTE, LENGTH, TEMPERATURE

Length = Annotated[Quantity, LENGTH]
LengthSeries = Annotated[Tensor, LENGTH]
AbsoluteTemperature = Annotated[Quantity, TEMPERATURE, ABSOLUTE]
```

Declare dimensions, not units, on model ports. A bare `Quantity` is refused;
a bare `Tensor` declares no physical dimension. A declared quantity port can accept a compatible
unit-carrying tensor; a declared tensor port requires a tensor.
Plain ports, such as `float`, stand outside the unit system.

## Built-in units

All names below are exported from `bilt.units`. Names use the spelling shown
(`meter`, `liter`, and `siemen`). Prefix variants are available only where listed;
use `Unit.new_named()` to define others.

| Quantity | Available names |
| --- | --- |
| Dimensionless and angle | `UNITLESS`, `fraction`, `percent`, `radian`, `degree` |
| Mass | `milligram`, `gram`, `kilogram`, `megagram`, `gigagram`, `teragram`, `tonne`, `kilotonne`, `megatonne` |
| Length | `millimeter`, `centimeter`, `meter`, `kilometer`, `inch`, `foot`, `yard`, `mile` |
| Time | `millisecond`, `second`, `minute`, `hour`, `day`, `year` |
| Temperature change | `kelvin`, `celsius`, `fahrenheit` |
| Absolute temperature | `absolute_kelvin`, `absolute_celsius`, `absolute_fahrenheit` |
| Absolute time | `absolute_year` |
| Electric current | `ampere` |
| Amount of substance | `mole`, `kilomole` |
| Luminous intensity | `candela` |
| Area | `acre`, `hectare` |
| Volume | `liter` |
| Force | `millinewton`, `newton`, `kilonewton`, `meganewton` |
| Pressure | `pascal`, `kilopascal`, `megapascal`, `bar`, `atmosphere`, `torr` |
| Power | `watt`, `kilowatt`, `megawatt`, `gigawatt`, `terawatt`, `petawatt`, `exawatt` |
| Energy | `joule`, `kilojoule`, `megajoule`, `gigajoule`, `terajoule`, `petajoule`, `exajoule`, `watthour`, `kilowatthour`, `megawatthour`, `gigawatthour`, `british_thermal_unit`, `therm`, `million_british_thermal_unit` |
| Voltage | `volt`, `millivolt` |
| Electric charge | `coulomb` |
| Electric resistance | `ohm` |
| Electric conductance | `siemen` |
| Dynamic viscosity | `poiseuille`, `poise`, `centipoise` |
| Annual currencies | `USD[year]`, `GBP[year]`, `EUR[year]`, `JPY[year]`, for integer years **2000–2025** |

Compose other units with `*`, `/`, integer powers, and reciprocals:
`meter / second`, `kilogram / meter**3`, `meter**2`, `1 / second`.

Conventions:

- `percent` represents `0.01 * UNITLESS`; `radian` is dimensionless and a degree is π/180 radians.
- `year` is 365.25 days. `absolute_year` exists as a unit, but model evaluation
  currently has no coherent absolute-second counterpart and rejects absolute time inputs.
- Use `celsius` for a temperature difference and `absolute_celsius` for a reading.
  Absolute and relative units cannot be interconverted.
- Currency conversions use fixed, bundled annual exchange and inflation data,
  relative to USD in 2025. They do not fetch live rates. Index a currency before
  using it as a unit; an unsupported year raises `AttributeError`.

```python
from bilt.units import GBP, USD, UNITLESS, percent

assert (50.0 * percent).convert_to(UNITLESS).value == 0.5
cost_in_gbp = (100.0 * USD[2024]).convert_to(GBP[2024])
```

## Quantities

`Quantity` is available from both `bilt` and `bilt.units`. Methods return a new
quantity without changing the original.

| API | Meaning |
| --- | --- |
| `Quantity(value, unit)` | Construct a scalar quantity with a numerical value and a `Unit`. |
| `5.0 * kilometer` | Shorthand for `Quantity(5.0, kilometer)`; use floating-point literals. |
| `q.value` | Numerical value in `q.unit`. |
| `q.unit` | The quantity's unit; `q.unit.dimension` gives its dimension. |
| `q.convert_to(unit)` | Rescale the value (and apply absolute offsets) to a compatible unit. Dimensions, including tags, must agree. |
| `q.cast_to(unit)` | Keep the number unchanged and relabel with an equivalent unit, or a unit equivalent after removing metadata. Does not convert scales. |
| `q.strip_metadata()` | Remove all unit metadata tags, preserving the number and physical scale. |
| `q * r`, `q / r` | Multiply or divide quantities, combining their values and units. |
| `q * 2.0`, `q / 2.0` | Scale a quantity. |
| `q**2` | Integer power. Floating-point powers are supported for dimensionless quantities. |
| `+q`, `-q`, `abs(q)`, `round(q, ndigits)` | Apply the numerical operation and preserve the unit. |

```python
distance = 5.0 * kilometer
assert distance.convert_to(meter).value == 5000.0
```

Outside models, convert operands to the same unit before addition, subtraction,
ordering or equality checks. These operations do not automatically convert
representations; distinct but equivalent unit objects may require `cast_to()`
for addition. Equality compares the numerical values and unit equivalence.

Absolute + relative and absolute − relative produce absolute quantities;
absolute − absolute produces a relative quantity. For direct Python arithmetic,
align the absolute units and the relative step scales first. Absolute + absolute
and relative − absolute are refused.

During model evaluation, Bilt converts inputs to coherent units and returns
calculated outputs in coherent units. For ordinary physical dimensions these
are SI units; currency uses USD2025 and metadata tags are retained.

Inside transforms, extract numbers with `q.convert_to(unit).value` and construct
quantities with `Quantity(value, unit)` or a float multiplied by a unit. Bare
quantity `.value`, representation-dependent `.unit` access, and `cast_to()` are
refused. Transform checks do not prove dimensional correctness of arbitrary Python.

## Defining and inspecting units

| API | Meaning |
| --- | --- |
| `Unit.new_named(name, relation, prefix=None)` | Define a named unit from a `Unit`, a scaled `Quantity`, or an internal unit composition. Its dimension follows from the relation. |
| `Unit.new_absolute(name, scale, zero=None, metadata=None)` | Define an absolute unit with a relative step `scale`. `zero`, when supplied, is an absolute quantity locating its origin. An absolute unit based on a non-SI-base scale needs a zero point for conversion. |
| `abs(unit)` | Look up the existing absolute counterpart, e.g. `abs(celsius)` is `absolute_celsius`. Does not create one. |
| `unit.to_si_base_units()` | Express the unit's multiplicative relation in base units, retaining its scale factor and tags. For absolute offsets, convert a quantity instead. |
| `unit.strip_metadata()` | Remove all tags from a unit. |
| `unit.name`, `str(unit)` | Name or printable representation. |
| `unit.dimension`, `unit.metadata`, `unit.prefix` | Physical dimension (including tags), directly attached metadata, and prefix, if any. |
| `unit.relation`, `unit.offset`, `unit.category` | Definition, optional absolute offset, and internal category (`BASE`, `NAMED`, `DERIVED`, `ABSOLUTE`). |
| `unit.is_dimensionless`, `unit.is_relative` | Whether the unit is dimensionless or contains only relative units. |
| `unit.is_si_base_unit`, `unit.is_si_derived_unit`, `unit.is_coherent` | Classification against Bilt's base unit system. |

Define shared custom units in a project module and import them where needed:

```python
from bilt.units import Unit, hour, meter

nautical_mile = Unit.new_named("nautical_mile", relation=1852.0 * meter)
knot = Unit.new_named("knot", relation=nautical_mile / hour)

assert (2.0 * nautical_mile).convert_to(meter).value == 3704.0
```

Names are cached process-wide. Repeating the same definition returns the cached
unit; a different relation under an existing name raises `BiltUnitError`.
Only one absolute counterpart may be registered for a given relative scale.
Use dimensions, such as `LENGTH / TIME`, in port declarations for these units too.

### Prefixes

`Prefix` is an integer enum whose values are powers of ten:

| Members | Exponents |
| --- | --- |
| `ATTO`, `FEMTO`, `PICO`, `NANO`, `MICRO` | −18, −15, −12, −9, −6 |
| `MILLI`, `CENTI`, `DECI` | −3, −2, −1 |
| `DECA`, `HECTO`, `KILO` | 1, 2, 3 |
| `MEGA`, `GIGA`, `TERA`, `PETA`, `EXA` | 6, 9, 12, 15, 18 |

```python
from bilt.units import Prefix, Unit, second

microsecond = Unit.new_named("microsecond", relation=second, prefix=Prefix.MICRO)
```

The name must start with the prefix's lowercase name. Use an unprefixed reference
unit as above; do not also multiply the relation by the prefix factor.

## Dimensions and metadata

Import dimension constants and `ABSOLUTE` from `bilt.units.dimensions`.

| Group | Available names |
| --- | --- |
| Primary | `MASS`, `LENGTH`, `TIME`, `TEMPERATURE`, `CURRENT`, `AMOUNT`, `LUMINOSITY`, `CURRENCY` |
| Derived | `AREA`, `VOLUME`, `FORCE`, `ENERGY`, `POWER`, `PRESSURE`, `DYNAMIC_VISCOSITY`, `CHARGE`, `VOLTAGE`, `RESISTANCE`, `CONDUCTANCE` |
| Dimensionless | `DIMENSIONLESS` |
| Absolute declaration marker | `ABSOLUTE` (a marker, not a dimension or unit) |

| API | Meaning |
| --- | --- |
| `dimension * other`, `dimension / other`, `dimension**n` | Compose dimensions using integer exponents. |
| `Dimension(mass=0, length=0, time=0, temperature=0, current=0, amount=0, luminosity=0, currency=0)` | Construct a dimension explicitly from integer exponents. Import `Dimension` from `bilt.units`. |
| `dimension.mass`, `.length`, `.time`, `.temperature`, `.current`, `.amount`, `.luminosity`, `.currency` | Individual exponents; `.dimensions` returns their tuple in this order. |
| `dimension.tagged` | Mapping of tagged dimension factors to exponents. |
| `dimension.is_dimensionless`, `.is_primary`, `.is_derived` | Dimension classification properties. |
| `Metadata(value)` | Create or retrieve a tag by a non-empty string. |
| `tag.value` | The tag's string value. |
| `dimension * tag`, `unit * tag`, `quantity * tag` | Attach a tag to a dimension, unit or quantity. |

```python
from bilt.units import Metadata, kilogram
from bilt.units.dimensions import MASS

fuel = Metadata("fuel")
FuelMass = Annotated[Quantity, MASS * fuel]
fuel_mass = 2.0 * kilogram * fuel
assert fuel_mass.strip_metadata().unit == kilogram
```

Tags are part of dimensional compatibility: tagged and untagged masses cannot
be converted directly. Use `strip_metadata()` deliberately, or `cast_to()` when
the units are equivalent after removing tags. Put tags into the dimension in
annotations; `Annotated[Quantity, MASS, fuel]` is refused.
Equations express tag changes through tagged unit factors; transforms support
`strip_metadata()` and multiplication by a tag.

## Unit-carrying tensors

Import `tensor` and `Tensor` from `bilt`.

| API | Meaning |
| --- | --- |
| `tensor(values, *, over=None, domain=None, unit=None)` | Label array values with a unit. Non-scalar values require `over` or `domain`; provide only one. |
| `Tensor.scalar(value, unit=None)` | Construct a rank-zero tensor, optionally with a unit. |
| `t.unit`, `t.values`, `t.domain` | Unit (or `None`), numerical array, and named domain. |
| `t.convert_to(unit)` | Convert all values, retaining the domain. Requires an existing compatible unit. |
| `t * unit` | Label an unlabelled tensor, or compose with its existing unit, without rescaling its values. |
| `t / unit` | Divide an existing tensor unit by another unit, without rescaling values. |

Unit multiplication labels values; `convert_to()` rescales them. Concrete tensor
arithmetic is otherwise dispatched through Bilt models. Tensors do not expose
the quantity helpers `cast_to()` or `strip_metadata()`.

## Errors and messages

Import exceptions from `bilt.errors`. Catch classes rather than matching full
message strings; messages include the offending ports, quantities and units.

| Exception | When raised / action |
| --- | --- |
| `BiltUnitsError` | Base for unit and dimensionality failures. |
| `BiltUnitError` | Incompatible representations, invalid named/absolute definitions, unsupported conversion or missing tensor unit. Convert operands, fix the definition, or label the tensor. |
| `BiltDimensionalityError` | Incompatible physical dimensions or metadata tags; base for the three model dimensionality errors below. |
| `BiltMetadataError` | Invalid metadata, such as an empty tag. |
| `BiltEquationDimensionalityError` | An equation is dimensionally inconsistent. Check operations, constants, tags and absolute/relative quantities. |
| `BiltImplementationDimensionalityError` | Shared ports have conflicting declarations or the implementation makes an equation inconsistent. Declare the same dimension everywhere. |
| `BiltInputDimensionalityError` | An input has the wrong dimension, tags, absolute/relative kind or scalar/tensor carrier, or is unlabelled. |
| `BiltPortDeclarationError` | Invalid annotation: use exactly one dimension on a quantity/tensor, not a unit. This is a `BiltDefinitionError`, not a `BiltUnitsError`. |
| `BiltDefinitionError` | Unsupported unit operations inside a transform, including bare quantity `.value` or `cast_to()`. |
| `BiltUnresolvedAnnotationError` | An annotation name is missing when the decorator runs. Import or define it at runtime, outside `if TYPE_CHECKING:`. |

Common message excerpts (`…` stands for context-dependent details):

| Message excerpt | Meaning / remedy |
| --- | --- |
| `Cannot convert unit with dimensionality … as they are different` | Dimensions or tags differ; a scale conversion cannot fix this. |
| `use the … Quantity.convert_to() … to convert them to consistent units` | Convert operands to the same representation before direct arithmetic. |
| `use the … Quantity.cast_to() … to cast them to consistent units` | Equivalent unit objects differ; explicitly align their labels. |
| `Cannot cast quantity … as the two units are not equivalent` | Use `convert_to()` to change scale; check tags and dimensions. |
| `relative and absolute units cannot be interconverted` | Use the correct unit kind for a reading or a difference. |
| `Cannot add two quantities with absolute units` | Subtract readings to obtain a difference, or add a relative change. |
| `No absolute unit corresponding to … exists` | `abs(unit)` cannot find a registered counterpart. |
| `has no offset (zero point) set` | Anchor a custom absolute scale to an existing absolute unit with `zero=`. |
| `already exists with the same scale` | An absolute counterpart is already registered for this relative scale. Reuse it or define a separate relative scale. |
| `Cannot instantiate unit … with the same name` | A name is already registered with a different relation. |
| `does not start with its defined prefix` | Match the unit name to `Prefix`, e.g. `microsecond` with `Prefix.MICRO`. |
| `Metadata must be a non-empty value` | Supply a non-empty tag string. |
| `Cannot convert a tensor with no unit` | Supply `unit=` or label the tensor before conversion. |
| `Input … is a plain port, outside the unit system` | Supply a plain value or declare a quantity dimension on the port. |

`BiltPortDeclarationError`, `BiltDefinitionError` and
`BiltUnresolvedAnnotationError` are definition failures; catching `BiltUnitsError`
alone does not catch them. See the [units guide](../guide/units.md) for model examples.
