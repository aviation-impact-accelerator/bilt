# Structured model values are systems

Many scientific models need to return several related values: a fuel mixture, a set of response coefficients, or the remaining resources after an allocation step.

You usually **do not need a dataclass or a multi-output transform** for this. A Bilt system is already a named, typed boundary. Its individual fields can be supplied or computed by equations and transforms, and other systems can connect to those fields.

## Define the structured boundary

Suppose a simple fuel model distinguishes two sustainable fuel pathways and fossil fuel:

```python exec="on" session="structured-boundaries"
from typing import Annotated

from bilt import Implementation, Interface, Quantity, QuantityTensor, equation, field, system
from bilt.units import UNITLESS
from bilt.units.dimensions import DIMENSIONLESS

Share = Annotated[Quantity, DIMENSIONLESS]
SharePort = Annotated[Quantity | QuantityTensor, DIMENSIONLESS]


@system
class FuelMix(metaclass=Interface):
    hefa: SharePort = field()
    btl: SharePort = field()
    saf: SharePort = field()
    fossil: SharePort = field()


@equation
def saf_balance(saf: Share, hefa: Share, btl: Share) -> bool:
    return saf == hefa + btl


@equation
def fuel_balance(fossil: Share, saf: Share) -> bool:
    return fossil + saf == 1.0 * UNITLESS


fuel_mix_model = Implementation(FuelMix, [saf_balance, fuel_balance])
```

The interface gives names and physical dimensions to **four ports**, while the two relationships determine which values can be calculated. No relationship returns a record. An implementation can compute several fields on the *same instance*, one requested field at a time.

```python exec="on" session="structured-boundaries"
with fuel_mix_model:
    mix = FuelMix(hefa=Quantity(0.2, UNITLESS), btl=Quantity(0.3, UNITLESS))
    assert abs(mix.saf.value - 0.5) < 1e-12
    assert abs(mix.fossil.value - 0.5) < 1e-12
```

Use `Annotated[Quantity | QuantityTensor, DIMENSION]` when a port may be a scalar or a named tensor. Bilt checks input dimensions and follows tensor dependencies without requiring a flattened `dict[str, float]`.

## Connect structured values to another system

A parent need not unpack a whole record in an external Python calculation. It can bind a child implementation and connect the named ports with equations:

```python exec="on" session="structured-boundaries"
@system
class FuelPortfolio(metaclass=Interface):
    hefa: SharePort = field()
    btl: SharePort = field()
    fossil: SharePort = field()


@equation
def hefa_connection(hefa: Share, mix: FuelMix) -> bool:
    return hefa == mix.hefa


@equation
def btl_connection(btl: Share, mix: FuelMix) -> bool:
    return btl == mix.btl


@equation
def fossil_connection(fossil: Share, mix: FuelMix) -> bool:
    return fossil == mix.fossil


portfolio_model = Implementation(
    FuelPortfolio,
    [hefa_connection, btl_connection, fossil_connection],
    internal_implementations={"mix": fuel_mix_model},
)

with portfolio_model:
    portfolio = FuelPortfolio(hefa=Quantity(0.2, UNITLESS), btl=Quantity(0.3, UNITLESS))
    assert abs(portfolio.fossil.value - 0.5) < 1e-12
```

This explicitly tells Bilt how the models relate. A child being present does **not** automatically connect matching names. More complex models may reuse the same child implementation several times under distinct instance names.

## What remains outside Bilt

An external application may still use dataclasses or Pydantic models to validate requests, hold metadata, and serialise results. Convert validated *physical inputs* into Bilt ports at the application boundary, and convert requested outputs into the application's output schema afterwards.

Do not turn UI forms, JSON payloads or chart descriptions into systems simply to eliminate Python classes. Do not build a transform merely to access an attribute of an application object.

**When should you use a transform?** When a value has a genuine directed computation; for example, a response coefficient may be provided by a zero-input transform, and a derived response curve by another transform. A multi-stage algorithm with a fixed number of stages can sometimes be modelled as reusable child systems whose outputs are computed independently.

**What this does not provide:** automatic object-to-port binding, arbitrary Python objects inside the analyser, or an operation that returns several named outputs atomically. These remain separate design questions that require motivating examples.

The same pattern is used in the [CYOF fuel and CO₂ migration](https://github.com/aviation-impact-accelerator/aia-product-parliamentary-event-2025/pull/580), with [reusable allocation stages](https://github.com/aviation-impact-accelerator/aia-product-parliamentary-event-2025/pull/582) and [composed CO₂ parameter systems](https://github.com/aviation-impact-accelerator/aia-product-parliamentary-event-2025/pull/583).

[← Systems and values](systems-and-values.md) · [Systems of systems →](systems-of-systems.md)
