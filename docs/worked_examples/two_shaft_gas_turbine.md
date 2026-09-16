# A two-shaft gas turbine with useful shaft power: a more complex acausal system of systems

This worked example takes the same compositional idea as the two-spring example and applies it to a more recognisable engineering system: a small two-shaft gas turbine.

You do not need to know gas-turbine design to follow the example.

At a high level, the engine contains two kinds of component:

- a **compressor** squeezes the incoming air and therefore requires mechanical work;
- a **turbine** extracts energy from the hot gas and turns it into mechanical work.

The engine uses two shafts to transfer that work.

```mermaid
flowchart LR
    inlet["Cool inlet air"] --> lpc["LP compressor<br/>uses shaft work"]
    lpc --> hpc["HP compressor<br/>uses shaft work"]
    hpc --> heat["Heat added<br/>represented here by<br/>turbine inlet temperature"]
    heat --> hpt["HP turbine<br/>extracts shaft work"]
    hpt --> lpt["LP turbine<br/>extracts shaft work"]
    lpt --> exhaust["Hot exhaust"]

    hpt -. "drives" .-> hpc
    lpt -. "drives" .-> lpc
    lpt -. "remaining work" .-> output["Useful output shaft<br/>generator / propeller / load"]
```

The **HP turbine** drives the HP compressor.

The **LP turbine** must do two jobs:

```text
LP turbine work
      =
LP compressor work
      +
useful output work
```

That useful output is the quantity we ultimately care about.

If the engine drives a generator, propeller, pump, or another mechanical load, this is the work available to that load.

## Why there are two kinds of work quantity

The component equations use **specific work**, measured here in `kJ/kg`.

That means:

> how much mechanical work is transferred for each kilogram of gas flowing through the engine?

To turn that into ordinary shaft power we also need the mass flow rate:

```text
shaft power
    =
specific output work × mass flow
```

Because:

```text
kJ/kg × kg/s = kJ/s = kW
```

So a specific output work of `100 kJ/kg` at a mass flow of `20 kg/s` corresponds to:

```text
2000 kW = 2 MW
```

## A simplifying exhaust assumption

This model only describes component temperatures and work; it does not yet include a nozzle or turbine pressure model.

To determine how much work the LP turbine extracts, we therefore specify the **core exit temperature** as an operating boundary condition.

For the baseline example we use:

```text
core exit temperature = 1000 K
```

A real engine model would usually determine the turbine expansion from pressures and component efficiencies. Here the fixed exit temperature keeps the physics intentionally low-order while giving the shaft-power calculation a well-defined endpoint.

## The Bilt modelling constraint

There is one important Bilt limitation in this example:

> Bilt does not yet solve equations that require a simultaneous coupled solve across sibling subsystems.

So we do not write one equation containing both a compressor and a turbine.

Instead:

```text
child component
      ↕
ordinary parent value
      ↕
parent equation
      ↕
ordinary parent value
      ↕
child component
```

This keeps the model acausal while ensuring every requested result can be reached through computation paths Bilt can solve today.

The gas path still runs physically from compressor to turbine, but those physical arrows do **not** prescribe the direction in which Bilt must use the equations.

```python exec="on" session="turbomachinery"
from io import StringIO
import importlib.util
from pathlib import Path
import tempfile

import matplotlib

# markdown-exec runs code in a worker thread.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def render_plot() -> None:
    buffer = StringIO()
    plt.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close()
    print(buffer.getvalue())
```

## 1. Define reusable component systems

We start with a compressor.

```python
from bilt import Implementation, Interface, equation, field, system


@system
class Compressor(metaclass=Interface):
    inlet_temperature_k: float = field()
    outlet_temperature_k: float = field()
    pressure_ratio: float = field()
    isentropic_efficiency: float = field()
    gamma: float = field()
    specific_heat_kj_per_kg_k: float = field()
    specific_work_kj_per_kg: float = field()


@equation
def compressor_temperature_relation(
    inlet_temperature_k: float,
    outlet_temperature_k: float,
    pressure_ratio: float,
    isentropic_efficiency: float,
    gamma: float,
) -> bool:
    return (
        (outlet_temperature_k - inlet_temperature_k)
        * isentropic_efficiency
        == inlet_temperature_k
        * (
            pressure_ratio ** ((gamma - 1.0) / gamma)
            - 1.0
        )
    )


@equation
def compressor_work_relation(
    specific_work_kj_per_kg: float,
    specific_heat_kj_per_kg_k: float,
    inlet_temperature_k: float,
    outlet_temperature_k: float,
) -> bool:
    return (
        specific_work_kj_per_kg
        == specific_heat_kj_per_kg_k
        * (outlet_temperature_k - inlet_temperature_k)
    )


compressor_model = Implementation(
    Compressor,
    [
        compressor_temperature_relation,
        compressor_work_relation,
    ],
)
```

These are equations, not transforms.

For example:

```text
specific work = cp × (outlet temperature - inlet temperature)
```

does not permanently designate specific work as the output. With a different set of known values, Bilt can use the same relationship in another direction.

The turbine is similarly small:

```python
@system
class Turbine(metaclass=Interface):
    inlet_temperature_k: float = field()
    outlet_temperature_k: float = field()
    specific_heat_kj_per_kg_k: float = field()
    specific_work_kj_per_kg: float = field()


@equation
def turbine_work_relation(
    specific_work_kj_per_kg: float,
    specific_heat_kj_per_kg_k: float,
    inlet_temperature_k: float,
    outlet_temperature_k: float,
) -> bool:
    return (
        specific_work_kj_per_kg
        == specific_heat_kj_per_kg_k
        * (inlet_temperature_k - outlet_temperature_k)
    )


turbine_model = Implementation(
    Turbine,
    [turbine_work_relation],
)
```

## 2. Define the two-shaft core

The parent system contains the values needed to connect the four child systems.

```python
@system
class TwoShaftCore(metaclass=Interface):
    # Gas-path temperatures.
    inlet_temperature_k: float = field()
    low_pressure_compressor_exit_temperature_k: float = field()
    high_pressure_compressor_exit_temperature_k: float = field()
    turbine_inlet_temperature_k: float = field()
    high_pressure_turbine_exit_temperature_k: float = field()
    exit_temperature_k: float = field()

    # Compressor design assumptions.
    low_pressure_compressor_pressure_ratio: float = field()
    high_pressure_compressor_pressure_ratio: float = field()
    low_pressure_compressor_efficiency: float = field()
    high_pressure_compressor_efficiency: float = field()

    # Shared fluid properties.
    gamma: float = field()
    specific_heat_kj_per_kg_k: float = field()

    # Shaft efficiencies.
    low_pressure_shaft_efficiency: float = field()
    high_pressure_shaft_efficiency: float = field()

    # Coupling values used by the two shaft balances.
    low_pressure_compressor_work_kj_per_kg: float = field()
    high_pressure_compressor_work_kj_per_kg: float = field()
    high_pressure_turbine_work_kj_per_kg: float = field()
    low_pressure_turbine_work_kj_per_kg: float = field()

    # Useful LP-shaft output.
    shaft_output_work_kj_per_kg: float = field()
    mass_flow_kg_per_s: float = field()
    shaft_power_kw: float = field()
```

Some of these values exist mainly to provide an explicit hand-off between child systems.

For example:

```text
LP compressor.outlet
        ↕
low_pressure_compressor_exit_temperature_k
        ↕
HP compressor.inlet
```

The two equations on either side each touch only one child.

## 3. Connect the LP compressor

The parent inlet is connected to the LP compressor inlet:

```python
@equation
def low_pressure_compressor_inlet_connection(
    inlet_temperature_k: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return (
        inlet_temperature_k
        == low_pressure_compressor.inlet_temperature_k
    )
```

The compressor exit is exposed on the parent:

```python
@equation
def low_pressure_compressor_exit_connection(
    low_pressure_compressor_exit_temperature_k: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return (
        low_pressure_compressor_exit_temperature_k
        == low_pressure_compressor.outlet_temperature_k
    )
```

and the parent supplies its pressure ratio, efficiency, and fluid properties:

```python
@equation
def low_pressure_compressor_pressure_ratio_connection(
    low_pressure_compressor_pressure_ratio: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return (
        low_pressure_compressor_pressure_ratio
        == low_pressure_compressor.pressure_ratio
    )


@equation
def low_pressure_compressor_efficiency_connection(
    low_pressure_compressor_efficiency: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return (
        low_pressure_compressor_efficiency
        == low_pressure_compressor.isentropic_efficiency
    )


@equation
def low_pressure_compressor_gamma_connection(
    gamma: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return gamma == low_pressure_compressor.gamma


@equation
def low_pressure_compressor_specific_heat_connection(
    specific_heat_kj_per_kg_k: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return (
        specific_heat_kj_per_kg_k
        == low_pressure_compressor.specific_heat_kj_per_kg_k
    )
```

Finally, its specific work is copied onto the parent:

```python
@equation
def low_pressure_compressor_work_connection(
    low_pressure_compressor_work_kj_per_kg: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return (
        low_pressure_compressor_work_kj_per_kg
        == low_pressure_compressor.specific_work_kj_per_kg
    )
```

## 4. Connect the HP compressor

The LP compressor exit temperature now acts as the HP compressor inlet temperature:

```python
@equation
def high_pressure_compressor_inlet_connection(
    low_pressure_compressor_exit_temperature_k: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        low_pressure_compressor_exit_temperature_k
        == high_pressure_compressor.inlet_temperature_k
    )
```

The HP compressor is otherwise connected in the same way:

```python
@equation
def high_pressure_compressor_exit_connection(
    high_pressure_compressor_exit_temperature_k: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        high_pressure_compressor_exit_temperature_k
        == high_pressure_compressor.outlet_temperature_k
    )


@equation
def high_pressure_compressor_pressure_ratio_connection(
    high_pressure_compressor_pressure_ratio: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        high_pressure_compressor_pressure_ratio
        == high_pressure_compressor.pressure_ratio
    )


@equation
def high_pressure_compressor_efficiency_connection(
    high_pressure_compressor_efficiency: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        high_pressure_compressor_efficiency
        == high_pressure_compressor.isentropic_efficiency
    )


@equation
def high_pressure_compressor_gamma_connection(
    gamma: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return gamma == high_pressure_compressor.gamma


@equation
def high_pressure_compressor_specific_heat_connection(
    specific_heat_kj_per_kg_k: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        specific_heat_kj_per_kg_k
        == high_pressure_compressor.specific_heat_kj_per_kg_k
    )


@equation
def high_pressure_compressor_work_connection(
    high_pressure_compressor_work_kj_per_kg: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        high_pressure_compressor_work_kj_per_kg
        == high_pressure_compressor.specific_work_kj_per_kg
    )
```

Again, no equation directly mentions both compressors.

The shared parent temperature is the hand-off point.

## 5. Connect the turbines

The turbine inlet temperature is connected to the HP turbine:

```python
@equation
def high_pressure_turbine_inlet_connection(
    turbine_inlet_temperature_k: float,
    high_pressure_turbine: Turbine,
) -> bool:
    return (
        turbine_inlet_temperature_k
        == high_pressure_turbine.inlet_temperature_k
    )
```

The HP turbine exit is exposed on the parent:

```python
@equation
def high_pressure_turbine_exit_connection(
    high_pressure_turbine_exit_temperature_k: float,
    high_pressure_turbine: Turbine,
) -> bool:
    return (
        high_pressure_turbine_exit_temperature_k
        == high_pressure_turbine.outlet_temperature_k
    )
```

and the same parent value independently feeds the LP turbine:

```python
@equation
def low_pressure_turbine_inlet_connection(
    high_pressure_turbine_exit_temperature_k: float,
    low_pressure_turbine: Turbine,
) -> bool:
    return (
        high_pressure_turbine_exit_temperature_k
        == low_pressure_turbine.inlet_temperature_k
    )
```

The remaining turbine connections are:

```python
@equation
def core_exit_connection(
    exit_temperature_k: float,
    low_pressure_turbine: Turbine,
) -> bool:
    return (
        exit_temperature_k
        == low_pressure_turbine.outlet_temperature_k
    )


@equation
def high_pressure_turbine_specific_heat_connection(
    specific_heat_kj_per_kg_k: float,
    high_pressure_turbine: Turbine,
) -> bool:
    return (
        specific_heat_kj_per_kg_k
        == high_pressure_turbine.specific_heat_kj_per_kg_k
    )


@equation
def low_pressure_turbine_specific_heat_connection(
    specific_heat_kj_per_kg_k: float,
    low_pressure_turbine: Turbine,
) -> bool:
    return (
        specific_heat_kj_per_kg_k
        == low_pressure_turbine.specific_heat_kj_per_kg_k
    )


@equation
def high_pressure_turbine_work_connection(
    high_pressure_turbine_work_kj_per_kg: float,
    high_pressure_turbine: Turbine,
) -> bool:
    return (
        high_pressure_turbine_work_kj_per_kg
        == high_pressure_turbine.specific_work_kj_per_kg
    )


@equation
def low_pressure_turbine_work_connection(
    low_pressure_turbine_work_kj_per_kg: float,
    low_pressure_turbine: Turbine,
) -> bool:
    return (
        low_pressure_turbine_work_kj_per_kg
        == low_pressure_turbine.specific_work_kj_per_kg
    )
```

## 6. Balance the shafts and expose useful output work

Once component work has been exposed as ordinary parent values, the shaft balances themselves contain no child-system parameters.

### The HP shaft

The HP turbine exists to drive the HP compressor.

After allowing for shaft efficiency:

```text
HP compressor work
    =
HP shaft efficiency × HP turbine work
```

In Bilt:

```python
@equation
def high_pressure_shaft_balance(
    high_pressure_compressor_work_kj_per_kg: float,
    high_pressure_turbine_work_kj_per_kg: float,
    high_pressure_shaft_efficiency: float,
) -> bool:
    return (
        high_pressure_compressor_work_kj_per_kg
        == high_pressure_shaft_efficiency
        * high_pressure_turbine_work_kj_per_kg
    )
```

### The LP shaft

The LP turbine has an additional job: after driving the LP compressor, any remaining work can leave the engine through the output shaft.

```mermaid
flowchart LR
    turbine["LP turbine work"] --> efficiency["LP shaft efficiency"]
    efficiency --> available["Work available on LP shaft"]
    available --> compressor["LP compressor requirement"]
    available --> output["Useful output work"]
```

The balance is:

```text
usable LP turbine work
    =
LP compressor work
    +
useful output work
```

or:

```python
@equation
def low_pressure_shaft_balance(
    low_pressure_compressor_work_kj_per_kg: float,
    low_pressure_turbine_work_kj_per_kg: float,
    low_pressure_shaft_efficiency: float,
    shaft_output_work_kj_per_kg: float,
) -> bool:
    return (
        low_pressure_shaft_efficiency
        * low_pressure_turbine_work_kj_per_kg
        == low_pressure_compressor_work_kj_per_kg
        + shaft_output_work_kj_per_kg
    )
```

Finally, mass flow converts specific output work into shaft power:

```python
@equation
def shaft_power_relation(
    shaft_output_work_kj_per_kg: float,
    mass_flow_kg_per_s: float,
    shaft_power_kw: float,
) -> bool:
    return (
        shaft_power_kw
        == shaft_output_work_kj_per_kg * mass_flow_kg_per_s
    )
```

These are all acausal equations.

For example, the last equation can be used either as:

```text
specific work + mass flow → shaft power
```

or:

```text
required shaft power + mass flow → required specific work
```

That second direction will be useful in the inverse-design example.

## 7. Put the implementation together

The four component occurrences are internal implementations:

```python
core_model = Implementation(
    TwoShaftCore,
    [
        low_pressure_compressor_inlet_connection,
        low_pressure_compressor_exit_connection,
        low_pressure_compressor_pressure_ratio_connection,
        low_pressure_compressor_efficiency_connection,
        low_pressure_compressor_gamma_connection,
        low_pressure_compressor_specific_heat_connection,
        low_pressure_compressor_work_connection,
        high_pressure_compressor_inlet_connection,
        high_pressure_compressor_exit_connection,
        high_pressure_compressor_pressure_ratio_connection,
        high_pressure_compressor_efficiency_connection,
        high_pressure_compressor_gamma_connection,
        high_pressure_compressor_specific_heat_connection,
        high_pressure_compressor_work_connection,
        high_pressure_turbine_inlet_connection,
        high_pressure_turbine_exit_connection,
        high_pressure_turbine_specific_heat_connection,
        high_pressure_turbine_work_connection,
        low_pressure_turbine_inlet_connection,
        core_exit_connection,
        low_pressure_turbine_specific_heat_connection,
        low_pressure_turbine_work_connection,
        high_pressure_shaft_balance,
        low_pressure_shaft_balance,
        shaft_power_relation,
    ],
    internal_implementations={
        "low_pressure_compressor": compressor_model,
        "high_pressure_compressor": compressor_model,
        "high_pressure_turbine": turbine_model,
        "low_pressure_turbine": turbine_model,
    },
)
```

The hierarchy organises the model. It does not prescribe an execution order.

A child system can also have more than one valid internal route to the same port. Which route Bilt uses depends on the values available in the surrounding model, so nesting a subsystem does not turn it into a fixed one-way calculation.

The important restriction is simply that the path can be decomposed into:

```text
parent value
    ↕
one child system
    ↕
parent value
    ↕
parent equation
    ↕
parent value
    ↕
one child system
```

For the executable documentation below, the same model is loaded from a temporary Python module. This gives Bilt a real source file for `inspect.getsource()` during the `markdown-exec` build.

```python exec="on" session="turbomachinery"
model_source = """
from bilt import Implementation, Interface, equation, field, system


@system
class Compressor(metaclass=Interface):
    inlet_temperature_k: float = field()
    outlet_temperature_k: float = field()
    pressure_ratio: float = field()
    isentropic_efficiency: float = field()
    gamma: float = field()
    specific_heat_kj_per_kg_k: float = field()
    specific_work_kj_per_kg: float = field()


@equation
def compressor_temperature_relation(
    inlet_temperature_k: float,
    outlet_temperature_k: float,
    pressure_ratio: float,
    isentropic_efficiency: float,
    gamma: float,
) -> bool:
    return (
        (outlet_temperature_k - inlet_temperature_k)
        * isentropic_efficiency
        == inlet_temperature_k
        * (
            pressure_ratio ** ((gamma - 1.0) / gamma)
            - 1.0
        )
    )


@equation
def compressor_work_relation(
    specific_work_kj_per_kg: float,
    specific_heat_kj_per_kg_k: float,
    inlet_temperature_k: float,
    outlet_temperature_k: float,
) -> bool:
    return (
        specific_work_kj_per_kg
        == specific_heat_kj_per_kg_k
        * (outlet_temperature_k - inlet_temperature_k)
    )


compressor_model = Implementation(
    Compressor,
    [
        compressor_temperature_relation,
        compressor_work_relation,
    ],
)


@system
class Turbine(metaclass=Interface):
    inlet_temperature_k: float = field()
    outlet_temperature_k: float = field()
    specific_heat_kj_per_kg_k: float = field()
    specific_work_kj_per_kg: float = field()


@equation
def turbine_work_relation(
    specific_work_kj_per_kg: float,
    specific_heat_kj_per_kg_k: float,
    inlet_temperature_k: float,
    outlet_temperature_k: float,
) -> bool:
    return (
        specific_work_kj_per_kg
        == specific_heat_kj_per_kg_k
        * (inlet_temperature_k - outlet_temperature_k)
    )


turbine_model = Implementation(
    Turbine,
    [turbine_work_relation],
)


@system
class TwoShaftCore(metaclass=Interface):
    inlet_temperature_k: float = field()
    low_pressure_compressor_exit_temperature_k: float = field()
    high_pressure_compressor_exit_temperature_k: float = field()
    turbine_inlet_temperature_k: float = field()
    high_pressure_turbine_exit_temperature_k: float = field()
    exit_temperature_k: float = field()

    low_pressure_compressor_pressure_ratio: float = field()
    high_pressure_compressor_pressure_ratio: float = field()
    low_pressure_compressor_efficiency: float = field()
    high_pressure_compressor_efficiency: float = field()

    gamma: float = field()
    specific_heat_kj_per_kg_k: float = field()

    low_pressure_shaft_efficiency: float = field()
    high_pressure_shaft_efficiency: float = field()

    low_pressure_compressor_work_kj_per_kg: float = field()
    high_pressure_compressor_work_kj_per_kg: float = field()
    high_pressure_turbine_work_kj_per_kg: float = field()
    low_pressure_turbine_work_kj_per_kg: float = field()

    # Useful LP-shaft output.
    shaft_output_work_kj_per_kg: float = field()
    mass_flow_kg_per_s: float = field()
    shaft_power_kw: float = field()


@equation
def low_pressure_compressor_inlet_connection(
    inlet_temperature_k: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return inlet_temperature_k == low_pressure_compressor.inlet_temperature_k


@equation
def low_pressure_compressor_exit_connection(
    low_pressure_compressor_exit_temperature_k: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return (
        low_pressure_compressor_exit_temperature_k
        == low_pressure_compressor.outlet_temperature_k
    )


@equation
def low_pressure_compressor_pressure_ratio_connection(
    low_pressure_compressor_pressure_ratio: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return low_pressure_compressor_pressure_ratio == low_pressure_compressor.pressure_ratio


@equation
def low_pressure_compressor_efficiency_connection(
    low_pressure_compressor_efficiency: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return (
        low_pressure_compressor_efficiency
        == low_pressure_compressor.isentropic_efficiency
    )


@equation
def low_pressure_compressor_gamma_connection(
    gamma: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return gamma == low_pressure_compressor.gamma


@equation
def low_pressure_compressor_specific_heat_connection(
    specific_heat_kj_per_kg_k: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return (
        specific_heat_kj_per_kg_k
        == low_pressure_compressor.specific_heat_kj_per_kg_k
    )


@equation
def low_pressure_compressor_work_connection(
    low_pressure_compressor_work_kj_per_kg: float,
    low_pressure_compressor: Compressor,
) -> bool:
    return (
        low_pressure_compressor_work_kj_per_kg
        == low_pressure_compressor.specific_work_kj_per_kg
    )


@equation
def high_pressure_compressor_inlet_connection(
    low_pressure_compressor_exit_temperature_k: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        low_pressure_compressor_exit_temperature_k
        == high_pressure_compressor.inlet_temperature_k
    )


@equation
def high_pressure_compressor_exit_connection(
    high_pressure_compressor_exit_temperature_k: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        high_pressure_compressor_exit_temperature_k
        == high_pressure_compressor.outlet_temperature_k
    )


@equation
def high_pressure_compressor_pressure_ratio_connection(
    high_pressure_compressor_pressure_ratio: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        high_pressure_compressor_pressure_ratio
        == high_pressure_compressor.pressure_ratio
    )


@equation
def high_pressure_compressor_efficiency_connection(
    high_pressure_compressor_efficiency: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        high_pressure_compressor_efficiency
        == high_pressure_compressor.isentropic_efficiency
    )


@equation
def high_pressure_compressor_gamma_connection(
    gamma: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return gamma == high_pressure_compressor.gamma


@equation
def high_pressure_compressor_specific_heat_connection(
    specific_heat_kj_per_kg_k: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        specific_heat_kj_per_kg_k
        == high_pressure_compressor.specific_heat_kj_per_kg_k
    )


@equation
def high_pressure_compressor_work_connection(
    high_pressure_compressor_work_kj_per_kg: float,
    high_pressure_compressor: Compressor,
) -> bool:
    return (
        high_pressure_compressor_work_kj_per_kg
        == high_pressure_compressor.specific_work_kj_per_kg
    )


@equation
def high_pressure_turbine_inlet_connection(
    turbine_inlet_temperature_k: float,
    high_pressure_turbine: Turbine,
) -> bool:
    return turbine_inlet_temperature_k == high_pressure_turbine.inlet_temperature_k


@equation
def high_pressure_turbine_exit_connection(
    high_pressure_turbine_exit_temperature_k: float,
    high_pressure_turbine: Turbine,
) -> bool:
    return (
        high_pressure_turbine_exit_temperature_k
        == high_pressure_turbine.outlet_temperature_k
    )


@equation
def high_pressure_turbine_specific_heat_connection(
    specific_heat_kj_per_kg_k: float,
    high_pressure_turbine: Turbine,
) -> bool:
    return (
        specific_heat_kj_per_kg_k
        == high_pressure_turbine.specific_heat_kj_per_kg_k
    )


@equation
def high_pressure_turbine_work_connection(
    high_pressure_turbine_work_kj_per_kg: float,
    high_pressure_turbine: Turbine,
) -> bool:
    return (
        high_pressure_turbine_work_kj_per_kg
        == high_pressure_turbine.specific_work_kj_per_kg
    )


@equation
def low_pressure_turbine_inlet_connection(
    high_pressure_turbine_exit_temperature_k: float,
    low_pressure_turbine: Turbine,
) -> bool:
    return (
        high_pressure_turbine_exit_temperature_k
        == low_pressure_turbine.inlet_temperature_k
    )


@equation
def core_exit_connection(
    exit_temperature_k: float,
    low_pressure_turbine: Turbine,
) -> bool:
    return exit_temperature_k == low_pressure_turbine.outlet_temperature_k


@equation
def low_pressure_turbine_specific_heat_connection(
    specific_heat_kj_per_kg_k: float,
    low_pressure_turbine: Turbine,
) -> bool:
    return (
        specific_heat_kj_per_kg_k
        == low_pressure_turbine.specific_heat_kj_per_kg_k
    )


@equation
def low_pressure_turbine_work_connection(
    low_pressure_turbine_work_kj_per_kg: float,
    low_pressure_turbine: Turbine,
) -> bool:
    return (
        low_pressure_turbine_work_kj_per_kg
        == low_pressure_turbine.specific_work_kj_per_kg
    )


@equation
def high_pressure_shaft_balance(
    high_pressure_compressor_work_kj_per_kg: float,
    high_pressure_turbine_work_kj_per_kg: float,
    high_pressure_shaft_efficiency: float,
) -> bool:
    return (
        high_pressure_compressor_work_kj_per_kg
        == high_pressure_shaft_efficiency
        * high_pressure_turbine_work_kj_per_kg
    )


@equation
def low_pressure_shaft_balance(
    low_pressure_compressor_work_kj_per_kg: float,
    low_pressure_turbine_work_kj_per_kg: float,
    low_pressure_shaft_efficiency: float,
    shaft_output_work_kj_per_kg: float,
) -> bool:
    return (
        low_pressure_shaft_efficiency
        * low_pressure_turbine_work_kj_per_kg
        == low_pressure_compressor_work_kj_per_kg
        + shaft_output_work_kj_per_kg
    )


@equation
def shaft_power_relation(
    shaft_output_work_kj_per_kg: float,
    mass_flow_kg_per_s: float,
    shaft_power_kw: float,
) -> bool:
    return (
        shaft_power_kw
        == shaft_output_work_kj_per_kg * mass_flow_kg_per_s
    )


core_model = Implementation(
    TwoShaftCore,
    [
        low_pressure_compressor_inlet_connection,
        low_pressure_compressor_exit_connection,
        low_pressure_compressor_pressure_ratio_connection,
        low_pressure_compressor_efficiency_connection,
        low_pressure_compressor_gamma_connection,
        low_pressure_compressor_specific_heat_connection,
        low_pressure_compressor_work_connection,
        high_pressure_compressor_inlet_connection,
        high_pressure_compressor_exit_connection,
        high_pressure_compressor_pressure_ratio_connection,
        high_pressure_compressor_efficiency_connection,
        high_pressure_compressor_gamma_connection,
        high_pressure_compressor_specific_heat_connection,
        high_pressure_compressor_work_connection,
        high_pressure_turbine_inlet_connection,
        high_pressure_turbine_exit_connection,
        high_pressure_turbine_specific_heat_connection,
        high_pressure_turbine_work_connection,
        low_pressure_turbine_inlet_connection,
        core_exit_connection,
        low_pressure_turbine_specific_heat_connection,
        low_pressure_turbine_work_connection,
        high_pressure_shaft_balance,
        low_pressure_shaft_balance,
        shaft_power_relation,
    ],
    internal_implementations={
        "low_pressure_compressor": compressor_model,
        "high_pressure_compressor": compressor_model,
        "high_pressure_turbine": turbine_model,
        "low_pressure_turbine": turbine_model,
    },
)
"""

model_path = Path(tempfile.gettempdir()) / "bilt_two_shaft_acausal_uncoupled.py"
model_path.write_text(model_source)

spec = importlib.util.spec_from_file_location(
    "_bilt_two_shaft_acausal_uncoupled",
    model_path,
)
assert spec is not None
assert spec.loader is not None

model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

TwoShaftCore = model_module.TwoShaftCore
core_model = model_module.core_model
```

## 8. Calculate useful shaft power

For the baseline operating point we specify:

```text
inlet temperature             288 K
turbine inlet temperature    1450 K
core exit temperature        1000 K
mass flow                      20 kg/s
```

The compressor pressure ratios and efficiencies are also supplied as design assumptions.

```python exec="on" source="above" session="turbomachinery"
baseline = TwoShaftCore(
    inlet_temperature_k=288.0,
    low_pressure_compressor_pressure_ratio=2.0,
    high_pressure_compressor_pressure_ratio=6.0,
    low_pressure_compressor_efficiency=0.88,
    high_pressure_compressor_efficiency=0.86,
    turbine_inlet_temperature_k=1450.0,
    exit_temperature_k=1000.0,
    low_pressure_shaft_efficiency=0.99,
    high_pressure_shaft_efficiency=0.99,
    gamma=1.4,
    specific_heat_kj_per_kg_k=1.005,
    mass_flow_kg_per_s=20.0,
)

with core_model:
    low_pressure_compressor_exit = (
        baseline.low_pressure_compressor_exit_temperature_k
    )
    high_pressure_compressor_exit = (
        baseline.high_pressure_compressor_exit_temperature_k
    )
    high_pressure_turbine_exit = (
        baseline.high_pressure_turbine_exit_temperature_k
    )

    low_pressure_compressor_work = (
        baseline.low_pressure_compressor_work_kj_per_kg
    )
    low_pressure_turbine_work = (
        baseline.low_pressure_turbine_work_kj_per_kg
    )
    shaft_output_work = baseline.shaft_output_work_kj_per_kg
    shaft_power = baseline.shaft_power_kw

print(f"LP compressor exit: {low_pressure_compressor_exit:.1f} K")
print(f"HP compressor exit: {high_pressure_compressor_exit:.1f} K")
print(f"HP turbine exit:    {high_pressure_turbine_exit:.1f} K")
print(f"Core exit:          {baseline.exit_temperature_k:.1f} K")
print()
print(f"Useful shaft work:  {shaft_output_work:.1f} kJ/kg")
print(f"Useful shaft power: {shaft_power / 1000:.2f} MW")
```

For this deliberately simple model, the useful output is approximately:

```text
94.7 kJ/kg

1.89 MW at 20 kg/s
```

### Follow the energy through the engine

The station temperatures show where the gas gains and loses energy:

```python exec="on" html="on" source="above" session="turbomachinery"
station_names = [
    "Inlet",
    "LP compressor",
    "HP compressor",
    "Turbine inlet",
    "HP turbine",
    "Core exit",
]

station_temperatures = [
    baseline.inlet_temperature_k,
    low_pressure_compressor_exit,
    high_pressure_compressor_exit,
    baseline.turbine_inlet_temperature_k,
    high_pressure_turbine_exit,
    baseline.exit_temperature_k,
]

plt.plot(
    np.arange(len(station_names)),
    station_temperatures,
    marker="o",
)
plt.xticks(
    np.arange(len(station_names)),
    station_names,
    rotation=25,
    ha="right",
)
plt.ylabel("Temperature (K)")
plt.title("Temperature through the two-shaft core")
render_plot()
```

The compressors raise the air temperature because mechanical work is being put **into** the gas.

The turbines lower the temperature because energy is being taken **out** of the gas and converted into shaft work.

### Where does the LP turbine work go?

The LP turbine's usable shaft work is divided between the LP compressor and the external load:

```python exec="on" html="on" source="above" session="turbomachinery"
usable_low_pressure_turbine_work = (
    baseline.low_pressure_shaft_efficiency
    * low_pressure_turbine_work
)

plt.bar(
    ["LP shaft"],
    [low_pressure_compressor_work],
    label="LP compressor",
)
plt.bar(
    ["LP shaft"],
    [shaft_output_work],
    bottom=[low_pressure_compressor_work],
    label="Useful output",
)
plt.axhline(
    usable_low_pressure_turbine_work,
    linestyle="--",
    label="Usable turbine work",
)
plt.ylabel("Specific work (kJ/kg)")
plt.title("How LP turbine work is used")
plt.legend()
render_plot()
```

The stacked bar represents:

```text
LP compressor requirement
        +
useful shaft output
        =
usable LP turbine work
```

This is more informative than simply reporting the final exhaust temperature: it gives the model a useful engineering output.

## 9. Ask a more interesting inverse-design question

The forward calculation answered:

```text
given component design + operating point
                ↓
        useful shaft power
```

We can now ask a less obvious question in the other direction.

Suppose:

```text
required shaft power       2.0 MW
turbine inlet temperature 1450 K
core exit temperature      1000 K
HP compressor pressure ratio   6
mass flow                    20 kg/s
```

and ask:

> **What HP compressor efficiency is required to make that operating point possible?**

This is a useful preliminary-design question.

The HP compressor pressure ratio tells us how much the compressor must raise the air pressure. Its efficiency determines how much temperature rise — and therefore shaft work — is required to achieve that pressure ratio.

A more efficient compressor needs less shaft work for the same pressure ratio.

### Why this is a better acausal example

The compressor equation is:

```text
(T_out - T_in) × efficiency
    =
T_in × (pressure_ratio^((gamma - 1) / gamma) - 1)
```

In the forward calculation, Bilt knows:

```text
pressure ratio
efficiency
inlet temperature
```

and uses the equation to find:

```text
outlet temperature
```

In the inverse calculation, the surrounding engine model determines how much HP compressor work is available. That determines the required compressor outlet temperature, and Bilt then uses the **same compressor equation** to solve for:

```text
efficiency
```

We are not writing a second, manually rearranged compressor model.

Importantly, the pressure-ratio exponent remains evaluated in its normal direction. The current Bilt isolator does not need to invert the power operation for this query.

## 10. Solve for required HP compressor efficiency

The inverse operating point is:

```python exec="on" source="above" session="turbomachinery"
inverse_case = TwoShaftCore(
    inlet_temperature_k=288.0,
    low_pressure_compressor_pressure_ratio=2.0,
    high_pressure_compressor_pressure_ratio=6.0,
    low_pressure_compressor_efficiency=0.88,
    # high_pressure_compressor_efficiency is deliberately omitted
    turbine_inlet_temperature_k=1450.0,
    exit_temperature_k=1000.0,
    low_pressure_shaft_efficiency=0.99,
    high_pressure_shaft_efficiency=0.99,
    gamma=1.4,
    specific_heat_kj_per_kg_k=1.005,
    mass_flow_kg_per_s=20.0,
    shaft_power_kw=2000.0,
)

with core_model:
    required_high_pressure_compressor_efficiency = (
        inverse_case.high_pressure_compressor_efficiency
    )

print(
    "Required HP compressor efficiency: "
    f"{required_high_pressure_compressor_efficiency:.3f}"
)
```

The result is approximately:

```text
0.877
```

or about:

```text
87.7%
```

The baseline case used an HP compressor efficiency of `86%` and produced about `1.89 MW`.

To reach `2 MW` without changing turbine inlet temperature, mass flow, pressure ratio, or exhaust condition, the HP compressor must therefore be slightly more efficient.

### Follow the reverse computation

The requested shaft power first determines how much specific output work is needed:

```text
shaft power
    ↓
specific output work
```

The LP shaft balance then determines how much work the LP turbine must provide:

```text
specific output work
        +
LP compressor work
        ↓
required LP turbine work
```

Because the core exit temperature is known, the LP turbine equation determines the temperature entering the LP turbine — which is also the HP turbine exit temperature.

The HP turbine equation then determines how much work the HP turbine provides.

That fixes how much work is available to the HP compressor.

Finally, the HP compressor equations determine the efficiency required to achieve the fixed pressure ratio with that amount of work.

```mermaid
flowchart LR
    power["Required shaft power<br/>2 MW"] --> output["Required output<br/>specific work"]
    flow["Mass flow<br/>20 kg/s"] --> output

    output --> lpbal["LP shaft balance"]
    lpc["Known LP compressor<br/>work"] --> lpbal

    lpbal --> lptwork["Required LP<br/>turbine work"]
    exhaust["Known core exit<br/>1000 K"] --> lpt["LP turbine"]
    lptwork --> lpt

    lpt --> hptexit["HP turbine<br/>exit temperature"]
    tit["Fixed turbine inlet<br/>1450 K"] --> hpt["HP turbine"]
    hptexit --> hpt

    hpt --> hpwork["HP turbine work"]
    hpwork --> hpbal["HP shaft balance"]
    hpbal --> hpcwork["Available HP<br/>compressor work"]

    hpcwork --> hpcout["Required HP compressor<br/>outlet temperature"]
    pr["Fixed HP pressure ratio<br/>6"] --> compressor["HP compressor equation"]
    hpcout --> compressor
    compressor --> efficiency["Required HP compressor<br/>efficiency"]
```

The route crosses most of the engine model, but every step is still decomposable: no equation directly couples two sibling subsystems.

## 11. Explore the nonlinear design trade-off

Now repeat the inverse query for several shaft-power requirements while keeping the same pressure ratio and turbine inlet temperature.

```python exec="on" source="above" session="turbomachinery"
required_powers_mw = np.linspace(0.5, 2.8, 24)
required_high_pressure_compressor_efficiencies = []

for required_power_mw in required_powers_mw:
    core = TwoShaftCore(
        inlet_temperature_k=288.0,
        low_pressure_compressor_pressure_ratio=2.0,
        high_pressure_compressor_pressure_ratio=6.0,
        low_pressure_compressor_efficiency=0.88,
        turbine_inlet_temperature_k=1450.0,
        exit_temperature_k=1000.0,
        low_pressure_shaft_efficiency=0.99,
        high_pressure_shaft_efficiency=0.99,
        gamma=1.4,
        specific_heat_kj_per_kg_k=1.005,
        mass_flow_kg_per_s=20.0,
        shaft_power_kw=required_power_mw * 1000.0,
    )

    with core_model:
        required_high_pressure_compressor_efficiencies.append(
            core.high_pressure_compressor_efficiency
        )
```

```python exec="on" html="on" source="above" session="turbomachinery"
plt.plot(
    required_powers_mw,
    required_high_pressure_compressor_efficiencies,
    marker="o",
)
plt.axhline(
    1.0,
    linestyle="--",
    label="100% efficiency",
)
plt.xlabel("Required shaft power (MW)")
plt.ylabel("Required HP compressor efficiency")
plt.title("Inverse design: compressor efficiency required for shaft power")
plt.legend()
render_plot()
```

Unlike the previous turbine-temperature example, this relationship is not a straight line.

As shaft-power demand increases, less turbine work is available to drive the HP compressor. The compressor must therefore achieve the same pressure ratio using progressively less work, which requires progressively higher efficiency.

The curve also exposes a useful physical limit.

Once the required efficiency exceeds:

```text
1.0
```

the requested operating point is impossible in this model: a compressor cannot be more than 100% efficient.

For these assumptions, that limit occurs at roughly:

```text
2.68 MW
```

So the plot is doing more than showing that Bilt can run equations backwards. It turns the same model into a simple **feasibility study**.

## 12. Explore the model yourself

The easiest experiments are changes to the operating assumptions.

For example, change mass flow:

```python
mass_flow_kg_per_s=25.0
```

or change the maximum turbine inlet temperature:

```python
turbine_inlet_temperature_k=1500.0
```

You can also change the compressor pressure-ratio target:

```python
high_pressure_compressor_pressure_ratio=7.0
```

and ask again what compressor efficiency is required for a specified shaft power.

Another useful experiment is to change the core exit boundary:

```python
exit_temperature_k=950.0
```

To explore locally, edit these values while `zensical serve` is running and save the Markdown file. Zensical will rebuild the page and rerun the executable blocks.

The two main forms of the example are now:

```text
known HP compressor efficiency
        ↓
ask for useful shaft power
```

and:

```text
known required shaft power
        ↓
ask for required HP compressor efficiency
```

The component equations remain unchanged.

## 13. What this example demonstrates

### Reusable component physics

The compressor and turbine models describe local physical relationships:

```text
Compressor
    temperature rise
    pressure ratio
    efficiency
    specific work

Turbine
    temperature drop
    specific work
```

The same compressor implementation is reused for the LP and HP compressors, and the same turbine implementation is reused for both turbines.

### A physical system made from smaller systems

The enclosing implementation describes the engine architecture:

```text
gas path:
inlet → LPC → HPC → HPT → LPT → exhaust

HP shaft:
HPT → HPC

LP shaft:
LPT → LPC + useful output
```

The child systems do not need to know what they are connected to.

### A useful engineering output

The forward model ends in:

```text
shaft power
```

rather than only an intermediate station temperature.

That makes the systems-of-systems calculation easier to interpret:

> the component relationships collectively determine how much useful mechanical power the engine can deliver.

### A more interesting inverse-design problem

The reverse query is now:

> **What HP compressor efficiency is required to deliver this shaft power while holding pressure ratio and turbine inlet temperature fixed?**

This is more informative than simply asking for turbine inlet temperature because the resulting design relationship is nonlinear and exposes a physical feasibility boundary.

### Acausal evaluation

The forward and inverse calculations use the same compressor equation.

In one direction:

```text
pressure ratio + efficiency
        ↓
compressor temperature rise
```

In the other:

```text
available compressor work + pressure ratio
        ↓
required compressor efficiency
```

No manually inverted model is required.

> **Acausal modelling means describing the relationships once, then asking for the quantity you need.**

The example remains within Bilt's current capabilities because every route is decomposable and no equation directly couples multiple sibling subsystems.
