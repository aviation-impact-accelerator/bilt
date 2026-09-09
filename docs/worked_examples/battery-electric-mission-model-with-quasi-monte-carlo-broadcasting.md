# Battery electric mission model with quasi-monte carlo broadcasting

This worked example uses independently scrambled Sobol designs to evaluate a small battery-electric mission model over uncertain inputs, routes, and payload cases.

The aircraft model is deliberately simple. The point is to show the separation between:

- **engineering relationships**, written as ordinary scalar Bilt transforms; and
- **evaluation domains**, carried by the values supplied to the model.

We write one scalar model:

```text
payload → gross mass → shaft energy → electrical energy → battery mass → margin
```

and evaluate it over:

```text
8 scrambles × 1024 samples × 6 routes × 4 payload cases
```

That is **196,608 mission cases** without loops over samples, routes, or payloads.

The only explicit loop creates independent Sobol randomizations.

## The four domains

There are four distinct domains in the calculation:

| Domain | Meaning | Size |
|---|---|---:|
| `scramble` | Independent Sobol randomizations | 8 |
| `sample` | Points within each scrambled Sobol design | 1024 |
| `route` | Mission distances | 6 |
| `payload_case` | Payload alternatives | 4 |

The uncertain inputs vary over:

```text
(scramble, sample)
```

Distance varies over `(route)` and payload over `(payload_case)`.

Bilt combines these domains when values meet in a transform. The final battery result therefore lives over:

```text
(scramble, sample, route, payload_case)
```

The word **independent** is reserved here for the separate scrambled Sobol replicates. The Sobol points within one design are not independent random samples.

```python exec="on" session="qmc"
from dataclasses import dataclass
from io import StringIO
import importlib.util
from pathlib import Path
import tempfile

import matplotlib

# markdown-exec runs Python in a worker thread. Use a non-interactive backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, lognorm, norm, qmc

from bilt import Dim, Implementation, Interface, Tensor, field, system, tensor, transform


def render_plot() -> None:
    buffer = StringIO()
    plt.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close()
    print(buffer.getvalue())
```

## 1. Define the engineering model

The Bilt model is written first, before introducing Sobol sampling or array shapes.

Each transform describes one scalar engineering relationship. Nothing in these functions refers to `scramble`, `sample`, `route`, NumPy axes, or broadcasting.

```python
@system
class Mission(metaclass=Interface):
    specific_energy_kwh_per_t: Tensor = field()
    propulsive_efficiency: Tensor = field()
    energy_multiplier: Tensor = field()

    distance_km: Tensor = field()
    payload_t: Tensor = field()

    empty_mass_t: float = field()
    energy_per_km_t: float = field()
    usable_fraction: float = field()
    max_battery_mass_t: float = field()

    gross_mass_t: Tensor = field()
    shaft_energy_kwh: Tensor = field()
    electrical_energy_kwh: Tensor = field()
    battery_mass_t: Tensor = field()
    battery_margin_t: Tensor = field()


@transform
def gross_mass_t(payload_t: float, empty_mass_t: float) -> float:
    return empty_mass_t + payload_t


@transform
def shaft_energy_kwh(
    energy_multiplier: float,
    distance_km: float,
    gross_mass_t: float,
    energy_per_km_t: float,
) -> float:
    return (
        energy_multiplier
        * distance_km
        * gross_mass_t
        * energy_per_km_t
    )


@transform
def electrical_energy_kwh(
    shaft_energy_kwh: float,
    propulsive_efficiency: float,
) -> float:
    return shaft_energy_kwh / propulsive_efficiency


@transform
def battery_mass_t(
    electrical_energy_kwh: float,
    specific_energy_kwh_per_t: float,
    usable_fraction: float,
) -> float:
    return electrical_energy_kwh / (
        specific_energy_kwh_per_t * usable_fraction
    )


@transform
def battery_margin_t(
    battery_mass_t: float,
    max_battery_mass_t: float,
) -> float:
    return max_battery_mass_t - battery_mass_t


mission_model = Implementation(
    Mission,
    [
        gross_mass_t,
        shaft_energy_kwh,
        electrical_energy_kwh,
        battery_mass_t,
        battery_margin_t,
    ],
)
```

The model above is ordinary displayed Python. For this documentation build, the same definitions are loaded from a temporary Python module. This is only needed because Bilt currently obtains transform source using `inspect.getsource()`, while `markdown-exec` normally defines functions dynamically.

```python exec="on" session="qmc"
model_source = """
from bilt import Implementation, Interface, Tensor, field, system, transform


@system
class Mission(metaclass=Interface):
    specific_energy_kwh_per_t: Tensor = field()
    propulsive_efficiency: Tensor = field()
    energy_multiplier: Tensor = field()

    distance_km: Tensor = field()
    payload_t: Tensor = field()

    empty_mass_t: float = field()
    energy_per_km_t: float = field()
    usable_fraction: float = field()
    max_battery_mass_t: float = field()

    gross_mass_t: Tensor = field()
    shaft_energy_kwh: Tensor = field()
    electrical_energy_kwh: Tensor = field()
    battery_mass_t: Tensor = field()
    battery_margin_t: Tensor = field()


@transform
def gross_mass_t(payload_t: float, empty_mass_t: float) -> float:
    return empty_mass_t + payload_t


@transform
def shaft_energy_kwh(
    energy_multiplier: float,
    distance_km: float,
    gross_mass_t: float,
    energy_per_km_t: float,
) -> float:
    return (
        energy_multiplier
        * distance_km
        * gross_mass_t
        * energy_per_km_t
    )


@transform
def electrical_energy_kwh(
    shaft_energy_kwh: float,
    propulsive_efficiency: float,
) -> float:
    return shaft_energy_kwh / propulsive_efficiency


@transform
def battery_mass_t(
    electrical_energy_kwh: float,
    specific_energy_kwh_per_t: float,
    usable_fraction: float,
) -> float:
    return electrical_energy_kwh / (
        specific_energy_kwh_per_t * usable_fraction
    )


@transform
def battery_margin_t(
    battery_mass_t: float,
    max_battery_mass_t: float,
) -> float:
    return max_battery_mass_t - battery_mass_t


mission_model = Implementation(
    Mission,
    [
        gross_mass_t,
        shaft_energy_kwh,
        electrical_energy_kwh,
        battery_mass_t,
        battery_margin_t,
    ],
)
"""

model_path = Path(tempfile.gettempdir()) / "bilt_qmc_docs_model.py"
model_path.write_text(model_source)

spec = importlib.util.spec_from_file_location("_bilt_qmc_docs_model", model_path)
assert spec is not None
assert spec.loader is not None

model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

Mission = model_module.Mission
mission_model = model_module.mission_model
```

The important point is that the transforms themselves remain scalar. `battery_mass_t()` means the same thing whether it is evaluated once or over hundreds of thousands of combinations. Bilt is responsible for aligning the named domains and lifting the calculation.

## 2. Generate the uncertain inputs

A single scrambled Sobol design gives one randomized quasi-Monte Carlo estimate. To estimate uncertainty in that result, we generate several **independently scrambled designs**.

Each Sobol point has three coordinates, mapped onto:

- battery specific energy;
- propulsive efficiency;
- mission energy multiplier.

The distribution choices below are illustrative rather than claims about a validated aircraft uncertainty model.

```python exec="on" source="above" session="qmc"
@dataclass(frozen=True)
class QMCInputs:
    specific_energy_kwh_per_t: np.ndarray
    propulsive_efficiency: np.ndarray
    energy_multiplier: np.ndarray


def scrambled_sobol_inputs(
    *,
    scrambles: int = 8,
    power: int = 10,
    seed: int = 1_000,
) -> QMCInputs:
    samples = 2**power
    uniforms = np.empty((scrambles, samples, 3), dtype=float)

    # The loop creates independent randomized QMC replicates.
    # It is not a loop over model evaluations.
    for scramble_index in range(scrambles):
        engine = qmc.Sobol(
            d=3,
            scramble=True,
            seed=seed + scramble_index,
        )
        uniforms[scramble_index] = engine.random_base2(power)

    eps = np.finfo(float).eps
    uniforms = np.clip(uniforms, eps, 1.0 - eps)

    return QMCInputs(
        specific_energy_kwh_per_t=norm.ppf(
            uniforms[..., 0],
            loc=450.0,
            scale=45.0,
        ),
        propulsive_efficiency=(
            0.70
            + 0.25
            * beta.ppf(
                uniforms[..., 1],
                a=12.0,
                b=3.0,
            )
        ),
        energy_multiplier=lognorm.ppf(
            uniforms[..., 2],
            s=0.08,
            scale=1.0,
        ),
    )


draws = scrambled_sobol_inputs(scrambles=8, power=10)
print(draws.specific_energy_kwh_per_t.shape)
```

All three uncertain arrays have shape `(8, 1024)`, but that shape alone does not say what either axis means.

### Inspect the input distributions

Before running the mission model, it is useful to inspect the physical distributions produced by the transformed Sobol points.

```python exec="on" html="on" source="above" session="qmc"
plt.hist(draws.specific_energy_kwh_per_t.ravel(), bins=40)
plt.xlabel("Battery specific energy (kWh/t)")
plt.ylabel("Count")
plt.title("QMC input: battery specific energy")
render_plot()
```

```python exec="on" html="on" source="above" session="qmc"
plt.hist(draws.propulsive_efficiency.ravel(), bins=40)
plt.xlabel("Propulsive efficiency")
plt.ylabel("Count")
plt.title("QMC input: propulsive efficiency")
render_plot()
```

```python exec="on" html="on" source="above" session="qmc"
plt.hist(draws.energy_multiplier.ravel(), bins=40)
plt.xlabel("Mission energy multiplier")
plt.ylabel("Count")
plt.title("QMC input: mission energy multiplier")
render_plot()
```

These plots are primarily a sanity check: the Sobol points are generated in the unit cube and then mapped through inverse CDFs into the physical distributions used by the model.

## 3. Give the values named domains

This is the key Bilt step.

We create four dimensions and attach them to the inputs. The engineering transforms do not change; only the values carry information about where they vary.

```python exec="on" source="above" session="qmc"
scramble = Dim("scramble")
sample = Dim("sample")
route = Dim("route")
payload_case = Dim("payload_case")

distances_km = np.array(
    [150.0, 250.0, 350.0, 450.0, 550.0, 650.0]
)
payloads_t = np.array([2.0, 4.0, 6.0, 8.0])

mission = Mission(
    specific_energy_kwh_per_t=tensor(
        draws.specific_energy_kwh_per_t,
        over=(scramble, sample),
    ),
    propulsive_efficiency=tensor(
        draws.propulsive_efficiency,
        over=(scramble, sample),
    ),
    energy_multiplier=tensor(
        draws.energy_multiplier,
        over=(scramble, sample),
    ),
    distance_km=tensor(
        distances_km,
        over=route,
    ),
    payload_t=tensor(
        payloads_t,
        over=payload_case,
    ),
    empty_mass_t=20.0,
    energy_per_km_t=0.18,
    usable_fraction=0.85,
    max_battery_mass_t=7.5,
)
```

The domains now describe the calculation directly:

```text
specific energy       (scramble, sample)
propulsive efficiency (scramble, sample)
energy multiplier     (scramble, sample)
distance              (route)
payload               (payload_case)
configuration         scalar
```

## 4. Evaluate the complete domain product

```python exec="on" source="above" session="qmc"
with mission_model:
    battery_mass = mission.battery_mass_t
    battery_margin = mission.battery_margin_t

print(battery_mass.domain)
print(battery_mass.values.shape)
```

The expected shape is:

```text
(8, 1024, 6, 4)
```

corresponding to:

```text
(scramble, sample, route, payload_case)
```

There is no loop over those 196,608 cases in the model code.

```python exec="on" session="qmc"
assert battery_mass.values.shape == (8, 1024, 6, 4)
assert battery_margin.values.shape == (8, 1024, 6, 4)
```

### Current comparison boundary

The model returns a numeric margin:

```text
margin = maximum battery mass - required battery mass
```

A positive margin is feasible. We form the Boolean mask after evaluation using NumPy.

```python exec="on" source="above" session="qmc"
feasible = battery_margin.values >= 0.0
```

## 5. Look at one output distribution

Below is the required battery mass for the 450 km route and 6 t payload case across all scrambles and Sobol samples.

```python exec="on" html="on" source="above" session="qmc"
route_index = 3
payload_index = 2

selected_battery_mass = battery_mass.values[
    :,
    :,
    route_index,
    payload_index,
].ravel()

plt.hist(selected_battery_mass, bins=40)
plt.axvline(7.5, linestyle="--", label="Maximum battery mass")
plt.xlabel("Required battery mass (t)")
plt.ylabel("Count")
plt.title("450 km route, 6 t payload")
plt.legend()
render_plot()
```

## 6. Reduce the QMC domains in the right order

The individual Sobol points are **not** independent replicate estimates.

First reduce over `sample`, leaving one probability estimate per independently scrambled design, then combine those independent scramble-level estimates.

```python exec="on" source="above" session="qmc"
probability_by_scramble = feasible.mean(axis=1)

probability = probability_by_scramble.mean(axis=0)
standard_error = (
    probability_by_scramble.std(axis=0, ddof=1)
    / np.sqrt(probability_by_scramble.shape[0])
)

print(probability.shape)
print(
    probability[route_index, payload_index],
    standard_error[route_index, payload_index],
)
```

The remaining shape is `(route, payload_case)`.

## 7. Plot the route and payload sweep

```python exec="on" html="on" source="above" session="qmc"
for payload_index, payload_t in enumerate(payloads_t):
    plt.plot(
        distances_km,
        probability[:, payload_index],
        marker="o",
        label=f"{payload_t:g} t",
    )

plt.xlabel("Route distance (km)")
plt.ylabel("Estimated probability of feasibility")
plt.ylim(0.0, 1.0)
plt.legend(title="Payload")
render_plot()
```

```python exec="on" html="on" source="above" session="qmc"
image = plt.imshow(
    probability.T,
    aspect="auto",
    origin="lower",
    extent=[
        distances_km[0],
        distances_km[-1],
        payloads_t[0],
        payloads_t[-1],
    ],
    vmin=0.0,
    vmax=1.0,
)
plt.colorbar(image, label="Estimated probability of feasibility")
plt.xlabel("Route distance (km)")
plt.ylabel("Payload (t)")
plt.title("Battery-feasibility probability")
render_plot()
```

```python exec="on" html="on" source="above" session="qmc"
for payload_index, payload_t in enumerate(payloads_t):
    plt.plot(
        distances_km,
        standard_error[:, payload_index],
        marker="o",
        label=f"{payload_t:g} t",
    )

plt.xlabel("Route distance (km)")
plt.ylabel("Estimated RQMC standard error")
plt.legend(title="Payload")
render_plot()
```

## 8. Check sampling stability

The independent scrambles let us check whether the selected 450 km / 6 t result is stable as the number of Sobol points increases.

```python exec="on" source="above" session="qmc"
route_index = 3
payload_index = 2

powers = np.arange(5, 11)
sample_counts = 2**powers

convergence_probability = []
convergence_standard_error = []

for sample_count in sample_counts:
    estimates = feasible[
        :,
        :sample_count,
        route_index,
        payload_index,
    ].mean(axis=1)

    convergence_probability.append(estimates.mean())
    convergence_standard_error.append(
        estimates.std(ddof=1) / np.sqrt(estimates.size)
    )
```

```python exec="on" html="on" source="above" session="qmc"
plt.errorbar(
    sample_counts,
    convergence_probability,
    yerr=2 * np.asarray(convergence_standard_error),
    marker="o",
    capsize=4,
)
plt.xscale("log", base=2)
plt.xlabel("Sobol samples per scramble")
plt.ylabel("Estimated feasibility probability")
plt.ylim(0.0, 1.0)
plt.title("Sampling stability: 450 km route, 6 t payload")
render_plot()
```

The estimate settles close to 0.93 as the Sobol sample count increases. The error bars show approximately ±2 estimated standard errors, using variation between the eight independent scrambles.

This is a **sampling-stability check**, not a runtime benchmark or a claim about a QMC convergence rate. The different sample counts are nested prefixes of the same Sobol designs, and the final feasibility indicator is discontinuous.

## 9. Explore the model

The easiest way to explore the model is to change the inputs and rerun the page.

To explore the model locally, change the inputs below and save the page while zensical serve is running. Zensical will rebuild the page and rerun the executable blocks automatically.

For example, change the route or payload grids:

```python
distances_km = np.array(
    [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
)
payloads_t = np.array([2.0, 4.0, 6.0, 8.0])
```

or change the scalar aircraft assumptions when constructing `Mission`:

```python
mission = Mission(
    ...
    empty_mass_t=20.0,
    energy_per_km_t=0.18,
    usable_fraction=0.85,
    max_battery_mass_t=7.5,
)
```

You can also change the uncertainty assumptions in `scrambled_sobol_inputs()`, for example the mean and spread of battery specific energy:

```python
specific_energy_kwh_per_t=norm.ppf(
    uniforms[..., 0],
    loc=450.0,
    scale=45.0,
)
```

The important point is that changing the number of route cases, payload cases, or Sobol samples does **not** require changing the scalar Bilt transforms. The values and their named domains determine how broadly the model is evaluated.

## 10. What this example demonstrates

The central separation is:

**The transform describes the engineering relationship; Bilt handles how that relationship is lifted over the participating domains.**

The model remains scalar:

```text
payload
  ↓
gross mass
  ↓
shaft energy
  ↓
electrical energy
  ↓
battery mass
  ↓
margin
```

while evaluation carries:

```text
(scramble, sample, route, payload_case)
```

and the statistical analysis then reduces `sample` and `scramble` away.

The model does not need to know how the uncertain inputs were generated. The QMC sampler does not need to know how the mission calculation is implemented.

## Try it yourself

Change one thing at a time:

- increase `power` to add Sobol samples;
- change the number of independent scrambles;
- add route distances;
- add payload cases;
- change one of the illustrative uncertainty distributions;
- change the battery-mass limit;
- add another uncertain scalar input to the mission model.

None of those changes requires adding a model loop over the new cases.

A useful next extension would be a Bilt reduction API that could replace positional NumPy operations such as:

```python
feasible.mean(axis=1)
```

with named operations such as:

```python
feasible.mean(over=sample)
```
