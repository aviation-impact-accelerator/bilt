<h1>Bilt<br><sub>Equation-oriented modelling in Python</sub></h1>

[Documentation](#documentation) · [Issues](https://github.com/aviation-impact-accelerator/bilt/issues)

![Python 3.15](https://img.shields.io/badge/python-3.15%2B-blue)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/aviation-impact-accelerator/bilt/blob/main/LICENSE)

Bilt is a Python framework for composable engineering models. Describe the relationships between quantities; Bilt derives a calculation from the values you know to the outputs you need.

## Why Bilt?

- **Acausal equations:** use the same relationship to calculate different quantities.
- **Composable systems:** connect reusable components through equations and directed transforms.
- **Physical unit analysis:** check equations for dimensional consistency, validate input quantities and convert compatible units automatically.
- **Named tensor dimensions:** align and broadcast data across time, location or scenario axes.
- **Python and JAX backends:** choose how supported models execute.

## Installation

Requires Python 3.14+, [uv](https://docs.astral.sh/uv/) and a [Rust toolchain](https://www.rust-lang.org/tools/install).

```bash
git clone https://github.com/aviation-impact-accelerator/bilt.git
cd bilt
uv sync
```

Use `uv sync --extra jax` to include the optional JAX backend.

## Quickstart

Declare **distance = speed × time**, then ask for either distance or speed:

```python
from typing import Annotated

from bilt import Implementation, Interface, Quantity, equation, field, system
from bilt.units import hour, kilometer, meter, second
from bilt.units.dimensions import LENGTH, TIME


Length = Annotated[Quantity, LENGTH]
Duration = Annotated[Quantity, TIME]
Speed = Annotated[Quantity, LENGTH / TIME]


@system
class Motion(metaclass=Interface):
    distance: Length = field()
    speed: Speed = field()
    time: Duration = field()


@equation
def kinematic_relation(distance: Length, speed: Speed, time: Duration) -> bool:
    return distance == speed * time


motion_model = Implementation(Motion, [kinematic_relation])

with motion_model:
    journey = Motion(speed=36.0 * kilometer / hour, time=5.0 * second)
    assert journey.distance == 50.0 * meter

    another_journey = Motion(distance=1.0 * kilometer, time=100.0 * second)
    assert another_journey.speed == 10.0 * meter / second
    assert another_journey.speed.convert_to(kilometer / hour).value == 36.0
```

Bilt rearranges the equation and handles unit conversions. The annotations declare physical dimensions; incompatible equations or input quantities raise dimensionality errors. Equation rearrangement is limited to supported symbolic operations.

Save the example as `motion.py` and run it with `uv run python motion.py`.

## Documentation

The [documentation](https://aviation-impact-accelerator.github.io/bilt/) includes a user guide, worked examples and an API reference covering units, transforms, subsystem composition, tensors and backends.

## Development

Developed at the [**Aviation Impact Accelerator, University of Cambridge**](https://aiazero.org). Bilt is under active development; expect API changes.

Contributions and modelling examples are welcome.

### Previewing the documentation

To preview the documentation locally while editing it, run from the repository root:

```bash
uv run --extra jax --group docs zensical serve
```

Then open [http://localhost:8000](http://localhost:8000).

### Roadmap

In order of priority, the following features are planned:

1. Migrate exhaustive search structural analysis to a more efficient algorithm
2. Detection and solving of coupled linear equations across subsystems
3. Extend Tensor indexing and reduction operations to computational model intermediates
3. Solving coupled non-linear equations across subsystems
4. Automatic differentiation through computational models
5. Solve ODE systems

### Maintainers

Bilt is maintained by [James Emberton](https://github.com/j-emberton)
and [Sam Brockie](https://github.com/brocksam) at the Aviation Impact
Accelerator, University of Cambridge.

For questions, bug reports and proposals, please open a GitHub issue, or a Github Discussion.
