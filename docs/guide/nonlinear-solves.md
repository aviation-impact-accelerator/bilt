# Nonlinear equation solves

Configure a model with initial guesses and scales, then read unknowns through normal attribute access.

```python
from typing import Annotated

from bilt import Implementation, Interface, Quantity, equation, field, system
from bilt.numerics import EvaluationOptions, RootOptions, VariableOptions
from bilt.units.dimensions import LENGTH
from bilt.units.units import meter as m

Length = Annotated[Quantity, LENGTH]
Area = Annotated[Quantity, LENGTH**2]


@system
class Rectangle(metaclass=Interface):
    width: Length = field()
    height: Length = field()
    area: Area = field()
    perimeter: Length = field()


@equation
def area_relation(width: Length, height: Length, area: Area) -> bool:
    return width * height == area


@equation
def perimeter_relation(width: Length, height: Length, perimeter: Length) -> bool:
    return 2 * (width + height) == perimeter


model = Implementation(Rectangle, [area_relation, perimeter_relation])

configured_model = model.with_options(
    EvaluationOptions(
        nonlinear=RootOptions(
            variables={
                "width": VariableOptions(initial=1.0 * m, scale=1.0 * m),
                "height": VariableOptions(initial=4.0 * m, scale=1.0 * m),
            },
            residual_scales={
                area_relation: 6.0 * m**2,
                perimeter_relation: 10.0 * m,
            },
        ),
    ),
)

with configured_model:
    rectangle = Rectangle(area=6.0 * m**2, perimeter=10.0 * m)
    width = rectangle.width  # approximately 2 m
    height = rectangle.height  # approximately 3 m
```

- **Inputs** supplied to `Rectangle(...)` are fixed. **Initial guesses** leave those quantities unknown.
- **Variable scales** give typical magnitudes for the unknowns. **Residual scales** give typical magnitudes for each equation's error. Both must be positive and have the appropriate units.
- The default solver is `ScipyHybrid`. Guesses can influence which root is found when several exist.

To select a solver or change tolerances, create another configuration:

```python
from dataclasses import replace

from bilt.numerics import RootControls, ScipyLeastSquares

settings = configured_model.options.nonlinear
alternative_model = model.with_options(
    EvaluationOptions(
        nonlinear=replace(
            settings,
            solver=ScipyLeastSquares(),
            controls=RootControls(
                max_evaluations=200,
                residual_tolerance=1e-8,
            ),
        ),
    ),
)
```

`with_options` leaves the original model unchanged. Successful block solutions are
reused across output reads on the same instance, implementation, configuration and
backend. A configured child can solve its own local block using its own options.

This version supports local scalar nonlinear systems on the Python backend;
nonlinear tensor and cross-subsystem solves are not yet supported.
