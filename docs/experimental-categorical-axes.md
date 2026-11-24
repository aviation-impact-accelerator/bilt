# Prototype: categorical coordinate axes and explicit enum coverage

This experimental PR explores Samuel Brockie's observation that CAMIA's
`scenario` is an independent variable, not necessarily an explicitly provided
tensor of selector values. It is an **alternative to the current CYOF usage**,
not a change to CYOF itself.

The existing Bilt `coordinate_axis()` represents the independent categorical
variable's available values:

```python exec="off"
class Scenario(enum.Enum):
    MINIMUM = enum.auto()
    AVERAGE = enum.auto()
    MAXIMUM = enum.auto()

scenario_axis = coordinate_axis(dim("scenario"), tuple(Scenario))

@system
class Prices(metaclass=Interface):
    scenario: Scenario | CoordinateAxis = field()
    price: Tensor = field()

@equation
def price(scenario: Scenario | Tensor, price: float) -> bool:
    match scenario:
        case Scenario.MINIMUM:
            selected = 0.65
        case Scenario.AVERAGE:
            selected = 0.70
        case Scenario.MAXIMUM:
            selected = 0.75
    return price == selected

with PYTHON_BACKEND, Implementation(Prices, [price]):
    prices = Prices(scenario=scenario_axis).price
```

The system boundary accepts the axis, binds its categorical coordinates into
the evaluation registry, and internally derives the ordinary tensor of
selector values **only for execution**. The equation retains its usual
scalar-or-tensor semantics, and output is a 1D tensor over `scenario`.
No new tensor class or enum-specific constructor is introduced.

Categorical coordinates are nominal: an `IntEnum` member is not
interchangeable with a matching integer or a member of another enum class.

## Exhaustive matching

In a Bilt model, every categorical case should be explicitly named.
An omitted member can otherwise silently inherit an unrelated wildcard
value when the enum is extended.

`bilt check .` now reports a `BILT004` error when an equation or transform
has an enum wildcard and fails to explicitly list every member. Fully
enumerated matches without a wildcard are accepted in equations.
For backward compatibility, ordinary model construction retains the
existing wildcard behaviour; the additional check runs during model
checking and does not change the backend.

Numeric `match/case` and Boolean `if/else` remain unchanged.

## Deliberately not in this prototype

- Automatic, implicit expansion of all categorical ports from global axes
- A second enum tensor class, new axis annotation syntax, or an enum-only
  evaluation context
- Changes to CYOF or its saved regression fixtures
- More permissive Python `if/else` enum comparisons

Before replacing the current CYOF approach, verify the stricter check
against existing projects and decide whether axis inputs should replace
or simply coexist with manually supplied selector tensors.
