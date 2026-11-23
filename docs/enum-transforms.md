## Enum matching in directed transforms

A scalar enum input can select numeric or categorical outputs using ordinary Python `match/case`.
Bilt preserves the original Python function for direct calls and compiles a supported subset to
backend-neutral conditional expressions for Python and JAX. The compiled enum values use the
same ordinal encoding as equation selectors.

```python
class Scenario(enum.Enum):
    MINIMUM = "minimum"
    AVERAGE = "average"
    MAXIMUM = "maximum"


@transform
def relative_cost(scenario: Scenario) -> float:
    match scenario:
        case Scenario.MINIMUM:
            return 0.7
        case Scenario.AVERAGE:
            return 1.0
        case Scenario.MAXIMUM:
            return 1.4
        case _:
            typing.assert_never(scenario)
```

Match subjects must be named scalar enum parameters. Guards and mixed enum types are rejected
at definition time. Each arm returns one expression, or assigns one common local followed by
a return. All enum members must be covered unless the final arm is `case _` with a value.
An exhaustive match may use `typing.assert_never(scenario)` as its unreachable fallback,
which is removed from the compiled function. Non-enum Python matches are unchanged.

As with other JAX conditionals, branch outputs must have compatible JAX types and shapes.
This feature does not implement enum tensors or arbitrary Python control-flow lowering.
