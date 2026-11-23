## Enum matching in directed transforms

A scalar enum input can select numeric or categorical outputs using ordinary Python `match/case`.
Bilt preserves the original Python function for direct calls and compiles a supported subset to
backend-neutral conditional expressions for Python and JAX. The compiled enum values use the
same ordinal encoding as equation selectors.

```python exec="off"
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


## Tensor-valued enum selectors in equations

Use ordinary `Tensor` values and the named-domain `tensor()` constructor; no
categorical tensor class or constructor is needed. A port declared as
`Scenario | Tensor` accepts either a member of `Scenario` or a `Tensor`
containing members of exactly that enum class.

```python exec="off"
class Scenario(enum.Enum):
    MINIMUM = enum.auto()
    AVERAGE = enum.auto()
    MAXIMUM = enum.auto()


@system
class PriceModel(metaclass=Interface):
    scenario: Scenario | Tensor = field()
    price: Tensor = field()


@equation
def price(scenario: Scenario | Tensor, price: float) -> bool:
    match scenario:
        case Scenario.MINIMUM:
            selected = 0.65
        case Scenario.AVERAGE:
            selected = 0.70
        case _:
            selected = 0.75
    return price == selected


all_scenarios = tensor(
    [Scenario.MINIMUM, Scenario.AVERAGE, Scenario.MAXIMUM],
    over=dim("scenario"),
)
```

Scalar enum matches retain scalar conditional execution. For a tensor selector,
Bilt compares encoded enum ordinals elementwise and lowers to its existing
named-domain `WHERE` operation, broadcasting scalar or tensor branches as
required. Inputs retain nominal enum identity at the API boundary and are
encoded only for backend execution. Numeric tensors do not implicitly become
enum tensors; a tensor constructed from raw integer codes is rejected.
Tensor-valued enum outputs of directed transforms are outside this PR's scope.
