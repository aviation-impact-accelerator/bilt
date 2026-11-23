# Equations and transforms API

Import `equation` and `transform` from `bilt`.
See [Equations](../guide/equations-and-acausal-modelling.md) and
[Transforms](../guide/transforms-and-implementations.md) for introductions.

| API | Role | Port naming |
| --- | --- | --- |
| `@equation` | Describe a relation that Bilt may rearrange. | Parameters identify participating ports; the function name identifies the equation. |
| `@transform` | Calculate one output in a fixed direction. | Parameters identify inputs; the function name identifies the output and its return annotation declares its type/dimension. |

## Equations

```python exec="on" id="wkyq36"
from bilt import equation


@equation
def motion(distance: float, speed: float, duration: float) -> bool:
    return distance == speed * duration
```

Bilt first checks the equation's Python syntax, then traces it with symbolic
parameters to capture one relationship, normally `lhs == rhs`. The `bool`
return annotation is conventional; Bilt checks the symbolic relation.

| Expression | Meaning |
| --- | --- |
| `+`, `-`, `*`, `/`, `**`, unary `+`/`-` | Build symbolic arithmetic. |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | Construct a symbolic relation; use equality for calculational models. |
| Integer/float literals and captured quantity constants | Constants in the expression. |
| `child.port` | Reference a subsystem interface parameter; see [Systems](systems.md#subsystems). |
| Tensor reductions and selection | See [Tensor API](tensors.md#reductions). |

Use named parameters; positional-only parameters, `*args` and `**kwargs` are
refused.

Bilt needs to read the Python code for each equation when `@equation` is applied.
Define equations in Python files rather than creating functions with `exec`.
Some notebooks and interactive environments may not make the source available.

### Supported equation syntax

| Construct | Rule |
| --- | --- |
| `return` | One `return` at the end, containing one relationship such as `a == b`. No early returns or tuples/lists of results. |
| Local variables | You can assign names to intermediate values, but only once per name. You cannot reassign function parameters. |
| Changing values | Do not modify object attributes, list/dictionary entries, or use `+=` and similar assignments. |
| Loops | `for`, `while` and comprehensions are not supported. |
| Function calls | Allowed when Bilt understands the mathematical result. Bilt cannot detect every side effect inside a called function. |
| Python `if/elif/else` | A single conditional chain may contain `elif` and nested `if/else`; all paths must end by assigning the same result name and end in an `else`; sequential local assignments are allowed in each branch. |
| Python `match/case` | Numeric literal cases, or enum-member cases for an enum-annotated input, followed by a final `case _`. All paths must end by assigning the same result name. Guards, captures and structural patterns are not supported. |
| Inline `if` | Expressions such as `a if flag else b` work when `flag` is an ordinary Python value known when Bilt reads the equation, not a value being solved for. |

Ordinary equations can use constants from outside the function, including
values from an enclosing function. With `if/else`, constants defined at
module level work, but values captured from an enclosing function are not
currently supported.

### Enum matching (prototype)

A scalar input annotated with a specific `enum.Enum`, `enum.IntEnum` or
`enum.StrEnum` subclass can be matched against members of **the same enum
class**. Bilt accepts enum instances at the system boundary, validates the
exact class, and retains the enum class and member identity in the symbolic
representation. Numerical ordinals are generated only when lowering to NumPy
and JAX execution.

```python exec="off"
import enum

from bilt import Implementation, Interface, equation, field, system


class Mode(enum.StrEnum):
    IDLE = "idle"
    CRUISE = "cruise"


@system
class Engine(metaclass=Interface):
    mode: Mode = field()
    output: float = field()


@equation
def output(mode: Mode, output: float) -> bool:
    match mode:
        case Mode.IDLE:
            selected = 0.0
        case Mode.CRUISE:
            selected = 5.0
        case _:
            selected = -1.0
    return output == selected


with Implementation(Engine, [output]):
    assert Engine(mode=Mode.CRUISE).output == 5.0
```

The enum class must be importable as a module-level name in the equation,
and all categorical declarations for the same port must use the exact same
class. In a parent equation, an enum port on a child interface can be a
`match` subject using its qualified name, such as `match engine.mode`.
Declare a concrete enum class directly on each scalar port that uses it.
Enum ports are forward-only match selectors. Bilt reports categorical
definition errors for flag types (`enum.Flag` and `enum.IntFlag`), union
annotations containing enums, unit or tensor annotations on enum ports, and
enum-typed transform ports. Only members of the selector's declared enum
class are valid case patterns, followed by a final `case _`.

With the JAX backend, the compiled branches must return the same numerical
dtype. For example, use `0.0` for a zero-valued branch when the other branches
return floats.

Enum values represent categorical configuration, **not** numerical or
physical quantities. For simultaneous scenario evaluation, use named tensor
dimensions instead of an enum port.

### Conditional equations

```python exec="off"
@equation
def piecewise(x: float, result: float) -> bool:
    if x > 0:
        selected = x**2
    else:
        selected = -x
    return result == selected
```

Conditions may compare scalar or tensor-valued expressions. Combine symbolic
comparisons with `&`, `|` or `~` (parenthesise comparisons); Python's
`and`, `or`, `not` and chained comparisons are not supported for symbolic
predicates. You can use `elif` or nest `if/else` blocks. Every path must
end by assigning the **same result name**, with a final `else` and the `return`
outside the branches. Each branch may use intermediate local assignments, including
values used by a nested condition. Such locals are not available after the
conditional; only the shared result is. You can nest `match/case` within
`if/else` or the other way around. An equation supports one top-level conditional.

A tensor predicate selects elementwise, broadcasting the predicate and branches
by named dimension. A scalar predicate with branches on the same domain stays
lazy; elementwise selection evaluates both branches even where the mask excludes
one. All inputs used in **any** branch must be available, and all branches must
be valid symbolic expressions. Both Python and JAX support these conditionals.
Branch results must have compatible physical dimensions and quantity kinds.

Bilt can calculate `result` when `x` is known, but cannot currently isolate
`x` backwards through a conditional. See the
[Tensor API](tensors.md#conditional-selection-in-equations) for full selection
and broadcasting rules.

See [Using `if/else` in an equation](../guide/equations-and-acausal-modelling.md#using-ifelse-in-an-equation)
for a worked example.

Rearrangement currently handles addition, multiplication, negation and
reciprocals. Powers can appear in forward calculations, but Bilt does not
generally invert powers, reductions or selections. For single-equation isolation, the requested variable must occur once in the
simplified relation. The default causalization analyser can instead collect
repeated unknowns inside supported coupled linear equalities and solve those
blocks together. This is not a general nonlinear solver at present.

## Transforms

```python exec="off" id="dqxv2h"
from bilt import transform


@transform
def kinetic_energy(mass: float, speed: float) -> float:
    return 0.5 * mass * speed**2
```

Bilt uses this only to calculate `kinetic_energy` from `mass` and `speed`.
Use explicit named input parameters. Transform source must be inspectable:
define transforms in source files. Runtime code is compiled from that source;
ordinary globals and closure state are not automatically copied into it.
Python-only dependencies can be imported inside a transform body.

The selected [backend](backends.md) determines which Python operations can run.
Named tensor reductions have a restricted return-expression form; see
[Tensors](tensors.md#reductions). Calling a decorated equation or transform
directly invokes its original Python function, not Bilt planning or backend execution.

## Declarations and units

Annotations are read when decorators run. Import or define their names at runtime,
outside `if TYPE_CHECKING:`. Every occurrence of a shared port must agree on its
physical dimension, metadata tags and absolute/relative kind.

Equations check dimensional consistency of their symbolic expressions.
Transforms check declarations and supported unit operations, rather than proving
arbitrary Python arithmetic correct. See [Units API](units.md).

## Errors

Import exceptions from `bilt.errors`.

| Exception | Meaning |
| --- | --- |
| `BiltEquationSignatureError` | Invalid equation parameter kinds. |
| `BiltEquationReturnError` | The equation returned a value/expression instead of a relation. |
| `BiltEquationDimensionalityError` | Inconsistent units, tags or absolute/relative arithmetic. |
| `BiltPortDeclarationError` | Invalid quantity/tensor annotation. |
| `BiltUnresolvedAnnotationError` | An annotation name is unavailable at decoration time. |
| `BiltDefinitionError` | Unsupported equation syntax, unavailable equation source, unsupported transform unit operation, or other invalid definition. |
| `BiltIsolationError` | Base for failed rearrangement, including missing/repeated targets and unsupported inversions; may surface through `BiltAnalysisError` when no route works. |
