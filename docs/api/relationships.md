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
| Python `if/elif/else` | A single conditional chain may contain `elif` and nested `if/else`; all paths must assign the same local name once and end in an `else`. |
| Inline `if` | Expressions such as `a if flag else b` work when `flag` is an ordinary Python value known when Bilt reads the equation, not a value being solved for. |

Ordinary equations can use constants from outside the function, including
values from an enclosing function. With `if/else`, constants defined at
module level work, but values captured from an enclosing function are not
currently supported.

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

Each condition must be a single comparison, such as `x > 0`. You can use
`elif` or nest `if/else` blocks. Every path must assign to the **same name**
exactly once, with a final `else` and the `return` outside the branches.
Multiple separate `if` statements are not yet supported.

Each condition must compare scalar expressions, and every branch must produce
a scalar result, not a tensor. Inputs used in **any** branch must be available,
even though only one path is executed numerically.

All branches must be valid symbolic expressions because Bilt traces every path
when building the equation. Both Python and JAX support these conditionals.

Bilt can calculate `result` when `x` is known, but cannot currently work
backwards through the `if/else` to find `x` from `result`.

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
