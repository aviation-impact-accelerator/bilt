# Equations and transforms API

Import `equation` and `transform` from `bilt`.
See [Equations](../guide/equations-and-acausal-modelling.md) and
[Transforms](../guide/transforms-and-implementations.md) for introductions.

| API | Role | Port naming |
| --- | --- | --- |
| `@equation` | Describe a relation that Bilt may rearrange. | Parameters identify participating ports; the function name identifies the equation. |
| `@transform` | Calculate one output in a fixed direction. | Parameters identify inputs; the function name identifies the output and its return annotation declares its type/dimension. |

## Equations

```python
from bilt import equation


@equation
def motion(distance: float, speed: float, duration: float) -> bool:
    return distance == speed * duration
```

The decorator calls the function with symbolic parameters and requires one
relation, normally `lhs == rhs`. The `bool` return annotation is conventional;
the returned symbolic relation is what Bilt checks.

| Expression | Meaning |
| --- | --- |
| `+`, `-`, `*`, `/`, `**`, unary `+`/`-` | Build symbolic arithmetic. |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | Construct a symbolic relation; use equality for calculational models. |
| Integer/float literals and captured quantity constants | Constants in the expression. |
| `child.port` | Reference a subsystem interface parameter; see [Systems](systems.md#subsystems). |
| Tensor reductions and selection | See [Tensor API](tensors.md#reductions). |

Use named parameters; positional-only parameters, `*args` and `**kwargs` are
refused. Equations require inspectable Python source: Bilt validates their
syntax when `@equation` is applied. Define equations in source files rather
than generating them with `exec`. Source inspection may fail in notebooks or
interactive sessions.

### Supported equation syntax

| Construct | Rule |
| --- | --- |
| Return | Exactly one final `return`, containing one symbolic relation. No tuple/list returns or early returns. |
| Local bindings | Simple assignments are allowed, including annotations, but each name is assigned once. Parameters cannot be rebound. |
| Mutation | Assignment to attributes/items, augmented assignment and other procedural statements are rejected. |
| Control flow | No `for`/`while` loops or comprehensions. A single supported `if/else` is allowed as below. |
| Function calls | Allowed when they construct supported symbolic expressions; AST validation cannot prove that arbitrary calls are pure. |
| Captured values | Constants and closures work with ordinary equations. The conditional tracing path does not yet support captured closures. |
| Ternary expressions | Python `a if flag else b` is permitted when `flag` is concrete during symbolic tracing; symbolic ternary branching is not supported. |

An ordinary equation is traced using its original function after validation.
A supported statement-level conditional is translated into a symbolic
conditional before tracing. The same symbolic analysis and execution pipeline
then applies to both.

### Conditional equations

```python
@equation
def piecewise(x: float, result: float) -> bool:
    if x > 0:
        selected = x**2
    else:
        selected = -x
    return result == selected
```

The conditional must use a single comparison, with one assignment to the
**same local name** in each branch. Nested and multiple conditionals are not
yet supported. Conditions and both branch values must be **scalar**; tensor
predicates and tensor-valued branches are not supported.

The conditional is part of the symbolic expression, so all the dependencies
in the predicate and both branches are available to structural analysis.
Both Python and JAX can evaluate supported scalar conditional equations.
Bilt can isolate values outside the conditional, but cannot currently invert
through the conditional itself. See [Conditional equations](../guide/equations-and-acausal-modelling.md#conditional-equations)
for a complete example using `Implementation` and system attributes.

Rearrangement currently handles addition, multiplication, negation and
reciprocals. Powers can appear in forward calculations, but Bilt does not
generally invert powers, reductions or selections. For single-equation isolation, the requested variable must occur once in the
simplified relation. The default causalization analyser can instead collect
repeated unknowns inside supported coupled linear equalities and solve those
blocks together. This is not a general nonlinear solver.

## Transforms

```python
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
