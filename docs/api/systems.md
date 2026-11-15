# Systems and implementations API

Import `system`, `Interface`, `field` and `Implementation` from `bilt`.
See [Systems and values](../guide/systems-and-values.md) and
[Systems of systems](../guide/systems-of-systems.md) for introductions.

## Systems

| API | Meaning |
| --- | --- |
| `@system` | Turn a class declared with `metaclass=Interface` into a system interface. |
| `name: Type = field()` | Declare a boundary value. `field()` takes no arguments. |
| `System(name=value, ...)` | Construct an instance using keyword arguments for known values; omit unknown values. |
| `instance.name` | Return a known value, or calculate and cache it using the selected implementation. |

Annotations declare ports; `ClassVar` annotations are excluded.
Use dimensions on pointwise unit-aware system ports, such as `Annotated[Quantity | QuantityTensor, LENGTH]`;
see [Units](units.md). Supply known values at construction.

## Implementations

| API | Meaning |
| --- | --- |
| `Implementation(system, components, *, internal_implementations=None)` | Associate a system interface with an iterable of equations/transforms and optional named child implementations. |
| `implementation.system` | The system interface being implemented. |
| `with implementation:` | Make this implementation the active choice for its system. The innermost context wins for instances without a selected implementation. |
| `implementation.evaluate(inputs, output, backend)` | Calculate one named output from a dictionary of inputs using an explicit backend. Applies the same unit checks as instance evaluation. |

```python exec="off"
from bilt import Implementation, Interface, field, system, transform


@system
class Doubling(metaclass=Interface):
    value: float = field()
    doubled: float = field()


@transform
def doubled(value: float) -> float:
    return value * 2.0


model = Implementation(Doubling, [doubled])

with model:
    instance = Doubling(value=3.0)
    assert instance.doubled == 6.0
```

An instance retains the implementation selected on its first calculated lookup
and caches requested outputs. Use a fresh instance when comparing implementations,
inputs or backends. Registering an implementation alone does not activate it.
Component order does not prescribe execution order.

## Instance derivatives (experimental)

Bilt can calculate exact scalar derivatives through supported acyclic equation
routes without requiring JAX. Ordinary property access remains unchanged; use
the `d` namespace to request a derivative of a field:

```python exec="off"
from bilt import Implementation, Interface, equation, field, system


@system
class Spring(metaclass=Interface):
    force: float = field()
    stiffness: float = field()
    displacement: float = field()


@equation
def law(force: float, stiffness: float, displacement: float) -> bool:
    return force == stiffness * displacement


implementation = Implementation(Spring, [law])

with implementation:
    spring = Spring(force=100.0, stiffness=20.0)
    assert spring.displacement == 5.0
    assert spring.d.displacement(wrt=Spring.force) == 0.05
    assert spring.d.displacement(wrt=Spring.stiffness) == -0.25
```

`spring.diff.displacement(wrt="force")` is equivalent. If a system
already declares a field named `d`, that field takes precedence and `diff`
remains available. If `diff` is also a declared field, neither convenience
namespace is installed.

Derivative queries use the constructor-supplied boundary inputs and the same
selected implementation and backend as ordinary instance lookups. Previously
calculated outputs do not become additional boundary inputs. The `wrt`
argument must identify a supplied input on the same system interface.
For a supplied output, its derivative with respect to itself is one and
with respect to another supplied input is zero.

This first prototype handles plain scalar inputs and supported `@equation`
expressions across acyclic execution plans. It rejects derivatives that require
unsupported transforms, subsystem computations or coupled solves, and
unit-bearing or tensor inputs. It does not silently use finite differences.
Derivatives are taken with respect to the numerical values in the model;
the returned result currently has no separate derivative-unit wrapper.
The generated `d` and `diff` properties are runtime additions and may
require local suppression in static type-checking tools until their typing
support is implemented.

## Evaluation and structural analysis

`Implementation.evaluate` and instance evaluation use the causalization (SCC)
structural analyser by default, with either the Python or JAX backend. It orders
ordinary calculations and solves supported coupled linear blocks, including
blocks spanning parent and child equations. There is no separate public solver
selection argument. Exhaustive analysis remains available internally for
comparison tests and benchmarks.

Instance results are cached, but only constructor-supplied values are boundary
inputs for subsequent requests. Reading one calculated output does not add an
extra supplied constraint when another output of the same block is requested.

SCC validates simultaneous constraints in the relevant scope. Two equations
that independently determine the same unknown from supplied values can therefore
be rejected as overconstrained rather than treated as alternative routes. A
square structural plan still does not guarantee a numerically nonsingular solve.

## Subsystems

Pass `internal_implementations={"child": child_model}`, then reference that
same name in a parent equation parameter annotated with the child's interface:

```python exec="off"
@equation
def connection(parent_value: float, child: Child) -> bool:
    return parent_value == child.value
```

Import `equation` from `bilt`; `Child` is an existing system interface.
Every child binding must be referenced by a parent equation, and every referenced
child needs a matching implementation of the exact interface class.

## Errors

Import exceptions from `bilt.errors`.

| Exception | Meaning |
| --- | --- |
| `BiltSystemDefinitionError` | Missing `metaclass=Interface`. |
| `BiltUnknownSystemAttributeError` | An unset value was requested without a selected or active implementation. |
| `BiltAnalysisError` | No valid route can produce the requested output. |
| `BiltInputDimensionalityError` | An input disagrees with its port declaration. |
| `BiltImplementationDimensionalityError` | Relationships or interface declarations disagree. |
| `BiltSubSystemMissingImplementationError` | A referenced child has no binding. |
| `BiltSubSystemMismatchedImplementationError` | A binding implements a different child interface. |
| `BiltSubSystemUnreferencedEntryError` | A child binding is not used by any equation. |
| `BiltSubSystemConflictError` | The same child name refers to different interfaces. |
| `BiltSubSystemNameCollisionError` / `BiltNameError` | A child name collides with a component or uses the reserved `__` separator. |
