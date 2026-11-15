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

## Parameter derivatives (experimental)

Bilt supports exact symbolic first and second derivatives of scalar model
outputs with respect to *constructor-supplied scalar inputs*. The API has three
namespaces:

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

    # One input: scalar first derivative
    assert spring.grad.displacement(wrt=Spring.force) == 0.05

    # Multiple inputs: ordered tuple
    assert spring.grad.displacement(wrt=(Spring.force, Spring.stiffness)) == (0.05, -0.25)

    # Explicit outputs and inputs, always a matrix
    jacobian = spring.jacobian(
        (Spring.displacement,),
        wrt=(Spring.force, Spring.stiffness),
    )

    # Second derivatives as a square tuple-of-tuples
    hessian = spring.hessian.displacement(
        wrt=(Spring.force, Spring.stiffness),
    )
```

A single field selector to `grad` returns a scalar; a non-empty tuple returns
an ordered tuple, even for one field. `jacobian` takes explicit non-empty
tuples of outputs and `wrt` inputs; its rows follow the order of outputs and
columns follow the order of inputs. `hessian` accepts a non-empty tuple of
inputs and returns a square matrix represented as tuples of tuples. Field
selectors can be class properties or their names as strings.

The `d` and `diff` aliases and the `jacobian.field(...)` form are not part
of this API. Existing fields named `grad`, `jacobian` or `hessian` take
precedence over the generated namespace.

Queries use the instance's selected implementation and active Python or JAX
backend, and do not modify its boundary inputs. Previously calculated outputs
remain derived quantities. A scalar output already supplied as a constructor
input has first derivatives of one with respect to itself and zero with respect
to other supplied inputs; all its second derivatives are zero.

Derivatives use forward-mode symbolic JVPs on supported scalar acyclic
`@equation` routes. The Hessian applies a second JVP to the differentiated
execution plan; it is **not** a finite-difference estimate. Unit-bearing or
tensor inputs, unsupported transforms and coupled solves are not supported
in this initial query layer. Returned values do not yet carry derivative
units. The generated namespace properties are runtime-defined and might need
static type-checker suppressions.

Conditional equations support branch-wise derivatives across scalar
`if/elif/else` and nested branches. At a switching point the result is the
selected branch's derivative, which need not be the mathematical derivative of
the piecewise function. Conditions themselves are not differentiated, and
conditional equations cannot be inverted through predicates depending on the
target variable.

## Differentiation inside equations (experimental)

You can also differentiate a **local symbolic expression** as part of an
`@equation` definition. These are standalone functions from `bilt`, rather
than queries on an instantiated model:

```python exec="off"
from bilt import equation, grad, hessian, jacobian


@equation
def energy_sensitivity(x: float, y: float, sensitivity: float) -> bool:
    expression = x**3 + x * y + y**2
    return sensitivity == grad(expression, wrt=x)


@equation
def mixed_curvature(x: float, y: float, curvature: float) -> bool:
    expression = x**3 + x * y + y**2
    return curvature == hessian(expression, wrt=(x, y))[0][1]


@equation
def cross_sensitivity(x: float, y: float, sensitivity: float) -> bool:
    outputs = (x**2 + y, x * y)
    return sensitivity == jacobian(outputs, wrt=(x, y))[1][0]
```

`grad(expression, wrt=x)` returns a symbolic scalar expression;
`wrt=(x, y)` returns a tuple of symbolic partials. `hessian` and
`jacobian` return tuples of tuples, which can be indexed to select a
scalar equation expression. They lower through the usual symbolic
simplification, dimensional analysis and backend code generation for
Python and JAX, with no finite-difference fallback.

**Scope:** `wrt` variables must be the symbolic parameter expressions
provided to the equation. These functions differentiate the supplied
expression tree only: `grad(y, wrt=x)` is zero when `x` and `y`
are separate parameter symbols, even if another model equation relates
them. For derivatives *through an implemented model*, use instance
`model.grad`, `model.jacobian` or `model.hessian`.
These expression operators do not introduce spatial derivatives or
infer implicit sensitivities through other equations.

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
