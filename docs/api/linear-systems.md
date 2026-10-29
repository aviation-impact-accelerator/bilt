# Linear systems reference

Bilt discovers coupled linear blocks from ordinary equations. There is no separate
public `LinearSystem` constructor, matrix-building API, initial guess or user-managed
block object. See [Coupled linear systems](../guide/coupled-linear-systems.md) for
an introduction and the [recuperated gas turbine](../worked_examples/two-shaft-gas-turbine.md#14-close-the-loop-with-recuperation)
for a physical example.

## Structural analysis and backends

The causalization (SCC) analyser is the default for both `Implementation.evaluate`
and instance access. It discovers supported coupled linear blocks automatically.
Python and JAX are computational backends, not alternative structural analysers.
There is no public `analyser_type` argument.

## Modelling and evaluation

| Existing API | Role in a coupled solve |
| --- | --- |
| `@equation` | Declare equalities from which linear coefficients or supported linear tensor operators can be extracted. |
| `Implementation(System, equations, internal_implementations=...)` | Compose primitive relationships and child systems. |
| `implementation.evaluate(inputs, output, backend)` | Request one output using supplied values and their named domains. Other block unknowns may be solved as part of that request. |
| `with backend, implementation: instance.output` | Instance-based evaluation with the selected implementation and backend. Already calculated instance values are cached. |
| `Annotated[..., domain(axis, ...)]` | Specify exact named axes for a primitive equation parameter. |
| `Annotated[..., Domain.scalar()]` | Specify an exact scalar domain for a primitive equation parameter. |

`Domain` metadata supplements the parameter's existing type and physical-dimension
annotation. It does not provide an extent, unit, numerical initial guess or solver
tolerance. A `float` or `Quantity` annotation alone is not an exact scalar-domain hint.
Put hints on primitive equation parameters; annotating an interface field alone is
not the current linear-block hint mechanism. Repeated hints for the same qualified
port must agree on their axes.

### Requesting several block outputs

For independent requests from the same original known values, call
`implementation.evaluate` with the same input dictionary and a different target.
Compiled calculations can still be reused.

Instance access also supports successive block-output requests. Only the original
constructor inputs are supplied boundary values; reading one result does not add
an extra constraint to the next request. Requested outputs are cached individually,
so a repeated read returns the cached result. Other outputs and intermediate
calculations may be recomputed; a coupled solve does not populate a multi-output
instance cache.

## Supported linear expressions

Linearity is assessed in the unknowns selected for a particular block. Coefficients
and right-hand sides may be nonlinear functions of already available inputs.

| Expression involving block unknowns | Current support |
| --- | --- |
| Addition, subtraction and negation | Supported, including repeated occurrences of an unknown. |
| Multiplication by an expression containing no block unknowns | Supported. |
| Division by an expression containing no block unknowns | Supported as a known reciprocal coefficient. A zero divisor remains invalid numerically. |
| Product of two unknown-dependent expressions | Nonlinear; unsupported. |
| Unknown-dependent reciprocals or general powers | Unsupported by linear-block extraction. |
| `.sum(dim)`, `.mean(dim)` | Supported linear operators, including supported composed tensor expressions. |
| `.isel(dim, literal_integer)` | Supported coordinate coupling; index must be literal for an unknown-dependent selection. |
| `.sel(dim, literal_coordinate)` | Supported coordinate coupling with an established coordinate binding. |
| Unknown-dependent `.min`, `.max`, `.prod`, arbitrary calls or matrix operators | Unsupported by the current linear-block grammar. |
| A directed transform inside a cyclic block | Unsupported. Transforms can run before or after the block. |

The selected backend must also support any operations on known inputs. Accepting
an equation at definition time does not guarantee that every direction or block
formulation can be executed.

Tensor methods belong to `Tensor` parameters and supported composed tensor
expressions. A subsystem expression such as `child.x` does not currently expose
those methods directly. Use a child-local equation or a parent `Tensor` alias
before reducing or selecting it. See [Tensor API](tensors.md#intermediate-values).

## Pointwise and joint layouts

| Layout | Execution |
| --- | --- |
| Unknowns share the same domain, with pointwise affine equations | Batch independent square solves at each broadcast coordinate. |
| Unknowns have different domains, or reductions/selections couple coordinates | Assemble a joint system for the coupled block and project each output back to its own domain. |
| Parent and child equations form one block | Preserve qualified primitive equations and instance identity across the boundary. |

Python uses NumPy and JAX uses JAX numerical assembly and solving. Joint assembly
is dense. Separate blocks in a larger plan remain separate solves.

Current supported formulations have equally many primitive equation rows and
unknown ports. Joint layouts also require the multiset of row domains to equal
the multiset of unknown domains. This is more restrictive than checking only the
number of flattened scalar entries at one particular shape. General rectangular,
least-squares and constrained optimisation solves are not provided.

This restriction applies to the selected block, not every relationship in the
model. A surplus underdetermined branch can be irrelevant to a valid requested
output. Equations in the validated scope are not freely discarded to hide
competing constraints.

## Domain inference

Missing unknown hints use the same inference rules for local and qualified
cross-boundary blocks. Inference uses original equation expressions, known block
input domains, reduction/selection axes and supplied hints.

For supported joint layouts it selects a unique least layout: a feasible axis
assignment contained in every other feasible assignment. It need not be the only
feasible assignment. If there is no least assignment, add a disambiguating hint.
If no assignment satisfies the equations and exact hints, correct the model or
its hints.

Only dimensions already present in known block inputs can be inferred. The
extent and, for `sel`, coordinate labels must be bound before execution. Domain
inference does not infer missing physical units or establish numerical rank.

## Failures and retries

Import public exceptions from `bilt.errors`.

| Failure | Meaning and remedy |
| --- | --- |
| `BiltAnalysisError` | No supported plan could be realised. Inspect the message and chained cause for missing inputs, inconsistent constraints, unsupported blocks or domain-layout failures. |
| `BiltInputDimensionalityError` | A supplied value disagrees with its port's physical declaration. |
| `BiltDomainBindingError` / `BiltTensorDomainError` | Missing or incompatible named axes, extents, coordinates or operator domains. Some block-lowering failures are wrapped by an analysis error. |
| `BiltLinearSolveError` | Python numerical solve/validation failed, for example a singular system or invalid numerical solution. |
| JAX runtime exception | Numerical validation inside JAX can surface as `jax.errors.JaxRuntimeError` wrapping the callback's error. |

Planning and realisation failures can trigger alternative supported formulations.
This does not provide a nonlinear fallback or guarantee that every unsupported
operator can be reformulated. Numerical failures during evaluation are propagated;
they do not automatically restart structural search or permanently blacklist a
route for later coefficient values.

A structurally square system may still be singular for particular inputs. A
successful numerical result still needs the model's physical feasibility checks.
