# Tensor API

Reference for tensors, named dimensions, domains, reductions and selection.
Import these public helpers from `bilt`; see [Domains and tensors](../guide/dimensions-domains-and-tensors.md)
for the modelling introduction and [Units API](units.md) for physical units.

## Dimensions and domains

| API | Meaning |
| --- | --- |
| `dim(name)` / `Dim(name)` | Create a named array dimension. In the current implementation, dimensions compare by name: two `dim("time")` values refer to the same dimension. Use distinct names for independent axes. |
| `d.name` | Dimension name. |
| `domain(*dims)` | Construct an ordered domain, e.g. `domain(location, time)`. |
| `Domain(dims=())` | Construct a domain from a tuple of dimensions. Repeated dimensions are refused. |
| `Domain.scalar()` / `domain()` | Empty domain: a scalar varies over no dimensions. |
| `as_domain(value)` | Accept an existing `Domain`, one `Dim`, or an iterable of dimensions. |
| `d.dims`, `d.is_scalar` | Ordered dimension tuple and whether it is empty, for a domain `d`. |
| `coordinate_axis(dimension, coordinates)` | Construct ordered coordinate labels for a dimension. Does not attach them to a tensor or bind them to an evaluation by itself. |
| `CoordinateAxis(dimension, coordinates)` | Direct constructor; pass a tuple of coordinates. |
| `axis.dimension`, `axis.coordinates`, `len(axis)` | The labelled dimension, coordinate tuple and number of labels. |

Coordinates are strings, integers, floats, `datetime.date` or `datetime.datetime`.
A named dimension such as `time` describes an array axis, not a physical
dimension such as `bilt.units.dimensions.TIME`.

## Constructing and inspecting tensors

```python
from bilt import Tensor, dim, domain, tensor

location = dim("location")
time = dim("time")
values = tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], over=(location, time))

assert values.shape == (2, 3)
assert values.domain == domain(location, time)
assert tensor(2.0).domain.is_scalar
```

| API | Meaning |
| --- | --- |
| `tensor(values, *, over=None, domain=None, unit=None)` | Convert values with `numpy.asarray` and associate each positional axis with a named dimension. |
| `tensor(values, over=time)` | One-dimensional data. |
| `tensor(values, over=(location, time))` | Data whose axes are in the specified order. |
| `tensor(values, domain=domain(location, time))` | Equivalent explicit-domain form. Supply `over` or `domain`, not both. |
| `tensor(value)` / `Tensor.scalar(value, unit=None)` | Rank-zero tensor; `Tensor.scalar` rejects arrays with rank greater than zero. |
| `Tensor(values, domain, unit=None)` | Direct constructor with an explicit `Domain`. |
| `t.values` | NumPy array containing the numerical data. |
| `t.domain`, `t.shape`, `t.ndim` | Named domain, numerical shape and rank. |
| `t.unit` | Unit or `None` for an unlabelled tensor. |

Non-scalar data requires an explicit domain, and array rank must equal the
number of domain dimensions. Dimension extents are checked across values during
evaluation: one dimension name cannot have two different lengths, even if one
length is one. Missing dimensions can broadcast; conflicting lengths cannot.

The tensor dataclass is frozen, but its NumPy array is not made read-only and
`numpy.asarray` need not copy the input.

## Units

| API | Meaning |
| --- | --- |
| `tensor(values, over=time, unit=meter)` | Label every value with a physical unit. |
| `t.convert_to(unit)` | Return a tensor with rescaled values and the same domain. Applies absolute offsets where appropriate. Requires a compatible existing unit. |
| `t * unit` | Label an unlabelled tensor or compose a labelled tensor's unit. Values are unchanged. |
| `unit * t`, `t / unit` | Compose units on an already labelled tensor. Premultiplying or dividing an unlabelled tensor is refused. |

Declare unit-aware model ports with `Annotated[Tensor, LENGTH]`, for example.
Bare `Tensor` ports are outside the physical unit system. A declared
`Annotated[Quantity, LENGTH]` port can also accept compatible tensor inputs;
an explicitly Tensor-declared quantity requires a tensor, including at rank zero.
Computed declared quantities use coherent units; convert returned tensors for display.

## Operations in models

Arithmetic in equations uses named-domain broadcasting. Shared dimensions align;
dimensions absent from an operand broadcast. Output dimensions follow the
evaluation's canonical order, derived from dimension binding order, which need
not equal an individual input's axis order.

There is no general named-domain API for slicing, reshape, transpose,
concatenation, matrix multiplication or boolean masking.

Concrete `Tensor` objects are data containers. Calling `t.mean(time)`, `t.isel(time, 0)`
or the other symbolic methods directly on a concrete tensor raises `TypeError`,
even inside an active implementation context. Ordinary concrete tensor arithmetic
raises `NotImplementedError`, except for the unit operations above. Use these
methods in decorated relationships, or work explicitly with `t.values` outside models.

| Operation | Equations | Transforms |
| --- | --- | --- |
| Arithmetic | Symbolic arithmetic and named-domain broadcasting. | Backend numerical arithmetic; domain inference is limited for arbitrary Python. |
| `x.mean(dim)`, `.sum(dim)`, `.min(dim)`, `.max(dim)`, `.prod(dim)` | Supported on tensor-valued parameters. | Supported in the restricted return-expression form described below. |
| `x.isel(dim, index)` | Supported with a scalar positional selector. | Not currently lowered; do not use as a named tensor operation in a transform. |
| `x.sel(dim, coordinate)` | Supported with a literal coordinate and a bound coordinate axis. | Not currently lowered. |
| `x[...]` | No Tensor/symbolic subscription API. | Raw backend indexing may execute, but does not infer a changed named domain. |

## Reductions

Each method takes **one named dimension**, which must occur in the operand's
domain. There is no public `axis=`, `keepdims=`, tuple-of-dimensions, `dtype=` or
`where=` argument for these methods.

| Method | Result |
| --- | --- |
| `x.mean(dim)` | Arithmetic mean over the dimension. |
| `x.sum(dim)` | Sum over the dimension. |
| `x.min(dim)` | Minimum over the dimension. |
| `x.max(dim)` | Maximum over the dimension. |
| `x.prod(dim)` | Product over the dimension. |

The result's logical domain removes the reduced dimension. Backend operations
retain singleton positional axes internally for alignment; those absent from the
final domain are squeezed when returning the result. Reducing the last dimension
produces a scalar domain. An explicitly unit-aware Tensor declaration retains a
rank-zero `Tensor`; other scalar-domain outputs are returned as a numerical scalar
or a declared `Quantity`, as applicable.

In unit-aware equations, mean/sum/min/max retain physical dimensionality and
tags. A product reduction requires a dimensionless operand, because the unit
power would otherwise depend on the axis length. Summing absolute readings is
refused; mean/min/max and selection preserve absolute readings. These equation
checks are not a general proof of dimensional correctness for transform bodies.

### Transform reduction restrictions

Use a single return statement with the reduction applied directly to a named
parameter, and a dimension captured by a simple global or closure name:

```python
from bilt import Tensor, transform

@transform
def averaged(values: Tensor) -> Tensor:
    return values.mean(time) * 2.0
```

The recognised return expression consists of parameter names, constants, binary
arithmetic and direct parameter reductions. Local assignments, branches,
reductions on `(a + b)`, chained reductions and inline `dim("time")` arguments
are not supported by this lowering path. Unrecognised bodies retain the union
of parameter domains as a fallback; an arbitrary backend operation that changes
shape does not therefore acquire a correct new domain automatically.

## Indexing and coordinate selection

| API | Selector | Effect |
| --- | --- | --- |
| `x.isel(dim, index)` | One integer position, or a symbolic scalar integer input/expression in an equation. | Select one element along `dim` and remove `dim` from the logical domain. |
| `x.sel(dim, coordinate)` | One literal coordinate, including a captured Python coordinate constant. | Resolve the label to a position during lowering, then perform the same selection as `isel`. |

`isel` is zero-based. Use integer positions; Bilt checks that a symbolic selector
has a scalar domain but does not separately enforce integer dtype or check bounds.
The backends use `numpy.take` and `jax.numpy.take`; their invalid-index behaviour
can differ. Valid negative positions count from the end.

`sel` needs coordinate labels as well as a known dimension length. Tensor values
do not carry those labels. Currently, explicit coordinate binding uses the
internal `EvaluationContext` API:

```python
from bilt import coordinate_axis
from bilt.core._evaluation_context import EvaluationContext

# Bind before requesting any calculated output in this evaluation.
with EvaluationContext() as evaluation:
    evaluation.domains.bind(coordinate_axis(time, (2024, 2025, 2026)))
    # Request a model output whose equation uses values.sel(time, 2025).
```

The axis length must agree with tensor inputs. A label must be present: there is
no interpolation or nearest-neighbour option. Lookup returns the first matching
coordinate. Float-to-float matches allow a small rounding tolerance of
`8 * numpy.finfo(float).eps * max(1, abs(candidate), abs(selector))`.
Runtime symbolic coordinate inputs, label ranges and lists are not supported.

### Intermediate values and new dimensions

**Selection can consume a named intermediate port whose domain is already known
over existing dimensions. It cannot introduce a new dimension or discover a new
axis extent during the calculation.**

Bilt binds dimension sizes from supplied inputs and any explicitly bound axes.
The first access to the evaluation's canonical order locks that registry against
new dimensions and against adding coordinate labels to a size-only binding.
Compilation then propagates domains through planned intermediate ports. Merely
creating a `Dim` object does not bind its size or labels.

| Case | Current behaviour |
| --- | --- |
| Select from an input tensor | Supported in an equation when the selected dimension is present. |
| Select from a named intermediate over input dimensions | Supported when its domain can be propagated from earlier relationships. |
| Reduce/select an existing dimension | Removes a logical dimension; does not create a replacement dimension. |
| Slice a computed array to a shorter axis | No named slicing API or automatic new domain/extent inference. |
| Create a dimension only inside a transform or intermediate calculation | No automatic registration or extent inference. New registry bindings after locking raise `BiltDomainRegistryLockedError`. |
| Select several positions, use a boolean mask, or perform vectorised gathers | Not supported by the scalar-selector API; cannot create an output index dimension. |
| Call `(values * 2.0).isel(time, 0)` or `values.mean(time).isel(location, 0)` | Not supported: arithmetic and these methods return a plain symbolic `Expr`, which lacks tensor methods. Name the intermediate in another relationship. |

Declaring a new dimension name alone does not make shape-changing intermediate
indexing work. Prepare sliced/reshaped data outside the model with an explicit
domain, then supply it as an input to a separate evaluation. For repeated
reductions or selections over existing dimensions, split the operations into
named intermediate ports.

### Example: select and reduce a named intermediate

```python
from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor

location = dim("location")
time = dim("time")

@system
class Measurements(metaclass=Interface):
    values: Tensor = field()
    doubled: Tensor = field()
    averaged: Tensor = field()
    selected: Tensor = field()

@equation
def doubling(doubled: Tensor, values: Tensor) -> bool:
    return doubled == values * 2.0

@equation
def averaging(averaged: Tensor, doubled: Tensor) -> bool:
    return averaged == doubled.mean(time)

@equation
def selecting(selected: Tensor, doubled: Tensor) -> bool:
    return selected == doubled.isel(time, 1)

with Implementation(Measurements, [doubling, averaging, selecting]):
    sample = Measurements(
        values=tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], over=(location, time)),
    )
    averaged = sample.averaged
    selected = sample.selected

assert averaged.domain.dims == (location,)
assert selected.domain.dims == (location,)
assert averaged.values.tolist() == [6.0, 8.0]
assert selected.values.tolist() == [6.0, 8.0]
```

## Errors and limitations

Import exceptions from `bilt.errors`; match exception classes rather than whole
message strings. Some failures surface during evaluation or route compilation.

| Exception / message excerpt | Cause / remedy |
| --- | --- |
| `BiltTensorDomainError`: `Non-scalar values need an explicit domain` | Supply `over=` or `domain=`. |
| `BiltTensorDomainError`: `Value rank … does not match domain rank …` | Match one dimension to each positional axis. |
| `BiltInvalidDomainError`: `same dimension more than once` | Use each dimension only once in a domain. |
| `BiltDomainBindingError`: `inconsistent sizes` / `conflicting coordinate definitions` | All uses of a dimension must agree on extent and coordinate labels. |
| `BiltTensorDomainError`: `Cannot reduce/index dimension … not present in domain` | Choose a dimension that the operand actually retains. |
| `BiltTensorDomainError`: `currently requires a scalar selector` | Supply one position, not a tensor of positions. |
| `BiltDomainBindingError`: `not bound` / `has no coordinate axis` | Bind dimensions/coordinate axes before compilation; a `sel` needs labels, not just a size. |
| `BiltDomainBindingError`: `Coordinate … is not present` | Use a label present on the bound axis. |
| `BiltDomainRegistryLockedError` | A new dimension or coordinate-axis upgrade was attempted after canonical order was fixed. Prepare bindings before calculation. |
| `BiltSymbolicLoweringError`: `sel requires a literal coordinate selector` | Coordinate selection must resolve at lowering time. |
| `TypeError`: `Tensor operations are symbolic` | Do not call model-only methods on concrete tensors. |
| `AttributeError` for a tensor method on `Expr` | Split inline/chained tensor operations into named intermediate ports. |
| `NotImplementedError`: `requires exactly one named dimension` / `currently require a named dimension` | Use `parameter.mean(time)` with one positional dimension name in a transform. |
| `BiltEquationDimensionalityError` | Incompatible units/tags, dimensional product reduction or a sum of absolute readings. |
| `BiltUnitError` / `BiltDimensionalityError` | Missing tensor unit or incompatible conversion; see [Units API](units.md#errors-and-messages). |

This reference describes the current implementation in
[`_tensors.py`](https://github.com/aviation-impact-accelerator/bilt/blob/main/src/bilt/core/_tensors/_tensors.py),
[`_expression_lowering.py`](https://github.com/aviation-impact-accelerator/bilt/blob/main/src/bilt/core/_tensors/_expression_lowering.py),
[`_transform_lowering.py`](https://github.com/aviation-impact-accelerator/bilt/blob/main/src/bilt/core/_tensors/_transform_lowering.py)
and [`_domain_registry.py`](https://github.com/aviation-impact-accelerator/bilt/blob/main/src/bilt/core/_domain_registry.py).
