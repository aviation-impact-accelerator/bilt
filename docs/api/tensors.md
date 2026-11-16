# Tensor API

Import tensor helpers from `bilt`. See [Domains and tensors](../guide/dimensions-domains-and-tensors.md)
for an introduction, [Units API](units.md) for physical units, and
[Quantity annotations and typing](quantity-annotations.md) for scalar/tensor system fields.

## Construction and inspection

| API | Meaning |
| --- | --- |
| `dim(name)` / `Dim(name)` | Named array dimension. Dimensions compare by name; use different names for independent axes. |
| `domain(*dims)` / `Domain(dims=())` | Ordered domain with no repeated dimensions. |
| `domain()` / `Domain.scalar()` | Empty domain for scalar values. |
| `as_domain(value)` | Convert a `Dim` or iterable of dimensions to a domain; return an existing domain unchanged. |
| `tensor(values, *, over=None, domain=None, unit=None)` | Construct a tensor from array-like data. Supply either `over` or `domain` for non-scalar data. |
| `Tensor(values, domain, unit=None)` | Direct constructor with an explicit domain. |
| `tensor(value)` / `Tensor.scalar(value, unit=None)` | Rank-zero tensor. |
| `t.values`, `t.shape`, `t.ndim` | NumPy array, shape and rank. |
| `t.domain`, `t.domain.dims`, `t.domain.is_scalar` | Domain, ordered dimensions and scalar-domain flag. |
| `t.unit` | Physical unit, or `None`. |
| `dimension.name` | Dimension name. |

```python exec="on" session="tensor-api"
from bilt import dim, domain, tensor

time = dim("time")
values = tensor([1.0, 3.0, 5.0], over=time)
assert values.domain == domain(time)
assert values.shape == (3,)
```

Array rank must match the number of dimensions. During evaluation, shared
dimensions must have equal lengths. Missing dimensions broadcast; shared ones
align by name. Result axes follow the evaluation's dimension binding order.

## Units

| API | Meaning |
| --- | --- |
| `tensor(values, over=time, unit=meter)` | Label the numerical values with a unit. |
| `t.convert_to(unit)` | Rescale values to a compatible unit, retaining the domain. |
| `t * unit` | Label an unlabelled tensor or compose its unit; values are unchanged. |
| `unit * t`, `t / unit` | Compose units on an already labelled tensor. |

Use `Annotated[QuantityTensor, LENGTH]` for tensor-only unit-bearing fields or equation parameters. For pointwise quantities that can be scalars or tensors, use `Annotated[Quantity | QuantityTensor, LENGTH]`. Dimensioned `Annotated[Tensor, LENGTH]` declarations are no longer supported.
Bare `Tensor` ports are outside the physical unit system. With a non-optional
unit, `tensor(..., unit=...)` returns `QuantityTensor`; without one it returns
`Tensor`. Computed declared
quantities use coherent units; convert returned tensors for display.

## Reductions

Use these methods in decorated relationships. Each takes one named dimension
present in the operand's domain and removes it from the result's domain.

| Method | Result |
| --- | --- |
| `x.mean(dim)` | Arithmetic mean. |
| `x.sum(dim)` | Sum. |
| `x.min(dim)` | Minimum. |
| `x.max(dim)` | Maximum. |
| `x.prod(dim)` | Product. |

In unit-aware equations, mean/sum/min/max retain dimensionality and tags.
`prod` requires a dimensionless operand; `sum` refuses absolute readings.
Reducing the last dimension gives a scalar domain. Unit-aware Tensor-declared
outputs remain rank-zero tensors; other scalar-domain outputs are numerical
scalars or declared quantities.

**Transforms:** use a single return statement, applying reductions directly to
named parameters with a dimension captured by a global or closure name:

```python exec="off"
from bilt import Tensor, transform


@transform
def averaged(values: Tensor) -> Tensor:
    return values.mean(time) * 2.0
```

## Concatenating domains

Use `Tensor.concat(..., along=..., into=...)` in an `@equation`
to join data from different named dimensions onto a new dimension.
This operation changes both the output domain and its extent.

```python exec="off"
@equation
def combine(
    historical: Tensor,
    projected: Tensor,
    combined: Tensor,
) -> bool:
    return combined == Tensor.concat(
        (historical, projected),
        along=(historical_date, modeling_date),
        into=combined_date,
    )
```

In this example, `size(combined_date) = size(historical_date)
+ size(modeling_date)`. The input axes may be of different lengths.
All *other* axes must match by name; their lengths are already validated
by Bilt's domain registry. Inputs must have compatible physical units.
The output retains the physical unit and absolute/relative kind.

If every source axis has coordinate labels, Bilt concatenates those labels
in order and registers a coordinate axis for the destination. If none
has labels, only the new extent is registered. Mixed labelled/unlabelled
axes are rejected rather than silently dropping some coordinates.

The destination axis must have a different name from the source axes.
Concatenation currently supports forward calculations in `@equation`.
Reverse isolation, coupled linear blocks, and `@transform` support
are not included in this first implementation.


## Positional slicing

The existing `isel(dim, index)` API selects one scalar position and removes
the selected axis. Pass a Python `slice` and an `into=` dimension to select
multiple positions and retain a named axis with a newly derived extent:

```python exec="off"
@equation
def select_period(values: Tensor, result: Tensor) -> bool:
    return result == values.isel(time, slice(1, 10, 2), into=selected_time)
```

The slice follows Python's positional semantics: the stop is exclusive;
omitted, negative and out-of-range bounds are supported, including negative
steps and empty selections. Bounds and steps must be literal integers or
`None`, and a zero step is invalid. A slice must use a different output
dimension name, while other named axes and physical units are preserved.

The derived dimension receives the sliced coordinate labels if the input
dimension has coordinate labels, otherwise Bilt infers only its new extent.
Like concatenation, positional slicing is a forward-only `@equation`
operation, not an eager method on a concrete tensor or a general reshape.
Coordinate-based slice selection with `sel()` is separate follow-up work.

## Generate dimensions from model inputs

`Tensor.arange(start, stop, over=..., inclusive=...)` creates a named axis
whose extent is derived from scalar input values. Unlike ordinary tensor
operations, the length is resolved before numerical execution and the backend
receives a statically sized range:

```python exec="off"
@equation
def modeling_years(start: float, end: float, years: Tensor) -> bool:
    return years == Tensor.arange(
        start,
        end,
        over=modeling_date,
        inclusive=True,
    )
```

For `start=2025` and `end=2030`, the result has six elements and
coordinate labels `(2025, 2026, 2027, 2028, 2029, 2030)`. By default the
stop is exclusive, matching Python `range`. Positive and negative literal
integer steps are supported.

A `unit=` can be supplied when generating a physical quantity, including
`absolute_year` for calendar dates. Inputs may be ordinary scalars or
unit-bearing scalar quantities; the latter are converted to the requested unit.
The output values are represented in coherent units inside the numerical
backend while coordinate labels retain the requested values.

**Current limitations:** start and stop must be integer-valued literals or
**directly supplied scalar inputs**. Scalar intermediates calculated by another
equation cannot yet determine a generated axis. Non-integral, infinite and
unresolved boundaries raise a domain binding error. The generated axis is
limited to 100,000 values to prevent excessive planning allocations.
Cycles involving an axis and its own generation are unsupported.

This first implementation targets forward `@equation` execution. Static
specialisation occurs separately for each evaluation so inputs may yield
different output lengths; NumPy and JAX both receive a fixed shape after
structural planning. Arbitrary dynamic dimensions and transforms remain
follow-up work.

## Coordinate-aware slicing

Use `sel(dim, slice(start_label, stop_label), into=new_dim)` in an
`@equation` to select a labelled interval while creating a new named
dimension. **Both coordinate endpoints are inclusive**, unlike positional
`isel()` slices. Labels are looked up on a bound coordinate axis, and
selection follows their existing order rather than numerically sorting them:

```python exec="off"
@equation
def select_years(values: Tensor, selected: Tensor) -> bool:
    return selected == values.sel(
        time,
        slice(2025, 2050),
        into=selected_time,
    )
```

Both endpoints may be omitted. An optional integer step selects every
`step`-th position; negative steps reverse the traversal. A coordinate
endpoint that is not found raises a binding error rather than being clipped.
The result retains physical units and the names of unaffected dimensions,
and its new extent and coordinate labels are derived automatically.

This operation requires a labelled source axis and an explicit, distinct
`into` dimension. Existing single-label `sel(dim, label)` is unchanged:
it selects one value and removes that axis. Coordinate slicing is currently
forward-only and is not supported within coupled linear solves or transforms.

## Indexing

Use selection in **equations**; `.isel()` and `.sel()` are not lowered in transforms.
With scalar selectors, both select one element along a dimension and remove that dimension from the result.

| Method | Selector |
| --- | --- |
| `x.isel(dim, index)` | Zero-based integer position or symbolic scalar integer expression. Valid negative positions count from the end. |
| `x.sel(dim, coordinate)` | Literal coordinate, including a captured Python constant, on a bound coordinate axis. |
| `x.sel(dim, slice(start, stop, step), into=new_dim)` | Inclusive coordinate interval on a labelled axis, with a new dimension. |

`isel` delegates bounds handling to the backend; invalid-index behaviour may differ.
`sel` requires a matching label. Coordinates may be strings, integers, floats,
`datetime.date` or `datetime.datetime`; float matches allow rounding tolerance.

### Coordinate axes

| API | Meaning |
| --- | --- |
| `coordinate_axis(dimension, coordinates)` | Construct an axis from iterable labels. |
| `CoordinateAxis(dimension, coordinates)` | Direct constructor with a coordinate tuple. |
| `axis.dimension`, `axis.coordinates`, `len(axis)` | Dimension, labels and length. |

Creating an axis does not bind it. Currently, bind coordinates through the
internal evaluation context **before requesting calculated outputs**:

```python exec="on" session="tensor-api"
from bilt import coordinate_axis
from bilt.core._evaluation_context import EvaluationContext

with EvaluationContext() as evaluation:
    evaluation.domains.bind(coordinate_axis(time, (2024, 2025, 2026)))
    # Request a model output using values.sel(time, 2025).
```

### Intermediate values

Selection works on **named intermediate ports over known dimensions**.
Source dimensions and coordinate axes must be established before calculation.
For supported domain transformations such as slicing or concatenation, Bilt
can derive the output dimension and its extent during structural planning.

Supported tensor arithmetic and reduction/selection expressions retain tensor
methods in equations, so composed receivers such as `(weights*x+y).sum(time)`
can be written directly. Broadcast alignment applies before the reduction.

Subsystem port expressions such as `child.x` still do not expose tensor methods
directly. A child-local equation can reduce its own Tensor parameter, or a parent
can expose the port through a Tensor alias:

```python exec="off"
from bilt import Tensor, equation


@equation
def expose(x: Tensor, child: Child) -> bool:
    return x == child.x


@equation
def aggregate(total: float, x: Tensor) -> bool:
    return total == x.sum(time)
```

Here, `child: Child` identifies a subsystem parameter, while the parent’s
`x: Tensor` provides an alias on which `.sum(time)` can be used.
`time` is defined earlier in the document.

Concrete tensor methods are
model-only: calling them directly on a returned `Tensor` raises `TypeError`.
Outside models, work with `t.values` for numerical operations.

## Errors

Import exceptions from `bilt.errors`.

| Exception | Meaning / remedy |
| --- | --- |
| `BiltTensorDomainError` | Missing domain, rank mismatch, absent reduction/selection dimension or non-scalar selector. Check the tensor domain and arguments. |
| `BiltInvalidDomainError` | Repeated dimension in a domain. |
| `BiltDomainBindingError` | Conflicting dimension sizes/coordinates, unbound dimension, missing coordinate axis or missing label. |
| `BiltDomainRegistryLockedError` | New dimension or coordinate binding attempted after dimension order was fixed. Establish bindings before calculation. |
| `BiltSymbolicLoweringError` | Invalid symbolic selection, such as a non-literal `sel` coordinate. |
| `BiltEquationDimensionalityError` | Incompatible dimensions/tags or an invalid unit-aware reduction. |
| `BiltUnitError` / `BiltDimensionalityError` | Missing unit or incompatible conversion. |
| `TypeError` / `AttributeError` | Concrete model-only operation, or tensor method called on a plain symbolic expression. |
| `NotImplementedError` | Concrete tensor arithmetic, or unsupported transform reduction form. Use direct parameter reductions in a single return expression. |
