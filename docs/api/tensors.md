# Tensor API

Import tensor helpers from `bilt`. See [Domains and tensors](../guide/dimensions-domains-and-tensors.md)
for an introduction and [Units API](units.md) for physical units.

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

```python
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

Use `Annotated[Tensor, LENGTH]`, for example, for a unit-aware tensor port.
Bare `Tensor` ports are outside the physical unit system. Computed declared
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

```python
from bilt import Tensor, transform

@transform
def averaged(values: Tensor) -> Tensor:
    return values.mean(time) * 2.0
```

## Indexing

Use selection in **equations**; `.isel()` and `.sel()` are not lowered in transforms.
Both select one element along a dimension and remove that dimension from the result.

| Method | Selector |
| --- | --- |
| `x.isel(dim, index)` | Zero-based integer position or symbolic scalar integer expression. Valid negative positions count from the end. |
| `x.sel(dim, coordinate)` | Literal coordinate, including a captured Python constant, on a bound coordinate axis. |

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

```python
from bilt import coordinate_axis
from bilt.core._evaluation_context import EvaluationContext

with EvaluationContext() as evaluation:
    evaluation.domains.bind(coordinate_axis(time, (2024, 2025, 2026)))
    # Request a model output using values.sel(time, 2025).
```

### Intermediate values

Selection works on **named intermediate ports over known dimensions**.
Dimensions, extents and coordinate axes must be established before calculation;
Bilt cannot infer new dimensions or changed extents on the fly.

Arithmetic and reduction/selection results are plain symbolic expressions
without tensor methods. To apply another reduction or selection, expose the
intermediate as a Tensor parameter in a separate relationship:

```python
from bilt import Tensor, equation

@equation
def doubling(doubled: Tensor, values: Tensor) -> bool:
    return doubled == values * 2.0

@equation
def selecting(selected: Tensor, doubled: Tensor) -> bool:
    return selected == doubled.isel(time, 1)
```

The same pattern supports repeated reductions. Concrete tensor methods are
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
