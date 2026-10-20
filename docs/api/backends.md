# Backends API

Bilt provides two built-in execution backends. Import them from `bilt`;
see the [backend guide](../guide/backends.md) for a walkthrough.

| Backend | Execution | Installation |
| --- | --- | --- |
| `PYTHON_BACKEND` | Python callables and NumPy tensor operations; default for instance evaluation. | Standard Bilt installation. |
| `JAX_BACKEND` | JAX operations with `jax.jit` compilation of transforms and the calculation chain. | Optional `jax` extra. |

```console
uv add "bilt[jax]"
```

For an existing `model` implementation and an instance `sample`:

```python
from bilt import JAX_BACKEND

with JAX_BACKEND, model:
    result = sample.output
```

| API | Meaning |
| --- | --- |
| `with backend:` | Select a backend for instance evaluation in the block; the innermost backend context wins. |
| `implementation.evaluate(inputs, output, backend)` | Evaluate directly with an explicit backend. |

Use a fresh instance for each backend comparison: values already calculated on an
instance are cached and are returned without recalculation. The first JAX call
may include compilation; later calls can reuse compiled work.

## Supported operations

Both backends implement named tensor alignment, reductions and positional
selection. Coordinate selection resolves to a position before backend execution.
See [Tensors](tensors.md) for their API restrictions.

| Python transforms | JAX transforms |
| --- | --- |
| Ordinary Python computations and imports, subject to Bilt's unit/domain handling. | Straight-line numerical expressions, assignments, calls, array indexing and list/tuple construction accepted by Bilt's AST validator. |
| NumPy intrinsics for Bilt tensor operations. | JAX intrinsics; list literals become JAX arrays. |

JAX compilation supplies `jnp` as `jax.numpy`. Accepted syntax must also be
compatible with JAX tracing: AST acceptance alone does not guarantee execution.
Python `if` statements, loops and imports are rejected by the current JAX validator.
For example:

```python
import jax.numpy as jnp
from bilt import transform


@transform
def root(value: float) -> float:
    return jnp.sqrt(value)
```

## Errors

Import Bilt exceptions from `bilt.errors`.

| Exception | Meaning |
| --- | --- |
| `ImportError` | Importing `JAX_BACKEND` requires the optional JAX dependency. |
| `BiltBackendError` | Base backend error. |
| `BiltUnsupportedNodeError` | The backend rejects an AST node. |
| `BiltCompilationError` | Compiling a generated function failed. |
| `BiltAnalysisError` | No route could be realised; may report a backend rejection as its cause. |

Runtime Python, NumPy and JAX exceptions can also propagate.
