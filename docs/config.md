## Configuration

### Ruff

Bilt reads the annotations of `@system` classes and of `@equation` and `@transform` functions when the decorator runs, so every name those annotations use must be available at runtime. Ruff's type-checking rules (`TC001`–`TC003`) don't know this, and may ask you to move an import that is only used in those annotations into an `if TYPE_CHECKING:` block. Doing so makes the decorator raise `BiltUnresolvedAnnotationError`.

Tell ruff which decorators read annotations at runtime by adding this to your `pyproject.toml`:

```toml
[tool.ruff.lint.flake8-type-checking]
runtime-evaluated-decorators = [
    "bilt.equation",
    "bilt.transform",
    "bilt.system",
]
```
