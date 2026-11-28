# Check model definitions

`bilt check` finds Bilt models in a project and runs the checks Bilt already performs when systems, equations, transforms and implementations are constructed. It also checks source for misuse of Bilt's internal API. It does not need numerical inputs or a configuration file.

Run from the model project:

```shell
uv run bilt check
uv run bilt check .
uv run bilt check src/
uv run bilt check myproject/
```

By default it scans the selected directory for Python files that directly use Bilt's public decorators (`@system`, `@equation`, `@transform`) or `Implementation(...)`, including aliases imported from `bilt`. It imports **only candidate modules**, not every file in the project. Both ordinary package layouts and `src/` layouts work. Directories such as `tests`, `benchmarks`, `scripts`, build directories and virtual environments are skipped by default.

You can optionally add modules that cannot be discovered by inspection, or exclude files, using the consumer project's `pyproject.toml`:

```toml
[tool.bilt.check]
exclude = ["myproject/generated/**"]
modules = ["myproject.dynamic_models"]
# recursive = true  # default for explicitly listed packages
```

The optional `modules` list **adds to automatic discovery**. Repeated `--module` arguments override that list, but still include automatically discovered models. Use `--config` to select a different `pyproject.toml`.

## Public API checks

The checker inspects all selected Python source files, including files that use models without defining them.

| Code | Meaning |
| --- | --- |
| BILT101 | Private Bilt import or namespace |
| BILT102 | Private attribute on a recognised Bilt object |
| BILT103 | Reserved model metadata |
| BILT104 | Internal implementation method |
| BILT105 | Internal backend method |
| BILT106 | Obsolete quantity, tensor or number constructor |

Use `2.0 * meter` rather than `Quantity(2.0, meter)`, and `tensor(value)` rather than `Tensor.scalar(value)`. The public quantity and tensor protocols are annotations, not constructors. Legacy `Quantity | QuantityTensor` annotations remain supported.

Evaluate models through declared system fields within implementation and backend contexts. If a feature is not available publicly, do not bypass the system interface.

The source checks recognise direct imports, aliases, simple assignments, annotations and straightforward imports between selected files. They are not a complete Python type checker: dynamic dispatch, arbitrary factories and complex re-exports may escape detection. Ordinary methods on unrelated objects are not rejected merely because their names match internal Bilt methods.

## Migrate the old quantity API

```shell
uv run bilt migrate .
uv run bilt migrate . --write
# Optional: format migrated files with Ruff installed in the project environment
uv run bilt migrate . --write --format
uv run bilt check .
```

Both commands use the same project source selection, default directory exclusions and `[tool.bilt.check].exclude` patterns. Use `--config` with either command to select an alternative configuration. Explicitly target a Python file or a normally skipped directory to migrate it. Symlinked files are not modified.

The first command previews a unified diff without writing files. The second applies safe source edits. Use `--format` on the initial preview or write to format changed files with Ruff, if installed. The migration validates AST equivalence after formatting; it does not format unchanged files. Explicitly approved obsolete quantity and tensor imports are removed when their last apparent reference disappears, except for ambiguous or protected imports.
BILT106 and the constructor migration rules share the same lexical reference analysis and obsolete API definitions. Unsupported constructor calls, including `Number(...)`, require manual review rather than guessing the intended value. The migration uses the shared source migration framework with the quantity protocol rule set. It does not import your models. It handles positional and ordered keyword quantity constructors, direct tensor constructors, scalar tensor factories, pointwise unions (including multiline unions without comments), and tensor annotations with directly imported physical dimensions. Required public imports are added automatically, using a fresh alias if necessary.
The old Tensor class included dimensioned tensors; the new Tensor protocol is unitless. For unchanged parameters explicitly annotated with a union of Tensor and Quantity or QuantityTensor, the migration widens builtin isinstance checks to `(Tensor, QuantityTensor)`. Other runtime tensor checks remain manual review items: a bare Tensor annotation or an untyped value does not establish the intended classification. Ambiguous expressions, reordered quantity keywords and commented expressions also remain for review.
Ordinary functions and async functions can be migrated using lexical scope information from Python's symbol table. Stable, unconditional imports are recognised; shadowed names, mutable globals, closure bindings and conditional imports are not assumed to refer to Bilt. Uncommented multiline calls, nested constructor calls, ordinary comprehension bodies and lambdas with stable global imports are migrated. Comprehension targets and lambda parameters cannot accidentally resolve to shadowed imports. Class-suite comprehensions, ambiguous closure aliases, comments inside calls and conservatively retained imports may still need manual cleanup. Review the diff and run the model tests afterwards.
Exit status 1 means that manual review or a source error remains; 2 means the target is invalid.

## Run with prek

Add a local hook to your project's `prek.toml`:

```toml
[[repos]]
repo = "local"

[[repos.hooks]]
id = "bilt-check"
name = "Check Bilt models"
entry = "uv run bilt check ."
language = "system"
pass_filenames = false
always_run = true
```

The hook uses the project's environment, where Bilt and the model's dependencies are installed. Run `prek run bilt-check --all-files` to try it.

## Scope and limitations

Discovery is based on Python source, not a complete static analysis. It recognises direct Bilt imports and aliases, but can miss re-exported decorators, dynamically created models or model factories requiring execution. Add such modules explicitly when needed. A module imported successfully is **not a guarantee** that a factory inside it was executed.

Imports execute module-level Python code. For recognised top-level Bilt definitions, a failed definition does not prevent independent later definitions from being checked. Definitions depending on a failed one are reported as skipped. The recovery mechanism is intentionally limited to recognised top-level declarations; nested Python code and unrelated Python errors retain normal behaviour. Bilt checks definitions constructed by imports, including dimensional consistency, declarations and subsystem wiring. It does not invoke transforms with dummy values, perform numerical evaluation, test structural solvability or infer tensor domains without inputs. Some checks remain runtime-dependent.

Files with Python syntax errors encountered during discovery are reported rather than silently ignored. No Bilt models found is a failure, not a successful check.

Exit status: `0` for success, `1` for model or source errors, `2` for invalid CLI usage or configuration.
