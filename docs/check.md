## Check model definitions

`bilt check` finds Bilt models in a project and runs the checks Bilt already performs when systems, equations, transforms and implementations are constructed. It does not need numerical inputs or a configuration file.

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

### Run with prek

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

### Scope and limitations

Discovery is based on Python source, not a complete static analysis. It recognises direct Bilt imports and aliases, but can miss re-exported decorators, dynamically created models or model factories requiring execution. Add such modules explicitly when needed. A module imported successfully is **not a guarantee** that a factory inside it was executed.

Imports execute module-level Python code. Bilt checks definitions constructed by those imports, including dimensional consistency, declarations and subsystem wiring. It does not invoke transforms with dummy values, perform numerical evaluation, test structural solvability or infer tensor domains without inputs. Some checks remain runtime-dependent.

Files with Python syntax errors encountered during discovery are reported rather than silently ignored. No Bilt models found is a failure, not a successful check.

Exit status: `0` for success, `1` for model or source errors, `2` for invalid CLI usage or configuration.
