---
name: bilt-development
description: Develop, debug, test and review changes to the Bilt framework itself. Use when modifying its Python or Rust implementation, public API, equation parsing, structural analysis, coupled solvers, units, tensors, JAX lowering, documentation, CI, or when reviewing pull requests to Bilt. Not for ordinary engineering modelling with Bilt.
license: MIT
---

# Develop Bilt

Use this skill for **contributions to Bilt**. When building models in a separate project, use the `bilt` skill instead.

Bilt is an equation-oriented scientific modelling framework. Its interfaces and relationships describe a model; structural analysis decides how to calculate requested values; the selected backend executes the plan. Maintain that separation.

## Find the right source of truth

- `src/bilt/`: public API, Python implementation, model machinery and backends.
- `rust/`: Rust implementation and supporting internals.
- `tests/`: API, analyser, numerical, error and integration coverage.
- `benchmarks/`: performance checks, especially structural-search edge cases.
- `docs/guide/`, `docs/api/` and `docs/worked_examples/`: supported concepts and public usage.
- `pyproject.toml`, `prek.toml` and `.github/workflows/check.yml`: supported development commands and CI.

Read the relevant implementation and existing tests before proposing an API or refactor. Treat source code and passing tests as more authoritative than roadmap prose.

## Design principles

- Keep `@equation` declarative. Its AST subset, validation, symbolic lowering and JAX translation must agree. Do not treat equations as arbitrary eager Python. Prefer `@equation` for invertible relationships and `@transform` for directed escape hatches.
- Preserve model-level analysis across parents and children. Validate constraints together, distinguish missing and overconstrained systems, and avoid pathological search as coupled blocks grow. Coupling can involve more than two unknowns.
- Preserve the distinction between scalar `Quantity`, named-tensor `QuantityTensor`, units and dimensions. Check broadcasting, reductions and unit conversion at the public boundary.
- Keep Python and JAX backends consistent where features are supported; JAX expressions and transforms must respect tracing and purity requirements.
- Keep public APIs minimal and avoid introducing an alternative code path that a later PR removes. Error reports should use Bilt's established error mechanisms where appropriate.
- Maintain meaningful performance baselines when changing structural planning; a faster special case must not regress ordinary models.

## Development workflow

From a checkout of Bilt, use the bundled script for a repeatable full check:

```bash
bash .agents/skills/bilt-development/scripts/check.sh
```

The script installs the locked environment, then runs Ruff lint and format checks, Pyrefly, ty, pytest, the two coupled-planning regression checks, Cargo format/lint/tests, the Zensical documentation build and prek. It stops at the first failure. This mirrors the checks in `.github/workflows/check.yml`; keep the script in sync if CI changes. CI validates the script's shell syntax but runs its constituent checks separately, so it does not duplicate the full suite.

For quicker iteration, run the relevant checks individually, for example:

```bash
uv run pytest tests/
uv run ruff check
uv run ruff format --check
uv run pyrefly check
uv run ty check
uv run prek run --all-files
```

Only claim checks have passed when they have actually executed successfully.

When touching model behaviour, add focused regression tests for successful calculations **and** meaningful failure cases. For planner changes, compare relevant benchmarks, including nested and coupled models. When changing a public API, update examples, reference docs and tests in the same change.

## Pull request review

For GitHub Copilot code review or a human PR review:

1. Identify the behaviour and public contract being changed.
2. Look for correctness regressions in equations, solution planning, unit/domain handling, backend consistency and state/caching.
3. Check important negative cases and whether a claimed improvement has appropriate tests or benchmarks.
4. Point to concrete changed lines and explain failure scenarios. Focus on actionable issues, not speculative style advice; avoid asking for checks already covered by Ruff, Pyrefly or ty.
5. Verify that docs and the PR description reflect the actual implementation. Do not assume tests passed without CI evidence.

This skill supplies detailed review context; the short `.github/copilot-instructions.md` holds review guidance that should apply to every pull request.
