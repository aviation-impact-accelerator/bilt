#!/usr/bin/env bash
set -euo pipefail

# Run the checks used by Bilt's pull-request CI from any working directory.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

if [[ ! -f pyproject.toml || ! -f prek.toml ]]; then
  printf 'Expected a Bilt checkout at %s\n' "$repo_root" >&2
  exit 1
fi

run() {
  printf '\n> '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

# Match CI's locked environment and point PyO3 at uv's Python.
export UV_LOCKED=1
run uv sync --all-extras --all-groups
export PYO3_PYTHON
PYO3_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"

run uv run ruff check
run uv run ruff format --check
run uv run pyrefly check
run uv run ty check
run uv run pytest --durations=100 --durations-min=0.01

# Exercise the structural-planning regression checks used in CI.
run uv run --with pytest-benchmark==5.3.1 pytest benchmarks/test_nested_coupled_plan_search.py --benchmark-disable --search-rounds=1 --timeout=30
run uv run --with pytest-benchmark==5.3.1 pytest benchmarks/test_coupled_group_planning.py --benchmark-disable --search-rounds=1 --timeout=30

run cargo fmt --all -- --check
run cargo clippy --all-targets --all-features -- -D warnings
run cargo test --all-targets --all-features
run uv run zensical build --clean --strict
run uv run prek run --all-files

printf '\nAll Bilt development checks passed.\n'
