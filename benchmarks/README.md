# Structural search benchmarks

```sh
uv run pytest benchmarks --benchmark-only --timeout=100
```

or for a more concise summary
```sh
uv run pytest benchmarks -q \
  --benchmark-columns=median,min,max \
  --benchmark-sort=name
```

This replaces `uv run benchmarks/structural_search.py`. The benchmarks measure
how much work each analyser needs to find a calculation plan, or establish that
no currently supported plan exists. They use small scalar models with inputs
`a = 4`, `b = 2` and requested output `x`.

Each case combines a **layout**, a **nesting depth**, **acyclic or coupled leaf
equations**, and a **square or rectangular structure**. Running both solvers over
all combinations gives 72 benchmark results. All fixture relations are acausal
`@equation` definitions: the analyser chooses their calculation directions and
execution order.

## What the layouts represent

A **parent** contains a child subsystem and equations connecting its ports to
parent values. The layouts change those parent equations. The examples below
show one parent layer; `inner`, `left` and `right` name child instances.

### Links: connect interfaces without changing their values

```text
inner.a = a
inner.b = b
x = inner.x
```

The parent forwards its inputs to one child and exposes the child's output.
This isolates the cost of crossing subsystem boundaries and following a chain
of nested definitions. With an acyclic child, `x` remains 6 at every depth.

### Arithmetic: calculate values on both sides of a child

```text
inner.a = 2*a + b
inner.b = b + 1
x = inner.x + inner.b
```

The parent calculates the child's inputs, then combines its result with another
boundary value. This exercises dependency ordering across parent and child
calculations. At depth 1, the child receives 10 and 3, produces 13, and the parent
returns `x = 16`.

### Siblings: connect two children through the parent

```text
left.a = a + b
left.b = b
right.a = left.x + a
right.b = b + 1
x = left.x + 2*right.x
```

The right child needs the left child's output before it can calculate its own.
The parent then combines both results. Both children use the same implementation
object, but represent distinct instances with separate values and constraints.
This exercises branching, sibling dependencies and reuse of a child definition.

At depth 1, the left child produces 8, the right child receives 12 and 3 and
produces 15, and the parent returns `x = 38`.

## What depth means

- **Depth 0:** the leaf equations alone, with no child subsystems. All three
  layouts have the same leaf model at this depth.
- **Depth 1:** one parent layer around the leaf model. Links and arithmetic have
  one child; siblings has two.
- **Depth 2:** two parent layers. Links and arithmetic form a chain ending in
  one leaf; siblings forms a tree with four leaves.

For acyclic cases, the expected outputs are calculated independently of the
analysers:

| Layout | Depth 0 | Depth 1 | Depth 2 |
| --- | ---: | ---: | ---: |
| Links | 6 | 6 | 6 |
| Arithmetic | 6 | 16 | 34 |
| Siblings | 6 | 38 | 620 |

## Acyclic and coupled cases

**Acyclic** leaves contain:

```text
x = a + b
y = x + 1
```

These can be ordered as individual calculations. Both solvers must find a plan
and produce the independently calculated `x`; each measured round checks that
numeric result outside the timer.

**Coupled** leaves contain:

```text
x + y = a
x - y = b
```

Neither equation can calculate its output from `a` and `b` alone without also
knowing the other unknown. Together they form a block requiring a simultaneous
solve. These equations have a unique mathematical solution, but coupled-block
execution is not supported yet, so the expected outcome is rejection of the
execution plan. This measures how quickly each analyser establishes that a
supported acyclic route does not exist.

## Square and rectangular cases

These labels describe the expanded primitive constraint graph after fixing the
supplied inputs.

- **Square:** the base fixture has equally many equations and unknowns.
- **Rectangular:** add `u + v = x` at every leaf. Each added equation introduces
  two new unknowns, so the graph has more unknowns than equations.

The additional variables are not needed to calculate `x`. Once `x` is known,
there are still many possible pairs of `u` and `v`. The rectangular fixtures
therefore check that surplus underdetermined branches do not invalidate a
determined target or force the analyser to calculate every variable.

For example, the depth-2 sibling model has 23 equations and 23 unknowns in square
form, or 27 equations and 31 unknowns in rectangular form. Its target remains in
the well-determined region in both forms. Equal equation and unknown counts
alone are not a general test of whether a model is solvable.

Expected outcomes are the same in both shapes: acyclic cases succeed and coupled
cases reject unsupported execution. An underconstraint or overconstraint error
is not accepted as the expected coupled rejection for these fixtures.

## Timing

Each case uses five pedantic rounds, with one analysis per round and no automatic
calibration or warmup. Every round builds a fresh model and enters a fresh
`EvaluationContext` before timing. The timer includes analyser construction,
analysis and realisation. Execution, correctness checks, structural-shape checks
and cleanup are outside it. Every round is checked, including when timing is
disabled with `--benchmark-disable`.

The interpreter is shared between rounds. This measures cold planning state, not
a fresh process; establish a new baseline rather than directly comparing these
numbers with the old subprocess runner. Framework timing and wrapper overhead
also differ. Avoid concurrent benchmark runs when comparing results.

`--timeout=100` uses pytest-timeout to bound each entire parameterised test,
including setup, all rounds, checks and optional counters. It is not the old
per-worker limit. pytest-benchmark's `--benchmark-max-time` does not interrupt a
single slow analysis. Timeout failures are never accepted as coupled rejection.
The timeout plugin may terminate the process on platforms without signal-based
interruption; the runner no longer provides subprocess isolation.

## Select, count and compare

```sh
uv run pytest benchmarks --benchmark-only --timeout=100 -k 'siblings and rectangular'
uv run pytest benchmarks --benchmark-only --timeout=100 --search-rounds=10
uv run pytest benchmarks --benchmark-only --timeout=100 --search-counters --benchmark-json=benchmark-results.json
```

`--search-counters` runs one additional fresh, untimed trial per case. Its counts
are stored under `extra_info.instrumented` in pytest-benchmark's JSON. Timed rounds
are never instrumented. Parameters, checked result, verified-round count and hash
seed are recorded alongside the timings. Machine and commit information are
provided by pytest-benchmark.

For repeatable comparisons, set the hash seed before starting Python and use the
same machine and options:

```sh
PYTHONHASHSEED=0 uv run pytest benchmarks --benchmark-only --timeout=100 --benchmark-autosave
PYTHONHASHSEED=0 uv run pytest benchmarks --benchmark-only --timeout=100 --benchmark-compare
```

Saved baselines live in the ignored `.benchmarks/` directory. For cProfile, use
`--benchmark-cprofile=cumtime --benchmark-cprofile-loops=1` to preserve fresh trial state.

Benchmarks are opt-in: normal `uv run pytest` discovers the existing `tests/` and
`packages/` suites. Benchmark helper regressions live in `tests/benchmarks/`.
