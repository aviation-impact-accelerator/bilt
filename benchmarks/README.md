# Structural search benchmarks

The engineering execution workloads at the end of this page are separate from
the structural-search counts below.

Run the routine selection:

```sh
uv run pytest benchmarks --benchmark-only --timeout=100
```

Run the extended matrix:

```sh
uv run pytest benchmarks --search-extended --benchmark-only --timeout=100
```

`--search-large` remains an alias for `--search-extended`. The default is a
small representative selection from every family; extended runs retain every
previously supplied case. Selection changes which cases run, not their models,
timed work, numerical checks or rejection checks. Both modes verify every
round, including `--benchmark-disable` correctness runs. Timeouts still fail.

| Suite | Routine cases | Extended cases |
| --- | ---: | ---: |
| Structural layouts | 30 | 72 |
| Failure recovery | 6 | 18 |
| Scaling and thermal networks | 8 | 25 |
| Difficult requests | 10 | 30 |
| **Total** | **54** | **145** |

Routine structural cases cover every layout, square/rectangular shape and
acyclic/coupled outcome at depth 1, plus depth-0 links as a primitive baseline.
The depth-2 rectangular coupled siblings case is explicitly retained for both
solvers: it caught the earlier approximately 78-second rejection regression.
Extended structural runs cover the full 72-case product of layouts, shapes,
outcomes, depths 0–2 and solvers.

The existing SCC-only stress endpoints (chain depth 32, tree depth 4 and
parallel depth 3) remain SCC-only in the extended suite. Other workloads are
paired across both solvers. No existing case is removed because its timing
resembles another. The largest exhaustive stress combinations are not newly
introduced by this reorganisation.

The benchmarks measure how much work each analyser needs to find a calculation
plan, or establish that no currently supported plan exists. Structural layout
fixtures use scalar inputs `a = 4`, `b = 2` and requested output `x`. They use
acausal equations; the analyser chooses their calculation directions and order.

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

Rejection checks follow explicit exception causes. SCC must report unsupported
coupled-block execution; an isolation, compilation or unexplained planning
failure is a benchmark failure. The exhaustive solver has no dedicated coupled
diagnostic, so it must report an unwrapped no-route error. Its result is recorded
as `no_acyclic_plan`, rather than claiming it diagnosed a coupled block.

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

`test_realisation_failure_recovery.py` runs six routine cases at depth 1. Extended runs cover
18 cases: two solvers, depths 0–2 and three
failure modes. Each timed trial serves two requests on the same initially fresh
analyser:

- **Control:** solve `z = 2*x` backwards, then forwards; both must succeed.
- **Isolation:** request `x` from `z` in `z = x*x`. The current isolator must
  refuse that direction; a subsequent request for `z` from `x` must return 25
  for `x=5`.
- **Backend:** use `z = 2*x` and deliberately refuse compilation of the inverse
  leaf function. Check exactly one refusal, then verify forward evaluation
  returns 10 for `x=5`. The refusal hook is restored after every trial.

These measure exhaustion, error propagation through child boundaries and survival
of another equation direction across requests. They **do not** demonstrate
finding an alternative route for the same request. Two independently supplied
producers can be a structural conflict under SCC validation; those would make an
invalid successful-retry fixture. Unit/tensor failures and successful same-request
route replacement need separate policy-valid fixtures before benchmarking.

The timing scope is the two requests together. Compare recovery cases with their
control at the same depth, rather than comparing directly with a single-request
structural-search measurement. Results use `recovery_scope=new_request_same_analyser`.

```sh
uv run pytest benchmarks/test_realisation_failure_recovery.py --benchmark-columns=median,min,max
```

Each structural-search case uses five pedantic rounds, with one analysis per round and no automatic
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

Benchmark helper regressions live in `tests/benchmarks/`. If `benchmarks` is in
pytest's `testpaths`, a normal `uv run pytest` also includes timing runs. Use
`--benchmark-disable` for the combined correctness run without repeated timing.

## Scaling and engineering workloads

`test_network_scaling.py` runs eight routine cases, with both solvers:

| Topology | Routine depth | Extended paired depths | SCC-only stress depth |
| --- | --- | --- | --- |
| Chain | 4 | 4, 8, 16 | 32 |
| Tree | 1 | 1, 2, 3 | 4 |
| Parallel heat exchangers | 1 | 0, 1, 2 | 3 |
| Series heat exchangers | 1 | 1, 2 | — |

The thermal leaf uses `heat = flow * capacity * (outlet - inlet)`. A parallel
wrapper splits heat and flow equally, propagates fluid heat capacity, feeds both
children from the common inlet, and mixes their equal-flow outlets. These are
acausal equations built through the normal modelling API. Values use consistent
SI units; the benchmark does not measure quantity conversion or tensor lowering.

Every round checks the independent network balance
`outlet = inlet + heat / (flow * capacity)`: the default inputs give 315 K.
Extra correctness cases cover zero heat and cooling. Timings exclude fixture
construction, execution, checks and the graph-size inspection. Reports include
actual expanded constraints, unknowns, subsystem instances and leaf counts.
The two solvers must each agree with the analytic reference.

A series wrapper preserves flow, allocates half the heat to each child,
connects the first outlet to the second inlet and exposes the final outlet.
The same whole-network energy balance applies.

Series depth 1 previously exposed multi-second SCC searches; producer-ordering
improvements address those on the updated solver branch. It is now a routine
regression case. Benchmark timings depend on the solver changes present in the
branch being measured. This benchmark-only change includes no solver fixes.

```sh
uv run pytest benchmarks/test_network_scaling.py --benchmark-columns=median,min,max
uv run pytest benchmarks/test_network_scaling.py --search-counters --benchmark-json=scaling.json
uv run pytest benchmarks/test_network_scaling.py --search-extended -k 'series and depth1' --search-rounds=1 --timeout=15
```

These small deterministic networks broaden coverage but do not establish
production-scale performance for arbitrary engineering models. Interpret scaling
by topology and graph size, rather than combining all cases into one score.


### Difficult request directions and network shapes

Run the additional families with:

```bash
uv run pytest benchmarks/test_heat_exchanger_requests.py --search-counters --benchmark-columns=median,min,max
```

Routine runs select one nested size per family, giving 10 cases.
`--search-extended` runs all 30 cases, preserving every earlier size. Both solvers
receive the same model, request and scalar input values. The timing lifecycle,
fresh contexts, per-round verification and separate counters match the other
search benchmarks. `--timeout=30` bounds each pytest case; timeouts fail.

| Family | Request and network | Expected behaviour | What it probes |
| --- | --- | --- | --- |
| `reverse` | Series exchangers; supply outlet, heat, flow and capacity; request inlet. | Success: 300 K. | Whether matching follows the supplied information backwards through nested children. |
| `heat` | Series exchangers; supply inlet, outlet, flow and capacity; request total heat. | Reject unsupported coupled execution. | A shared unknown connects both children and intermediate temperatures; the normal forward direction cannot resolve it. |
| `flow` | Series exchangers; supply inlet, outlet, heat and capacity; request shared flow. | Reject unsupported coupled execution. | The same coordination problem with flow appearing as a factor in every exchanger balance. |
| `mixed` | Alternate series and parallel layers; supply inlet, heat, flow and capacity; request outlet. | Success: 315 K. | Dependencies that run along series connections and reconverge at parallel mixing equations. |
| `unanchored` | Series exchangers; supply heat, flow and capacity, but neither temperature; request outlet. | Reject an underconstrained target. | Terminating an impossible search when temperature differences are determined but absolute temperatures are not. |

All cases use capacity 4,000 J/(kg K), flow 2 kg/s and total heat 120,000 W,
where supplied. The reference temperatures are inlet 300 K and outlet 315 K.
For success cases the reference comes from the whole-network energy balance,
`outlet - inlet = heat / (flow * capacity)`, independently of solver output.

`heat` and `flow` have unique physical answers (120,000 W and 2 kg/s), but a
unique answer does not imply an acyclic execution plan. In a two-exchanger
series network, the intermediate temperature and the shared unknown must be
solved together. These are rejection benchmarks until coupled solving is
implemented; they must eventually become numeric success benchmarks.

`unanchored` is different: adding an arbitrary constant to every temperature
preserves all equations, so no unique outlet temperature exists. SCC must
report an underconstrained problem, including through its normal error
wrapping. The exhaustive solver has only a generic no-acyclic-plan error;
that is a weaker diagnostic and is recorded as such. Unrelated isolation,
compilation or other analysis failures do not count as correct rejections.

Depth counts binary wrapper layers: depth 0 has one exchanger and depth `d`
has `2**d` exchanger instances. Routine sizes are depth 1 for reverse, heat, flow and unanchored, and
depth 2 for mixed. Extended sizes are depths 1–3 for reverse, heat and flow;
2–4 for mixed; and 0–2 for unanchored. Mixed depth 2 genuinely contains both series
and parallel connections; it is not merely a renamed parallel network.

These cases measure search and rejection with scalar, consistent SI values.
They do not cover unit conversion, tensor shape failures, nonlinear block
execution or successful same-request recovery after a refused realisation.

## Engineering reporting sessions and intermediate caching

`test_intermediate_workloads.py` compares complete reporting sessions using
three existing repository model families:

| Family | Existing model | Reporting order |
| --- | --- | --- |
| Geometry | Five functions from `examples.hx.geometry`; the two pi constants are inlined because imported globals are not resolved by the transpiler. | Flow area, inner diameter, outer area per element, inner radius. |
| Series | `build_thermal_model("series", 1)`: two exchangers in series. | Temperature rise, outlet temperature, heat per unit flow. |
| Mixed | `build_mixed_model(2)`: four exchangers with series and parallel connections. | Temperature rise, outlet temperature, heat per unit flow. |

The thermal reporting adapter adds boundary connections and two directed report
transforms; it does not change the exchanger equations. References are computed
independently from geometry and whole-network energy balance. These are small
engineering examples, not evidence for production-scale models.

Each family runs with scalar inputs and a batch of 10,000 design/operating
points, both Python/JAX backends and both structural analysers: 48 cases in all.

- **Cold:** each round uses a new implementation and instance. Timing includes
  planning, realization, backend compilation, execution, and lifting. Model
  construction and input creation are outside the timer. The interpreter is
  shared, so this is not fresh-process startup timing.
- **Warm:** the complete report is compiled and checked before timing. Each
  measured round still creates a fresh instance, so its values are not cached.
- Timing covers the complete report. `extra_info.first_output_ns` and
  `subsequent_outputs_ns` split the first request from the remaining requests.
  Scalar JAX values are explicitly synchronized; tensor lifting materializes
  NumPy values before the timer stops.
- Every output of every round is checked outside timing, including with
  `--benchmark-disable`. Old sessions are released as rounds advance.
- Retained tensor bytes exclude input buffers and count aliasing views once.
  Intermediate bytes also exclude requested-output buffers. This measures
  storage reachable after the report, not peak process or device memory.

For a before/after comparison, run the **same benchmark source and dependency
environment** against both revisions. Use 64-bit JAX to match NumPy precision,
fix the hash seed, and alternate revision order across independent processes.
For example, from the feature checkout after installing dependencies:

```sh
git worktree add --detach /tmp/bilt-cache-base 9fa701b
PYTHONPATH=/tmp/bilt-cache-base/src JAX_ENABLE_X64=1 PYTHONHASHSEED=0 \
  .venv/bin/pytest benchmarks/test_intermediate_workloads.py --timeout=30 \
  --execution-cold-rounds=5 --execution-warm-rounds=60 \
  --benchmark-json=/tmp/base.json
PYTHONPATH=src JAX_ENABLE_X64=1 PYTHONHASHSEED=0 \
  .venv/bin/pytest benchmarks/test_intermediate_workloads.py --timeout=30 \
  --execution-cold-rounds=5 --execution-warm-rounds=60 \
  --benchmark-json=/tmp/feature.json
```

Repeat with reversed order rather than treating one run as a stable result.
Compare report totals and first-request costs separately: cheaper subsequent
requests need not compensate for exposing and lifting more intermediates on
the first request. Do not combine these timings with the search-only suites.

### Reference comparison, 28 October 2026

Compared base `9fa701b` with the intermediate-caching implementation on Linux,
AMD EPYC 9V74, CPython 3.15.0rc3, using the same dependency environment and
64-bit JAX. Five independent process pairs alternated base/feature order.
Each case used five cold rounds or 60 warm rounds: 15,600 checked report
sessions across the two revisions. Values below are medians of the five
per-process medians, using the default exhaustive analyser.

**Complete warm reporting sessions, milliseconds (base → caching):**

| Workload | Python | JAX |
| --- | ---: | ---: |
| Geometry, scalar | 0.172 → 0.148 | 0.339 → 0.237 |
| Geometry, 10,000 points | 0.268 → 0.211 | 0.620 → 0.523 |
| Series, scalar | 0.139 → 0.159 | 0.222 → 0.233 |
| Series, 10,000 points | 0.265 → 0.239 | 0.495 → 0.453 |
| Mixed, scalar | 0.176 → 0.179 | 0.206 → 0.221 |
| Mixed, 10,000 points | 0.395 → 0.288 | 0.500 → 0.434 |

There was substantial process-to-process variation. Geometry/JAX at 10,000
points and mixed/Python at 10,000 points improved
in every pair: paired speedup ranges were 1.11–1.25 and 1.11–1.56 respectively.
Scalar geometry/Python also improved in every pair, with a range of 1.01–1.42.
Every other warm comparison reversed direction in at least one pair. Do not treat the other median
improvements as established wins. Warm first-output costs generally rose; for
example, the mixed batch's first JAX output rose from 0.172 to 0.235 ms even
though its complete report median fell.

**Complete cold JAX reporting sessions, milliseconds (base → caching):**

| Workload | Time |
| --- | ---: |
| Geometry, scalar | 130.987 → 107.422 |
| Geometry, 10,000 points | 148.389 → 118.373 |
| Series, scalar | 135.993 → 91.977 |
| Series, 10,000 points | 147.384 → 99.962 |
| Mixed, scalar | 140.525 → 104.674 |
| Mixed, 10,000 points | 163.435 → 110.349 |

These cold JAX sessions improved in every process pair, with median-of-median
time reductions of approximately 18–32%. First-output times mostly rose; the
saving comes from serving later report requests with fewer compilation steps.
A separate untimed check counted backend `transpile` calls for the batched
reports: geometry fell from 11 to 8, series from 29 to 23, and mixed from 39 to
33. These are generated backend functions, not a count of XLA compilation events.
Cold Python results were less consistent: geometry improved in every default-
analyser pair, while the scalar thermal cases had no reliable improvement.
The causalization analyser also showed consistent cold JAX improvements across
all six workloads, but mixed warm results.

At 10,000 points, retained geometry tensor storage stayed at 320,000 bytes per
instance. Thermal reports stayed at 240,000 bytes in Python, but increased from
240,000 to 400,000 bytes in JAX. Scalar/device bookkeeping and peak memory are
not included. The result supports cold multi-output reporting and some batched
workflows, rather than a general warm-execution speedup.
