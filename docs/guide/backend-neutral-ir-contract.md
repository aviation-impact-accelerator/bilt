# Backend-neutral IR contract (internal)

This document records the **production contract** introduced by #491's first implementation step. The IR is an internal compiler representation, not a public modelling API. This PR **does not change realisation, Python lowering, JAX lowering or structural search**.

## Two independent levels

Acausal `symbolic.Tree` is still the model's mathematical relationship. Structural search chooses an output, isolates relationships, and resolves tensor domains before generating executable code.

The intended future boundary is:

```text
Symbolic Tree -> selected computation -> verified ProgramIR -> backend lowering
                                             |
                                  typed functions and regions
```

`FunctionIR` has named typed inputs, operations, and a single yielded value. Each operation defines one SSA result; nested `scf.if` regions have lexical scope and are **lazy**. `tensor.where` is an elementwise operation, not scalar control flow.

`ProgramIR` contains a sequence of typed function evaluations, qualified child calls, linear solves and projections. Each step explicitly declares its dependencies. It contains neither Python AST nor backend callables.

## Scientific type information

A `ValueType` distinguishes:

- element kind (float, integer, boolean or opaque)
- named tensor `Domain` (distinct from numerical array shape)
- optional physical `Dimension` (including its tagged physical dimensions)
- optional scientific `Metadata`

An absent physical dimension represents **unknown**, not dimensionless. `Domain.scalar()` represents a scalar value. The contract retains named dimensions but does not yet infer or validate coordinate array lengths.

The initial validator is deliberately conservative. It checks SSA definitions/dominance, operation arity and attributes, scalar conditional predicates, elementwise mask domains, matching branches, positional tensor axes, ordered program inputs, result types and explicit projection ownership. Only known operation semantics are accepted. It is not a complete physical dimensional algebra or broadcasting system: operations whose dimension result cannot be derived safely from the available information are not silently inferred.

## Examples

An isolated arithmetic equation may become:

```text
func @sum(%a: float, %b: float) -> float {
    %sum = arith.add %a, %b
    return %sum
}
```

A scalar conditional has two independent regions:

```text
%result = scf.if %predicate {
    %positive = arith.neg %x
    yield %positive
} else {
    yield %x
}
```

An execution schedule is a sequence of explicitly bound steps:

```text
%solution = linear.solve(%known) unknowns=(%x, %y)
%x = linear.project(%solution, x)
%answer = call @compute(%x)
return %answer
```

These are illustrative, **not** a parser or a supported text format. Golden tests assert the actual Python data model rather than depending on the rendering of a future printer.

## Responsibilities and planned changes

**PR 1:** Contract, immutable nodes, verifier and tests. No conversion to IR and no change in execution time.

**PR 2:** Translate isolated equations and a conservative subset of pure transforms into typed FunctionIR; keep unsupported Python explicit as a legacy fallback. Python and JAX adapters still use existing numerics.

**PR 3:** Lower coupled equation systems into explicit, self-contained solve components. Typed coefficient/rhs output shapes and layouts need to be verified here, rather than assuming scalar FunctionIRs are adequate.

**PR 4:** Eliminate source-plan references during program compilation, including child routing, qualified names and failure/retry attribution.

**PR 5:** Remove redundant conversion layers, add inspection tools and benchmark full cold evaluation and JAX compilation. Do not introduce caching without a demonstrated hit rate.

No external MLIR or xDSL dependency is needed for this contract. The contract should be reviewed before migrating compilers.
