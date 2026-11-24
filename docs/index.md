# Bilt

Bilt is a Python framework for scientific and engineering modelling. Define systems and relationships, supply known values, and request the result you need. Bilt finds a supported calculation, including across reusable subsystems.

## Start here

**[Build your first Bilt model](getting-started/first-model.md)** — set up a Python project, define a system and an equation, then answer two different questions with the same model.

If you already have experience with equation-oriented modelling, you can jump to the [user guide](#user-guide), [worked examples](#worked-examples) or [API reference](#api-reference).

## User guide

The guide introduces Bilt's ideas and shows how to use its modelling features:

- [Modelling with Bilt](guide/modelling-philosophy.md) — why relationships are kept separate from computation.
- [Systems and values](guide/systems-and-values.md) — declare model interfaces.
- [Equations and acausal modelling](guide/equations-and-acausal-modelling.md) — express relationships that can answer different questions.
- [Transforms and implementations](guide/transforms-and-implementations.md) — use directed calculations and choose implementations.
- [Models to computation](guide/models-to-computation.md) — understand how Bilt chooses a calculation.
- [Dimensions, domains and tensors](guide/dimensions-domains-and-tensors.md) — work with named axes and broadcasting.
- [Systems of systems](guide/systems-of-systems.md) — compose reusable models.
- [Coupled systems](guide/coupled-systems.md) — solve supported simultaneous linear equations.
- [Backends](guide/backends.md) — use Python and JAX.
- [Units](guide/units.md) — use physical quantities and dimensional analysis.

## Worked examples

These examples show how Bilt fits into complete engineering models:

- [Two springs in series](worked_examples/two-springs-in-series.md) — a small model with reusable subsystems and a coupled solve.
- [Two-shaft gas turbine](worked_examples/two-shaft-gas-turbine.md) — a larger interacting system with forward and inverse questions.
- [Battery-electric mission uncertainty analysis](worked_examples/battery-electric-mission-model-with-quasi-monte-carlo-broadcasting.md) — named tensors and a large scenario sweep.
- [Compiling a Bilt model with JAX](worked_examples/compiling-a-bilt-model-with-JAX.md) — Python and JAX execution on a larger calculation.

## API reference

Look up precise declarations, supported operations and limitations:

- [Systems and implementations](api/systems.md)
- [Equations and transforms](api/relationships.md)
- [Units](api/units.md)
- [Quantity annotations and typing](api/quantity-annotations.md)
- [Tensors](api/tensors.md)
- [Backends](api/backends.md)

## Tools and configuration

- [Check model definitions](check.md) — use `bilt check` in your project or CI.
- [Ruff configuration](config.md#ruff) — keep imports needed by Bilt's runtime annotations.
