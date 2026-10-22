# Bilt

Bilt is a Python-native framework for equation-oriented modelling.

Models are built from systems, equations and transforms, with Bilt determining how available values can be used to calculate requested outputs. Models can be composed from reusable subsystems and executed using different computational backends.

## User guide

The user guide introduces the main Bilt concepts:

- [Modelling philosophy](guide/modelling-philosophy.md)
- [Systems and values](guide/systems-and-values.md)
- [Equations and acausal modelling](guide/equations-and-acausal-modelling.md)
- [Transforms and implementations](guide/transforms-and-implementations.md)
- [Models to computation](guide/models-to-computation.md)
- [Dimensions, domains and tensors](guide/dimensions-domains-and-tensors.md)
- [Systems of systems](guide/systems-of-systems.md)
- [Backends](guide/backends.md)
- [Units](guide/units.md)

## Worked examples

Worked examples show how these concepts fit together in complete models.

### Systems of systems

- [Two springs in series](worked_examples/two-springs-in-series.md)
- [Two shaft gas turbine](worked_examples/two-shaft-gas-turbine.md)

### Dimensions, Tensors and broadcasting

- [Battery electric mission model with quasi-monte carlo broadcasting](worked_examples/battery-electric-mission-model-with-quasi-monte-carlo-broadcasting.md) - a simple example of broadcasting in a model with a large number of variables.

### Choosing computational backends

- [Compiling a Bilt model with JAX](worked_examples/compiling-a-bilt-model-with-JAX.md) - a detailed example of how to compile a Bilt model with JAX.

## API reference

- [Systems and implementations](api/systems.md)
- [Equations and transforms](api/relationships.md)
- [Backends](api/backends.md)

- [Tensors](api/tensors.md) - tensor construction, domains, broadcasting, reductions, selection and current limitations.

- [Units](api/units.md) - built-in units, quantity and unit helpers, dimensions, metadata and errors.

## Configuration

- [Using Bilt with ruff](config.md#ruff) - keep ruff from moving imports that Bilt needs at runtime into `if TYPE_CHECKING:` blocks.
