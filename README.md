# Bilt

**Composable engineering models from declarative relationships.**

Bilt is a Python framework for building acausal, composable and unit-aware engineering models.

Instead of writing a fixed calculation pipeline, you describe the relationships in the model. Given what is known and what is requested, Bilt works out how to calculate the result.

> **The modeller declares meaning. Bilt derives mechanics.**

## Core ideas

* **Systems** define composable model boundaries.
* **Values** describe the quantities in a model.
* **Equations** declare acausal relationships between values.
* **Transforms** declare directional relationships.
* **Structural analysis** finds a valid route from known values to requested values.
* **Domains and units** preserve the meaning of numerical data.

For example, an equation such as

[
d = vt
]

does not inherently say whether to calculate `d`, `v`, or `t`.

Bilt keeps the relationship intact and chooses a direction when the model is evaluated.

The model declares the web.

Evaluation finds a route through it.

## Why Bilt?

Engineering models often begin as equations and end up encoded as rigid sequences of calculations.

Bilt keeps the model separate from the mechanics used to execute it. This makes relationships easier to compose, reuse and inspect, while leaving Bilt free to derive an appropriate computation plan for each evaluation.

That plan can include structural analysis, equation isolation, domain alignment, lowering and execution.

## Install

```bash
git clone https://github.com/aviation-impact-accelerator/bilt.git
cd bilt
uv sync
```

Run the tests with:

```bash
uv run pytest
```

Bilt currently targets Python 3.14.

## Status

Bilt is under active development at the **Aviation Impact Accelerator (AIA)** at the University of Cambridge.

We currently use it to develop engineering models within AIA while continuing to evolve the public API and modelling architecture. Expect breaking changes as the project develops.

We are interested in collaboration around scientific modelling, structural analysis, compiler techniques, heterogeneous execution and explainable model evaluation.

## Learn more

See the [documentation](./docs/) and [examples](./examples/) for the current API.

For the less conventional explanation, read [**The Teachings**](./docs/the-teachings.md).

> **The model is declared.
> The question is asked.
> Execution is derived.**
