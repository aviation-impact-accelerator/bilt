# Bilt repository instructions for GitHub Copilot

Bilt is an equation-oriented modelling framework. This repository contains two separate Agent Skills in `.agents/skills/`:

- `bilt`: writing scientific and engineering models **with** Bilt.
- `bilt-development`: implementing, testing, maintaining or **reviewing changes to Bilt itself**.

When reviewing a pull request against this repository, use the `bilt-development` skill where relevant, especially for changes to the parser, structural analysis, coupled solves, units, tensors or backends. For user-facing model examples, also consult `bilt` when relevant.

Prioritise specific correctness and regression risks, actionable test gaps and inconsistencies with the public API. Consider invalid and overconstrained models, subsystem boundaries, unit/domain consistency and Python/JAX behaviour. Refer to concrete changed lines. Avoid style-only comments that Ruff, Pyrefly or ty already address; do not infer that CI passed unless the checks show that it did.

Skills provide context, not a substitute for inspecting code and running or reading tests. Keep comments concise and flag only issues supported by the change.
