# Bilt — Context

A glossary of the core domain language for Bilt, a package for systems modelling.
This file defines terms only; it is not a spec and holds no implementation detail.

## Modelling stance

**Hybrid (acausal + causal) by design.** Bilt supports both modelling styles as
first-class citizens, because real systems models need both:

- **Acausal** modelling suits governing physics, where a relationship holds in no
  particular direction (e.g. the governing equations of turbomachinery). The
  modeller states the relationship; the tool decides what to solve for.
- **Causal** modelling suits relationships that can only ever be evaluated in one
  direction (e.g. interpolating experimental fluid-property data — you read a
  table forwards, you cannot invert it).

A single model is expected to mix both.

## Terms

- **System** — a named collection of attributes describing some thing being
  modelled (e.g. a `Compressor`). Carries no behaviour itself.

- **Interface** — the public face of a System: the set of attributes it exposes.
  A System's Interface is separate from any particular way of computing those
  attributes.

- **Implementation** — a particular set of relationships that realise a System's
  Interface. One Interface may have many Implementations.

- **Transform** — a *causal* relationship: a directed function that computes one
  named output from its inputs. Evaluable in one direction only. Because its
  causality is fixed, the tool never symbolically analyses a Transform's body,
  so a Transform may contain arbitrary Python (e.g. table interpolation).
  Understood as a special case of an Equation whose solve direction is
  pre-defined.

- **Equation** — an *acausal* relationship: a residual/relation that holds
  between attributes without designating an output. The tool decides which
  variable to solve for, which requires the Equation's body to be symbolically
  analysable.

Both Transforms and Equations are relationships over a System's attributes and
are resolved by a single unified solver; the distinction (a) declares the
modeller's intent about reversibility and (b) lets the backend treat the two
differently where useful.
