"""Proposal: equations that return an inequality.

Bilt already states equalities in two places: in the model, as an `@equation` returning
`==`, and in a request, as a known value. This proposal does the same for inequalities,
so that a bound is one concept with two spellings, each where its equality already is:

| | Holds for every request (model) | Holds for one request (constructor) |
|---|---|---|
| Equality | `@equation` returning `==` | a known value, `x=3.0 * meter` |
| Inequality | `@equation` returning `<`, `<=`, `>` or `>=` | `x=ContinuousUnknown(lower=...)` |

A bound belongs in the model when the model rules out the other values for every
request (a circle's radius is never negative), and in the request when only the
question can choose (the positive or the negative root of `y == x**2`):

    @equation
    def nonnegative_radius(radius: Annotated[Quantity, LENGTH]) -> bool:
        return radius >= 0.0 * meter

The only change is in what `@equation` accepts: no decorator, keyword or type is added,
and `ContinuousUnknown`'s bounds become inequalities the request adds. Optimisation needs
inequality constraints anyway, and an inequality may bound an expression
(`x + y <= limit`), which a bound declared on a single port could not.

**Meaning.** An inequality never determines an unknown, so structural analysis does not
match it to one. (Today Bilt counts it like an equality, so `y == 2 * x` with `x >= 0`
is rejected as overconstrained.) What it does depends on the route Bilt takes:

- isolating a port analytically: a bound on that port selects among its branches, so
  `radius >= 0` takes the positive square root of `radius**2`;
- solving numerically: it bounds the solver;
- optimising: it is a constraint;
- with every port known: it is a check, and a violated inequality is an error.

Strict and non-strict inequalities are accepted, and differ only in that check.

`!=` is not part of this proposal: it excludes a single value, so cannot bound a solve.

**`equation` (decorator).** The same as `bilt.equation`, except that it raises
`ProposalNotImplementedError` for a relation other than `==`; it exists so a module can
import a single `equation` for all its relations.
"""

import typing

import bilt
from bilt_proposals._errors import not_implemented

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._equation import BasicEquation


def equation[R, **P](function: collections.abc.Callable[P, R]) -> BasicEquation[R, P]:
    """Decorate a function returning an equality or an inequality between ports."""
    declared = bilt.equation(function)
    tree = declared.__bilt_equation_tree__
    if tree.is_relation and not tree.is_equality:
        not_implemented(f"an @equation returning {tree.operator} ({declared.__bilt_name__})")
    return declared
