"""Proposal: saying what is known about an unknown port when constructing a system.

Today a constructor argument either gives a port's value (the port is known) or is
omitted (the port is unknown, to be found from the equations). That is not enough when
the equations have more than one solution. A circle and a line meet at two points, and
`radius ** 2` has two square roots: Bilt correctly refuses to choose between them.

**`ContinuousUnknown(guess=..., lower=..., upper=...)` (constructor argument).** Passed in
place of a value, it marks the port as an unknown taking real values, exactly as omitting
it does, and adds what the request knows about the answer:

    Point(x=ContinuousUnknown(lower=5.0 * meter))

- `guess` is a starting point for an iterative solve. It selects the root nearest to it,
  but is a hint, not a guarantee.
- `lower` and `upper` bound the port. They select a root by excluding the others, and are
  a guarantee: a solution outside the bounds is an error, not an answer. A bound alone
  also lets Bilt isolate a port from an even power, so
  `radius=ContinuousUnknown(lower=0.0 * meter)` takes the positive square root.

Each argument may be given alone: a guess without bounds, or bounds with the guess
inferred. `ContinuousUnknown()` with none means the same as omitting the port; a
`bilt-check` lint could flag it as redundant.

Root selection belongs to the request, not to the system: the same circle and line are
equally valid with either intersection, and a different request may want the other one.
So `ContinuousUnknown` changes no system, interface or implementation, only how one is
constructed.

**Continuous, so that a discrete sibling can follow.** Mixed-integer problems need
unknowns restricted to integers, such as a spring's coil count. Naming the continuous
case explicitly leaves room for a `DiscreteUnknown` beside it, rather than a flag on one
type that would change what `guess` and the bounds mean.

**Typing.** Eventually the constructor Bilt synthesises for a system should accept a
`ContinuousUnknown[T]` wherever it accepts a `T`. Until then, `__new__` is annotated as
returning `typing.Any`, which type checkers treat as "this call does not produce a
`ContinuousUnknown`", so a `ContinuousUnknown` can be passed for any port.
"""

import typing

from bilt_proposals._errors import not_implemented

if typing.TYPE_CHECKING:
    from bilt import Quantity, QuantityTensor, Tensor


class ContinuousUnknown[T: float | Quantity | Tensor | QuantityTensor]:
    """An unknown port taking real values, with an optional starting guess and bounds."""

    guess: T | None
    lower: T | None
    upper: T | None

    def __new__(
        cls,
        *,
        guess: T | None = None,
        lower: T | None = None,
        upper: T | None = None,
    ) -> typing.Any:  # noqa: ANN401
        """Mark a constructor argument's port as unknown, with a starting guess or bounds."""
        del cls, guess, lower, upper
        not_implemented("ContinuousUnknown")
