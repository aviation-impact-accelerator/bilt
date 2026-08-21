__all__ = ["EvaluationRequest"]

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """A request to compute an output from a set of available values."""

    output: str
    available: frozenset[str]
