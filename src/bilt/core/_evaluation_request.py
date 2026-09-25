__all__ = ["EvaluationRequest"]

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from bilt.symbolic._domains import Domain


@dataclasses.dataclass(slots=True)
class EvaluationRequest:
    """A request to compute an output from inputs with known domains."""

    output: str
    input_domains: dict[str, Domain]
