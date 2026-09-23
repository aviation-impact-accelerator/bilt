__all__ = ["EvaluationRequest"]

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from bilt.symbolic._domains import Domain


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """A request to compute an output from inputs with known domains."""

    output: str
    input_domains: frozenset[tuple[str, Domain]]

    @classmethod
    def from_domains(
        cls,
        output: str,
        input_domains: dict[str, Domain],
    ) -> EvaluationRequest:
        return cls(
            output=output,
            input_domains=frozenset(input_domains.items()),
        )

    @property
    def available(self) -> frozenset[str]:
        """Return the available input names."""
        return frozenset(name for name, _ in self.input_domains)

    @property
    def domains(self) -> dict[str, Domain]:
        """Return the domains of the available inputs."""
        return dict(self.input_domains)
