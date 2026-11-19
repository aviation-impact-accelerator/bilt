"""Backend-independent call schedule for a realised Bilt evaluation.

The schedule describes dependency ordering and output domains; executable
callable objects remain owned by BackendCase in this incremental prototype.
"""

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from bilt._backend._case import BackendCase
    from bilt.symbolic._domains import Domain


@dataclasses.dataclass(frozen=True, slots=True)
class CallStep:
    """One named computation and the previously available values it consumes."""

    output: str
    inputs: tuple[str, ...]
    domain: Domain | None = None
    kind: typing.Literal["primitive", "child", "linear_solve", "projection"] = "primitive"


@dataclasses.dataclass(frozen=True, slots=True)
class ProgramIR:
    """A directed dependency schedule, without Python AST or native callables."""

    parameters: tuple[str, ...]
    steps: tuple[CallStep, ...]
    result: str

    def verify(self) -> None:
        available = set(self.parameters)
        if len(available) != len(self.parameters):
            raise ValueError("Duplicate program parameter.")
        for step in self.steps:
            if step.output in available:
                raise ValueError(f"Program value {step.output!r} has multiple producers.")
            missing = set(step.inputs) - available
            if missing:
                raise ValueError(f"Program step {step.output!r} needs unavailable inputs: {sorted(missing)!r}.")
            available.add(step.output)
        if self.result not in available:
            raise ValueError(f"Program result {self.result!r} has no producer.")


def program_from_case(case: BackendCase) -> ProgramIR:
    """Snapshot compiled case dependencies without embedding backend callables."""
    steps = tuple(
        CallStep(
            output=step.name,
            inputs=step.parameters,
            domain=step.output_domain,
            kind="linear_solve" if step.name.startswith("__bilt_linear_solution_") else "primitive",
        )
        for step in case.intermediates
    )
    program = ProgramIR(parameters=case.parameters, steps=steps, result=case.return_name)
    program.verify()
    return program
