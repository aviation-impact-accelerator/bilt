import dataclasses
import typing

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._domains import Domain


@dataclasses.dataclass(frozen=True)
class BackendOutput:
    """A computed port, its compiled identifier, and its domain before lifting."""

    backend_name: str
    port: str
    domain: Domain


@dataclasses.dataclass
class BackendCallable:
    name: str
    parameters: tuple[str, ...]
    callable: collections.abc.Callable
    output_domain: Domain | None = None
    outputs: tuple[BackendOutput, ...] | None = None

    def evaluate(
        self,
        inputs: collections.abc.Mapping[str, object],
    ) -> object:
        return self.callable(**{name: inputs[name] for name in self.parameters})

    def evaluate_outputs(self, inputs: collections.abc.Mapping[str, object]) -> dict[str, object]:
        """Name positional results outside the compiled callable, using logical port names."""
        if self.outputs is None:
            msg = "This callable was not compiled to expose intermediates."
            raise RuntimeError(msg)
        values = typing.cast("tuple[object, ...]", self.evaluate(inputs))
        return {output.port: value for output, value in zip(self.outputs, values, strict=True)}
