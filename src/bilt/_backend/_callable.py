import dataclasses
import typing

from bilt._backend._errors import BiltBackendOutputArityError

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class BackendCallable:
    name: str
    parameters: tuple[str, ...]
    callable: collections.abc.Callable
    outputs: tuple[str, ...] | None = None

    def evaluate(self, inputs: collections.abc.Mapping[str, object]) -> dict[str, object]:
        """Run the callable and name every value it computed."""
        result = self.callable(**{name: inputs[name] for name in self.parameters})
        if self.outputs is None:
            return {self.name: result}
        if len(result) != len(self.outputs):
            msg = (
                f"Callable {self.name!r} returned {len(result)} values but names {len(self.outputs)}: {self.outputs!r}."
            )
            raise BiltBackendOutputArityError(msg)
        return dict(zip(self.outputs, result, strict=True))
