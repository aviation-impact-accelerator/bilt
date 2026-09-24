import dataclasses
import typing

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class BackendCallable:
    name: str
    parameters: tuple[str, ...]
    callable: collections.abc.Callable
    outputs: tuple[str, ...] | None = None

    def evaluate(self, inputs: collections.abc.Mapping[str, object]) -> dict[str, object]:
        """Run the callable and name every value it computed.

        A callable with no ``outputs`` returns one unnamed value, reported under ``name``. A callable with
        ``outputs`` returns its values positionally, in the order ``outputs`` lists them.
        """
        result = self.callable(**{name: inputs[name] for name in self.parameters})
        if self.outputs is None:
            return {self.name: result}
        return dict(zip(self.outputs, result, strict=True))
