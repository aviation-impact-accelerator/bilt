import dataclasses
import typing

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class BackendCallable:
    name: str
    parameters: tuple[str, ...]
    callable: collections.abc.Callable

    def evaluate(self, inputs: collections.abc.Mapping[str, object]) -> object:
        return self.callable(**{name: inputs[name] for name in self.parameters})
