import dataclasses
import typing

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class BackendCallable:
    """A compiled unit of work: one operation, or a whole chained case.

    ``outputs`` names the keys of the mapping ``callable`` returns. It is ``None`` when ``callable``
    returns a single value, which is the value named by ``name`` -- the contract for a transpiled
    operation, and for a chained case built with ``expose_intermediates=False``.
    """

    name: str
    parameters: tuple[str, ...]
    callable: collections.abc.Callable
    outputs: tuple[str, ...] | None = None

    def evaluate(self, inputs: collections.abc.Mapping[str, object]) -> dict[str, object]:
        """Run the callable and name every value it computed."""
        result = self.callable(**{name: inputs[name] for name in self.parameters})
        if self.outputs is None:
            return {self.name: result}
        return result
