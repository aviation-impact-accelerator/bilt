import dataclasses
import typing

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._domains import Domain
    from bilt.symbolic._tree import Tree


@dataclasses.dataclass
class BackendCallable:
    name: str
    parameters: tuple[str, ...]
    callable: collections.abc.Callable
    output_domain: Domain | None = None
    symbolic_tree: Tree | None = None
    symbolic_parameters: tuple[str, ...] = ()

    def evaluate(
        self,
        inputs: collections.abc.Mapping[str, object],
    ) -> object:
        return self.callable(**{name: inputs[name] for name in self.parameters})
