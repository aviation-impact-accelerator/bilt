__all__ = ["Implementation"]

import types
import typing

from bilt.core._realize_analysis import realize_plan
from bilt.core._structural_analysis._exhaustive import exhaustive_structural_backend

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend, BackendCallable
    from bilt.core._model_constituent import BiltModelConstituent

    from ._interface import Interface


class Implementation:
    """Register a collection of transforms as an implementation of a system."""

    def __init__(
        self,
        system: Interface,
        components: collections.abc.Iterable[BiltModelConstituent],
    ) -> None:
        self._system: Interface = system
        self._components: types.MappingProxyType[str, BiltModelConstituent] = types.MappingProxyType(
            {component.__bilt_name__: component for component in components}
        )
        self._callable_cache: dict[tuple[str, frozenset[str], Backend], BackendCallable] = {}
        system.__bilt_system_registry__.add(self)

    @property
    def system(self) -> Interface:
        return self._system

    def evaluate(self, inputs: dict[str, object], output: str, backend: Backend) -> object:
        cache_key = (output, frozenset(inputs.keys()), backend)
        if output in inputs:
            output_value = inputs[output]
        elif self._callable_cache.get(cache_key):
            output_value = self._callable_cache[cache_key].callable(**inputs)
        else:
            plan = exhaustive_structural_backend(tuple(self._components.values()), frozenset(inputs.keys()), output)
            self._callable_cache[cache_key] = backend.chain_case(realize_plan(plan, backend))
            output_value = self._callable_cache[cache_key].callable(**inputs)
        return output_value

    def __enter__(self) -> typing.Self:
        self._system.__bilt_system_implementation_default__.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        self._system.__bilt_system_implementation_default__.pop()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Implementation):
            return NotImplemented
        return self.system == other.system and self._components == other._components

    def __hash__(self) -> int:
        return hash((self.system, frozenset(self._components.items())))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(system={self.system}, transforms={frozenset(self._components.values())!r})"
