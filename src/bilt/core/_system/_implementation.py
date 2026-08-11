__all__ = ["Implementation"]

import functools
import types
import typing

from bilt.core._equation import BasicEquation
from bilt.core._realize_analysis._realize import canonicalize_backend_callable
from bilt.core._structural_analysis._exhaustive import ExhaustiveStructuralAnalyser
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system._error import (
    BiltSubSystemConflictError,
    BiltSubSystemMismatchedImplementationError,
    BiltSubSystemMissingImplementationError,
    BiltSubSystemNameCollisionError,
    BiltSubSystemUnreferencedEntryError,
)

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend, BackendCallable
    from bilt.core._interface import Interface
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._protocol import StructuralAnalyser


def _gather_sub_system_references(
    components_list: list[BiltModelConstituent],
) -> dict[str, Interface]:
    """Gather and validate sub-system references from components."""
    all_sub_systems: dict[str, Interface] = {}
    for component in components_list:
        if not isinstance(component, BasicEquation):
            continue
        for ref_name, interface in component.__bilt_sub_systems__.items():
            if ref_name in all_sub_systems:
                if all_sub_systems[ref_name] != interface:
                    msg_interfaces = [all_sub_systems[ref_name].__name__, interface.__name__]
                    raise BiltSubSystemConflictError(ref_name, msg_interfaces)
            else:
                all_sub_systems[ref_name] = interface
    return all_sub_systems


def _validate_sub_system_implementations(
    all_sub_systems: dict[str, Interface],
    internal_implementations_dict: dict[str, Implementation],
    component_dict: dict[str, BiltModelConstituent],
) -> None:
    """Validate sub-system references against implementations and component names."""
    for ref_name, interface in all_sub_systems.items():
        if ref_name not in internal_implementations_dict:
            raise BiltSubSystemMissingImplementationError(
                ref_name, interface.__name__, list(internal_implementations_dict.keys())
            )

        implementation = internal_implementations_dict[ref_name]

        if implementation.system != interface:
            raise BiltSubSystemMismatchedImplementationError(
                ref_name, interface.__name__, implementation.system.__name__
            )

    for entry_name in internal_implementations_dict:
        if entry_name not in all_sub_systems:
            raise BiltSubSystemUnreferencedEntryError(entry_name, sorted(all_sub_systems))

    for ref_name in all_sub_systems:
        if ref_name in component_dict:
            raise BiltSubSystemNameCollisionError(ref_name)


class Implementation:
    """Register a collection of transforms as an implementation of a system."""

    def __init__(
        self,
        system: Interface,
        components: collections.abc.Iterable[BiltModelConstituent],
        internal_implementations: dict[str, Implementation] | None = None,
    ) -> None:
        self._system: Interface = system
        components_list = list(components)
        component_dict: dict[str, BiltModelConstituent] = {
            component.__bilt_name__: component for component in components_list
        }

        internal_implementations_dict = internal_implementations or {}

        all_sub_systems = _gather_sub_system_references(components_list)
        _validate_sub_system_implementations(all_sub_systems, internal_implementations_dict, component_dict)

        for ref_name, interface in all_sub_systems.items():
            component_dict[ref_name] = SubSystemConstituent(
                __bilt_name__=ref_name, interface=interface, implementation=internal_implementations_dict[ref_name]
            )

        self._components: types.MappingProxyType[str, BiltModelConstituent] = types.MappingProxyType(component_dict)
        self._callable_cache: dict[tuple[str, frozenset[str], Backend], BackendCallable] = {}
        system.__bilt_system_registry__.add(self)

    @property
    def system(self) -> Interface:
        return self._system

    @functools.cached_property
    def analyser(self) -> StructuralAnalyser:
        """Lazily build and cache the structural analyser for this implementation."""
        return ExhaustiveStructuralAnalyser(tuple(self._components.values()))

    def realized_callable(
        self,
        input_names: frozenset[str],
        output: str,
        backend: Backend,
    ) -> BackendCallable:
        """Return the cached or freshly-realized chained callable for this output."""
        cache_key = (output, input_names, backend)
        if cache_key in self._callable_cache:
            return self._callable_cache[cache_key]

        planned_case = self.analyser.analyse(set(input_names), output, backend)
        mangled_callable = backend.chain_case(planned_case)
        final_callable = canonicalize_backend_callable(mangled_callable, tuple(sorted(input_names)))

        self._callable_cache[cache_key] = final_callable
        return final_callable

    def evaluate(self, inputs: dict[str, object], output: str, backend: Backend) -> object:
        if output in inputs:
            return inputs[output]

        final_callable = self.realized_callable(frozenset(inputs.keys()), output, backend)
        return final_callable.callable(**{k: inputs[k] for k in final_callable.parameters})

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
