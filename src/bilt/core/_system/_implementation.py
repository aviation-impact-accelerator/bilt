from __future__ import annotations

__all__ = ["Implementation"]

import collections.abc
import types
import typing

from bilt.core._system._analysis import (
    encode_inputs,
    find_successful_case,
    flatten_inputs,
)

from ._interface import Interface

if typing.TYPE_CHECKING:
    from bilt._backend import Backend, BackendCallable
    from bilt.core._model_constituent import BiltModelConstituent


class Implementation:
    """Register transforms and hierarchical subsystem implementations.

    ``children`` maps public system-valued fields to their implementations.
    ``internal_systems`` defines private implementation-owned subsystems.
    ``internal_values`` supplies fixed values in this implementation's local
    hierarchical namespace.
    """

    def __init__(
        self,
        system: Interface,
        components: (
            collections.abc.Iterable[BiltModelConstituent] | collections.abc.Mapping[str, BiltModelConstituent]
        ),
        *,
        children: collections.abc.Mapping[str, Implementation] | None = None,
        internal_systems: (collections.abc.Mapping[str, Implementation] | None) = None,
        internal_values: collections.abc.Mapping[str, object] | None = None,
    ) -> None:
        self._system: Interface = system

        if isinstance(components, collections.abc.Mapping):
            registered_components = dict(components)
        else:
            registered_components = {component.__bilt_name__: component for component in components}

        self._components: types.MappingProxyType[
            str,
            BiltModelConstituent,
        ] = types.MappingProxyType(registered_components)
        self._children: types.MappingProxyType[
            str,
            Implementation,
        ] = types.MappingProxyType(dict(children or {}))
        self._internal_systems: types.MappingProxyType[
            str,
            Implementation,
        ] = types.MappingProxyType(dict(internal_systems or {}))
        self._internal_values: types.MappingProxyType[
            str,
            object,
        ] = types.MappingProxyType(dict(internal_values or {}))
        self._callable_cache: dict[
            tuple[str, tuple[str, ...], Backend],
            tuple[BackendCallable, tuple[str, ...]],
        ] = {}

        self._validate_children()
        self._validate_internal_systems()
        self._validate_internal_values()
        system.__bilt_system_registry__.add(self)

    @property
    def system(self) -> Interface:
        return self._system

    def evaluate(
        self,
        inputs: dict[str, object],
        output: str,
        backend: Backend,
    ) -> object:
        known_values = self._collect_internal_values()
        known_values.update(flatten_inputs(inputs))

        if output in known_values:
            return known_values[output]

        cache_key = (
            output,
            tuple(sorted(known_values)),
            backend,
        )
        cached = self._callable_cache.get(cache_key)

        if cached is None:
            backend_case = find_successful_case(
                self,
                known_values,
                output,
                backend,
            )
            backend_callable = backend.chain_case(backend_case)
            parameters = backend_case.parameters
            self._callable_cache[cache_key] = (
                backend_callable,
                parameters,
            )
        else:
            backend_callable, parameters = cached

        return backend_callable.callable(
            **encode_inputs(
                known_values,
                parameters,
            )
        )

    def __enter__(self) -> typing.Self:
        self._system.__bilt_system_implementation_default__.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        self._system.__bilt_system_implementation_default__.pop()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Implementation):
            return NotImplemented

        return (
            self.system == other.system
            and self._components == other._components
            and self._children == other._children
            and self._internal_systems == other._internal_systems
            and self._internal_values == other._internal_values
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.system,
                frozenset(self._components.items()),
                frozenset(self._children.items()),
                frozenset(self._internal_systems.items()),
                frozenset(self._internal_values.items()),
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"system={self.system}, "
            f"transforms={frozenset(self._components.values())!r}, "
            f"children={dict(self._children)!r}, "
            f"internal_systems={dict(self._internal_systems)!r}, "
            f"internal_values={dict(self._internal_values)!r}"
            ")"
        )

    def _validate_children(self) -> None:
        for name, implementation in self._children.items():
            attribute = self.system.__bilt_system_attributes__.get(name)

            if attribute is None:
                message = f"System {self.system.__name__!r} has no child field {name!r}."
                raise TypeError(message)

            if not isinstance(attribute.annotation, Interface):
                message = f"Field {name!r} on {self.system.__name__!r} is not a system-valued field."
                raise TypeError(message)

            if attribute.annotation is not implementation.system:
                message = (
                    f"Child field {name!r} expects "
                    f"{attribute.annotation.__name__!r}, but the supplied "
                    f"implementation implements "
                    f"{implementation.system.__name__!r}."
                )
                raise TypeError(message)

    def _validate_internal_systems(self) -> None:
        public_names = set(self.system.__bilt_system_attributes__)

        for name in self._internal_systems:
            if not name.isidentifier():
                message = f"Internal system name {name!r} must be a valid Python identifier."
                raise TypeError(message)

            if name in public_names:
                message = f"Internal system name {name!r} conflicts with a public field on {self.system.__name__!r}."
                raise TypeError(message)

    def _validate_internal_values(self) -> None:
        for name, value in self._internal_values.items():
            if not name or any(not part for part in name.split(".")):
                message = f"Invalid internal value path {name!r}."
                raise TypeError(message)

            try:
                hash(value)
            except TypeError:
                message = f"Internal value {name!r} must currently be hashable because implementations are hashable."
                raise TypeError(message) from None

    def _collect_internal_values(
        self,
        *,
        scope: tuple[str, ...] = (),
    ) -> dict[str, object]:
        values = {".".join((*scope, name)): value for name, value in self._internal_values.items()}

        for name, implementation in self._children.items():
            values.update(
                implementation._collect_internal_values(  # noqa: SLF001
                    scope=(*scope, name),
                )
            )

        for name, implementation in self._internal_systems.items():
            values.update(
                implementation._collect_internal_values(  # noqa: SLF001
                    scope=(*scope, name),
                )
            )

        return values
