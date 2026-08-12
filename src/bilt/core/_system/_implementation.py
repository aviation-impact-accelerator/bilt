__all__ = ["Implementation"]

import types
import typing

import numpy as np

from bilt.core._evaluation_context import EvaluationContext
from bilt.core._structural_analysis._protocol import get_current_structural_analysis_backend
from bilt.core._tensors._lowering_and_lifting import prepare_any_tensor_inputs_for_lowering_to_callable
from bilt.core._tensors._lowering_and_lifting import lift_output

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend, BackendCallable
    from bilt.core._interface import Interface
    from bilt.core._model_constituent import BiltModelConstituent


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
        with EvaluationContext() as evaluation:
            evaluation.domains.bind_all(inputs.values())

            if output in inputs:
                output_value = inputs[output]

            cache_key = (output, frozenset(inputs.keys()), backend)

            if self._callable_cache.get(cache_key):
                final_callable = self._callable_cache[cache_key]
            else:
                structural_analysis_backend = get_current_structural_analysis_backend()
                planned_case = structural_analysis_backend(
                    tuple(self._components.values()), set(inputs.keys()), output, backend
                )
                final_callable = backend.chain_case(planned_case)
                self._callable_cache[cache_key] = final_callable

            inputs_to_use = {var_name: inputs[var_name] for var_name in final_callable.expected_parameters.keys()}
            result = prepare_any_tensor_inputs_for_lowering_to_callable(inputs_to_use, final_callable)

            output_value = typing.cast(
                np.ndarray,
                final_callable.callable(**result.prepared_inputs)
            )

            output_value = lift_output(output_value, result)
            evaluation.domains.bind(output_value)

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
