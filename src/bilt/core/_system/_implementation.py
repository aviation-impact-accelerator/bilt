__all__ = ["Implementation"]

import types
import typing

import numpy as np

from bilt.core._evaluation_context import EvaluationContext
from bilt.core._structural_analysis._protocol import get_current_structural_analysis_backend
from bilt.core._tensors._tensor_analysis import align_extra_dims, any_tensors, canonicalised, convert_any_tensors_to_raw_arrays, identify_extra_dims, transpose_any_tensors_to_canonical_order, union
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.symbolic._domains import Domain

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

            any_tensor_inputs = any_tensors(list(inputs.values()))

            if any_tensor_inputs:
                inputs = transpose_any_tensors_to_canonical_order(inputs, list(evaluation.domains._bindings.keys()))
                inputs_raw = convert_any_tensors_to_raw_arrays(inputs)
            else:
                inputs_raw = inputs

            cache_key = (output, frozenset(inputs.keys()), backend)
            if output in inputs:
                output_value = inputs[output]
            elif self._callable_cache.get(cache_key):
                cached = self._callable_cache[cache_key]
                output_value = cached.callable(**{k.name: inputs_raw[k.name] for k in cached.parameters.values()})
            else:
                structural_analysis_backend = get_current_structural_analysis_backend()
                planned_case = structural_analysis_backend(
                    tuple(self._components.values()), set(inputs.keys()), output, backend
                )

                final_callable = backend.chain_case(planned_case)
                callable_inputs = {var.name: inputs_raw[var.name] for var in final_callable.parameters.values()}

                input_domains: list[Domain] = []

                for name in callable_inputs.keys():
                    input = inputs[name]
                    if isinstance(input, Tensor):
                        input_domains.append(input.domain)
                
                input_union = canonicalised(union(*input_domains))
                callable_union = final_callable.union_domain
                extra_dims = identify_extra_dims(input_union, callable_union)

                for name in callable_inputs.keys():
                    corresponding_input = inputs[name]

                    if isinstance(corresponding_input, Tensor):
                        callable_inputs[name] = align_extra_dims(corresponding_input, list(extra_dims)).values                    

                self._callable_cache[cache_key] = final_callable
                output_value = typing.cast(
                    np.ndarray,
                    final_callable.callable(**callable_inputs)
                )

                if any_tensor_inputs:
                    output_value = tensor(
                        output_value,
                        over=planned_case.returns.domain.dims + extra_dims
                    )
                else:
                    output_value = output_value.item()

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
