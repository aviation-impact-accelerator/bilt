import dataclasses
import typing

from bilt._backend._tensor_handling._lowering_and_lifting import lift_output, prepare_any_tensor_inputs_for_lowering

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class BackendCallable:
    name: str
    parameters: tuple[str, ...]
    callable: collections.abc.Callable

    def evaluate(
        self,
        inputs: collections.abc.Mapping[str, object],
    ) -> object:
        inputs_to_use = {param_name: inputs[param_name] for param_name in self.parameters}
        lowering_result = prepare_any_tensor_inputs_for_lowering(inputs_to_use)
        output_value = self.callable(**lowering_result.prepared_inputs)
        return lift_output(output_value, lowering_result)
