import dataclasses
import typing

from bilt._backend._tensor_handling._lowering_and_lifting import generate_lifting_plan, lift_output, lower_inputs

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
        lowered_inputs = lower_inputs(inputs_to_use)
        lifting_plan = generate_lifting_plan(inputs_to_use)
        output_value = self.callable(**lowered_inputs)
        return lift_output(output_value, lifting_plan)
