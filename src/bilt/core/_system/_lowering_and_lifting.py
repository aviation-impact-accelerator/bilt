import dataclasses
import functools
from typing import TYPE_CHECKING, cast

from bilt.core._canonical_ordering import canonicalized
from bilt.core._evaluation_context import current_evaluation
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.units import Quantity
from bilt.units._units import BASE_UNIT_SYSTEM

if TYPE_CHECKING:
    from bilt.core._canonical_ordering import CanonicalizedDomain
    from bilt.core._port_declaration import PortDeclaration
    from bilt.symbolic._domains import Domain
    from bilt.units import Unit


@dataclasses.dataclass(frozen=True)
class LiftingPlan:
    output_domain: CanonicalizedDomain
    squeezed_axes: tuple[int, ...]
    output_unit: Unit | None

    @functools.cached_property
    def should_lift_into_tensor(self) -> bool:
        return not self.output_domain.domain.is_scalar


def lower_inputs(inputs: dict[str, object]) -> dict[str, object]:
    lowered_inputs: dict[str, object] = {}

    for input_name, input_value in inputs.items():
        if isinstance(input_value, Tensor):
            lowered_inputs[input_name] = input_value.values
        elif isinstance(input_value, Quantity):
            lowered_inputs[input_name] = BASE_UNIT_SYSTEM.coherent_value(input_value)
        else:
            lowered_inputs[input_name] = input_value

    return lowered_inputs


def generate_lifting_plan(output_domain: Domain, output_declaration: PortDeclaration | None = None) -> LiftingPlan:
    canonical_order = current_evaluation().canonical_order

    return LiftingPlan(
        output_domain=canonicalized(output_domain, canonical_order),
        squeezed_axes=tuple(axis for axis, dim in enumerate(canonical_order.dims) if dim not in output_domain.dims),
        output_unit=None if output_declaration is None else output_declaration.coherent_unit,
    )


def lift_output(
    output_value: object,
    lifting_plan: LiftingPlan,
) -> Tensor | object:
    if lifting_plan.squeezed_axes:
        output_value = _squeeze_axes(
            output_value,
            lifting_plan.squeezed_axes,
        )

    if lifting_plan.should_lift_into_tensor:
        output_value = tensor(
            output_value,
            over=lifting_plan.output_domain.domain.dims,
        )
    elif lifting_plan.output_unit is not None:
        output_value = Quantity(cast("float", output_value), lifting_plan.output_unit)

    return output_value


def _squeeze_axes(
    value: object,
    axes: tuple[int, ...],
) -> object:
    squeeze = getattr(value, "squeeze", None)

    if squeeze is None:
        return value

    return squeeze(axis=axes)
