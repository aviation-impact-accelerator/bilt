"""Resolve named domains for tensor linear-block unknowns."""

import inspect
import typing

from bilt.core._port_declaration import annotated_parts
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from bilt.core._equation import BasicEquation
    from bilt.core._structural_analysis._plan import PlanBlock


def declared_linear_domains(block: PlanBlock) -> dict[str, Domain]:
    """Read optional Annotated[..., Domain(...)] hints on primitive equation ports.

    A hint specifies named axes, not concrete extents. Repeated declarations
    must agree. Other annotation metadata retains its existing meaning.
    """
    result = {}
    for step in block.steps:
        function = typing.cast("BasicEquation", step.operation)._function  # noqa: SLF001 - primitive equation metadata
        for name, parameter in inspect.signature(function, eval_str=True).parameters.items():
            _, metadata = annotated_parts(parameter.annotation)
            hints = [item for item in metadata if isinstance(item, Domain)]
            if len(hints) > 1 or (hints and name in result and set(result[name].dims) != set(hints[0].dims)):
                message = f"Conflicting linear-block domain hints for {name!r}."
                raise BiltLinearBlockError(message)
            if hints:
                result[name] = hints[0]
    return result
