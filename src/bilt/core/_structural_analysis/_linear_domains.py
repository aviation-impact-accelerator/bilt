"""Resolve named domains for tensor linear-block unknowns."""

import inspect
import typing

from bilt.core._port_declaration import annotated_parts
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._equation import BasicEquation
    from bilt.core._structural_analysis._plan import PlanBlock


def declared_linear_domains(block: PlanBlock) -> dict[str, Domain]:
    """Read optional Annotated[..., Domain(...)] hints on primitive equation ports.

    A hint specifies named axes, not concrete extents. Repeated declarations
    must agree. Other annotation metadata retains its existing meaning.
    """
    return merge_domain_hints(
        hint for step in block.steps for hint in equation_domain_hints(typing.cast("BasicEquation", step.operation))
    )


def equation_domain_hints(equation: BasicEquation) -> tuple[tuple[str, Domain], ...]:
    """Read the existing primitive-parameter hint syntax without assigning a namespace."""
    result = []
    function = equation._function  # noqa: SLF001 - primitive equation metadata
    for name, parameter in inspect.signature(function, eval_str=True).parameters.items():
        _, metadata = annotated_parts(parameter.annotation)
        hints = [item for item in metadata if isinstance(item, Domain)]
        if len(hints) > 1:
            message = f"Conflicting linear-block domain hints for {name!r}."
            raise BiltLinearBlockError(message)
        if hints:
            result.append((name, hints[0]))
    return tuple(result)


def merge_domain_hints(hints: collections.abc.Iterable[tuple[str, Domain]]) -> dict[str, Domain]:
    """Merge hints by attribute name, preserving the existing axis-set agreement rule."""
    result: dict[str, Domain] = {}
    for name, hint in hints:
        if name in result and set(result[name].dims) != set(hint.dims):
            message = f"Conflicting linear-block domain hints for {name!r}."
            raise BiltLinearBlockError(message)
        result[name] = hint
    return result
