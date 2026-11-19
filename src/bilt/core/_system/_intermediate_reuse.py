"""Reuse completed ports only within an execution plan built from supplied inputs."""

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from bilt._backend._case import BackendCase


def reuse_plan_outputs(case: BackendCase, available_ports: frozenset[str]) -> BackendCase:
    """Prune a *validated* pipeline backwards from its requested output.

    Cached values replace only operations actually needed by the selected plan.
    Structural analysis and its input boundary remain unchanged. In particular,
    the internal output of a coupled linear solve is never mistaken for a port.
    """
    known_names = {item.backend_name for item in case.outputs if item.port in available_ports}
    if not known_names:
        return case

    needed = {case.return_name}
    kept = []
    for step in reversed(case.intermediates):
        if step.name not in needed:
            continue
        needed.remove(step.name)
        if step.name in known_names:
            needed.add(step.name)
            continue
        kept.append(step)
        needed.update(step.parameters)

    permitted = set(case.parameters) | known_names
    if not needed <= permitted:
        msg = f"Plan has unresolved execution dependencies: {sorted(needed - permitted)}."
        raise RuntimeError(msg)

    kept.reverse()
    computed = {step.name for step in kept}
    return dataclasses.replace(
        case,
        parameters=tuple(name for name in case.parameters if name in needed)
        + tuple(sorted(needed & known_names)),
        intermediates=kept,
        outputs=tuple(item for item in case.outputs if item.backend_name in computed),
    )
