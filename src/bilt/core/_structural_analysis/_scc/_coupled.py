__all__ = ["analyse_coupled_block"]

import typing

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._errors import BiltSystemAnalysisError
from bilt.core._structural_analysis._plan import (
    AnalysisPlan,
    CoupledBlock,
    PlanStep,
)
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform
from bilt.symbolic._atoms import SymbolType
from bilt.symbolic._linear import SymbolicLinearSystem, linearise
from bilt.symbolic._tree import Tree


def analyse_coupled_block(
    block: CoupledBlock,
) -> SymbolicLinearSystem:
    """Analyse a structurally coupled block as a symbolic linear system."""
    equations: list[Tree] = []
    unknowns: list[str] = []

    for step in block.steps:
        _expand_step(
            step,
            equations=equations,
            unknowns=unknowns,
            prefix=None,
        )

    return linearise(
        equations=tuple(equations),
        unknowns=tuple(unknowns),
    )


def _expand_plan(
    plan: AnalysisPlan,
    *,
    equations: list[Tree],
    unknowns: list[str],
    prefix: str,
) -> None:
    """Expand a selected structural route through a child system."""
    for item in plan.steps:
        match item:
            case PlanStep() as step:
                _expand_step(
                    step,
                    equations=equations,
                    unknowns=unknowns,
                    prefix=prefix,
                )

            case CoupledBlock() as block:
                for step in block.steps:
                    _expand_step(
                        step,
                        equations=equations,
                        unknowns=unknowns,
                        prefix=prefix,
                    )


def _expand_step(
    step: PlanStep,
    *,
    equations: list[Tree],
    unknowns: list[str],
    prefix: str | None,
) -> None:
    """Add the equations represented by one structural step."""
    match step.operation:
        case BasicEquation() as equation:
            tree = equation.__bilt_equation_tree__

            if prefix is not None:
                tree = _qualify_tree(
                    tree,
                    prefix,
                )

            equations.append(tree)
            _append_unknown(
                unknowns,
                _qualify_name(
                    step.output,
                    prefix,
                ),
            )

        case SubSystemConstituent() as sub_system:
            sub_system_plan = step.sub_system_plan

            if sub_system_plan is None:
                msg = (
                    f"Coupled sub-system step {step.output!r} "
                    "has no selected child plan."
                )
                raise BiltSystemAnalysisError(msg)

            _expand_plan(
                sub_system_plan.plan,
                equations=equations,
                unknowns=unknowns,
                prefix=_qualify_name(
                    sub_system.__bilt_name__,
                    prefix,
                ),
            )

        case BasicTransform():
            msg = (
                "Coupled blocks containing transforms "
                "are not yet supported."
            )
            raise BiltSystemAnalysisError(msg)

        case _:
            msg = (
                f"Constituent type {type(step.operation)} "
                "is not supported in a coupled block."
            )
            raise NotImplementedError(msg)


def _qualify_tree(
    tree: Tree,
    prefix: str,
) -> Tree:
    """Qualify symbols in a child equation into the parent namespace."""
    cache: dict[Tree, Tree] = {}

    def qualify(node: Tree) -> Tree:
        if node in cache:
            return cache[node]

        if (
            node.is_leaf
            and node.head.atom_type is SymbolType
        ):
            name = typing.cast("str", node.head.value)
            result = Tree(
                SymbolType(
                    _qualify_name(name, prefix)
                )
            )

        elif node.is_leaf:
            result = node

        else:
            result = Tree(
                node.head,
                tuple(
                    qualify(child)
                    for child in node.args
                ),
            )

        cache[node] = result
        return result

    return qualify(tree)


def _qualify_name(
    name: str,
    prefix: str | None,
) -> str:
    if prefix is None:
        return name

    return f"{prefix}.{name}"


def _append_unknown(
    unknowns: list[str],
    unknown: str,
) -> None:
    if unknown not in unknowns:
        unknowns.append(unknown)