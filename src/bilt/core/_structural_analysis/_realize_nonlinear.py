"""Local nonlinear eligibility and runtime operations using implementation numerical options."""

import ast
import typing

from bilt.core._errors import BiltNonlinearBlockError
from bilt.core._evaluation_options import current_root_options
from bilt.core._port import mangle_name
from bilt.core._structural_analysis._prepare_nonlinear import (
    prepare_nonlinear_block,
    scalar_nonlinear_declarations,
)
from bilt.core._structural_analysis._realize_linear import _function
from bilt.core._structural_analysis._residual_block import ResidualBlock, extract_residual_block
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from bilt._backend import Backend, BackendCallable
    from bilt.core._structural_analysis._plan import PlanBlock


def nonlinear_candidate(block: PlanBlock) -> ResidualBlock:
    """Check original equations and required configuration, independently of linear extraction."""
    options = current_root_options()
    if options is None:
        message = "Local nonlinear execution requires an implementation configured with RootOptions."
        raise BiltNonlinearBlockError(message)
    residual = scalar_nonlinear_candidate(block)
    missing = set(residual.unknowns) - options.variables.keys()
    if missing:
        message = f"Missing initial values and scales for nonlinear unknowns {sorted(missing)!r}."
        raise BiltNonlinearBlockError(message)
    missing_rows = [row.label for row in residual.rows if row.source not in options.residual_scales]
    if missing_rows:
        message = f"Missing residual scales for equations {missing_rows!r}."
        raise BiltNonlinearBlockError(message)
    return residual


def scalar_nonlinear_candidate(block: PlanBlock) -> ResidualBlock:
    """Recognise the formulation without conflating it with configuration availability."""
    residual = extract_residual_block(block)
    scalar_nonlinear_declarations(residual)
    return residual


def realize_nonlinear_block(
    block: PlanBlock,
    backend: Backend,
    symbol_domains: dict[str, Domain],
    *,
    solution_name: str,
) -> tuple[BackendCallable, ...]:
    """Compile one operation and projections; configuration is read only at execution."""
    residual = nonlinear_candidate(block)
    if any(not symbol_domains[name].is_scalar for name in residual.inputs):
        message = "Local nonlinear execution supports scalar input domains only."
        raise BiltNonlinearBlockError(message)
    prepared = prepare_nonlinear_block(residual, backend)
    parameters = tuple(sorted(residual.inputs))

    def execute(*values: object) -> tuple[float, ...]:
        options = current_root_options()
        if options is None:
            message = "The nonlinear operation requires its implementation configuration at execution."
            raise BiltNonlinearBlockError(message)
        result = prepared.solve_lowered(
            dict(zip(parameters, values, strict=True)),
            initial_guess={name: options.variables[name].initial for name in residual.unknowns},
            variable_scales={name: options.variables[name].scale for name in residual.unknowns},
            residual_scales=tuple(options.residual_scales[row.source] for row in residual.rows),
            solver=options.solver,
            controls=options.controls,
        )
        return tuple(result[name] for name in residual.unknowns)

    names = tuple(map(mangle_name, parameters))
    runtime_name = solution_name + "_execute"
    while runtime_name in names:
        runtime_name += "_"
    function = _function(
        solution_name,
        names,
        ast.Call(
            func=ast.Name(id=runtime_name, ctx=ast.Load()),
            args=[ast.Name(id=name, ctx=ast.Load()) for name in names],
            keywords=[],
        ),
    )
    function.__dict__["_bilt_runtime_bindings"] = {runtime_name: execute}
    callables = [backend.transpile(function)]
    for index, output in enumerate(residual.unknowns):
        projection = _function(
            mangle_name(output),
            (solution_name,),
            ast.Subscript(value=ast.Name(id=solution_name, ctx=ast.Load()), slice=ast.Constant(index), ctx=ast.Load()),
        )
        callable_ = backend.transpile(projection)
        callable_.output_domain = Domain.scalar()
        callables.append(callable_)
    return tuple(callables)
