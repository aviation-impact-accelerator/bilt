"""Direct preparation of scalar residual blocks, without changing plan selection."""

import ast
import dataclasses
import math
import numbers
import typing

from bilt._backend import NonlinearSolveBackend
from bilt._backend._errors import BiltBackendError, BiltNonlinearSolveError
from bilt._backend._nonlinear_solve import NonlinearControls
from bilt.core._dimensional_analysis import dimensionality_of
from bilt.core._errors import BiltNonlinearBlockError
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._port import mangle_name
from bilt.core._port_declaration import PortDeclaration
from bilt.core._system._input_validation import validate_inputs
from bilt.core._system._lowering_and_lifting import generate_lifting_plan, lift_output, lower_inputs
from bilt.core._tensors._tensors import Tensor
from bilt.core._to_ast import tree_to_functiondef
from bilt.symbolic._atoms import QuantityType
from bilt.symbolic._domains import Domain
from bilt.symbolic._walk import subtrees
from bilt.units.dimensions import DIMENSIONLESS

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from bilt._backend import Backend, BackendCallable
    from bilt._backend._nonlinear_solve import NonlinearSolver
    from bilt.core._structural_analysis._residual_block import ResidualBlock


@dataclasses.dataclass(frozen=True)
class PreparedNonlinearBlock:
    """Compiled residuals and unit declarations, reusable across direct solves.

    Guesses are independent of boundary inputs. Variable scales follow unknown
    names; residual scales follow row order. All scales are relative quantities
    in their variable/residual dimensions, or bare numbers for plain expressions.
    Runtime failures do not alter this prepared structure or cache an answer.
    """

    block: ResidualBlock
    backend: NonlinearSolveBackend
    residual: BackendCallable
    declarations: dict[str, PortDeclaration]
    residual_declarations: dict[str, PortDeclaration]

    def solve(  # noqa: PLR0913 - explicit solver, scales and numerical controls
        self,
        inputs: Mapping[str, object],
        *,
        initial_guess: Mapping[str, object],
        variable_scales: Mapping[str, object],
        residual_scales: tuple[object, ...],
        controls: NonlinearControls | None = None,
        solver: NonlinearSolver | None = None,
    ) -> dict[str, object]:
        """Solve one square scalar block and lift each result to its declared quantity."""
        unknowns = self.block.unknowns
        _require_names(inputs, self.block.inputs, "external inputs")
        _require_names(initial_guess, unknowns, "initial guesses")
        _require_names(variable_scales, unknowns, "variable scales")
        if len(residual_scales) != len(self.block.rows):
            message = "A residual scale is required for every equation row, in row order."
            raise BiltNonlinearBlockError(message)
        boundary = _numeric(inputs, self.declarations)
        guess = _numeric(initial_guess, self.declarations)
        scale_declarations = {
            name: dataclasses.replace(declaration, absolute=False) for name, declaration in self.declarations.items()
        }
        x_scale = _numeric(variable_scales, scale_declarations)
        f_scale = _numeric({str(i): scale for i, scale in enumerate(residual_scales)}, self.residual_declarations)

        def residual(values: tuple[float, ...]) -> object:
            arguments = boundary | dict(zip(unknowns, values, strict=True))
            return self.residual.evaluate({mangle_name(name): value for name, value in arguments.items()})

        try:
            solution = self.backend.solve_nonlinear(
                residual,
                tuple(guess[name] for name in unknowns),
                variable_scales=tuple(x_scale[name] for name in unknowns),
                residual_scales=tuple(f_scale[str(i)] for i in range(len(residual_scales))),
                controls=NonlinearControls() if controls is None else controls,
                solver=solver,
            )
        except BiltNonlinearSolveError:
            raise
        except BiltBackendError as error:
            raise BiltNonlinearBlockError(str(error)) from error
        with EvaluationContext():
            return {
                name: lift_output(value, generate_lifting_plan(Domain.scalar(), self.declarations.get(name)))
                for name, value in zip(unknowns, solution, strict=True)
            }


def prepare_nonlinear_block(block: ResidualBlock, backend: Backend) -> PreparedNonlinearBlock:
    """Compile scalar arithmetic residuals; leave planner integration to a later step."""
    if not isinstance(backend, NonlinearSolveBackend):
        message = "The selected backend does not support scalar nonlinear solves."
        raise BiltNonlinearBlockError(message)
    if len(block.unknowns) != len(block.rows):
        message = "A scalar nonlinear solve requires equally many residual rows and unknowns."
        raise BiltNonlinearBlockError(message)
    declarations = _declarations(block)
    residual_declarations = {}
    dimensions = {
        name: declarations[name].dimensionality if name in declarations else DIMENSIONLESS
        for name in (*block.unknowns, *block.inputs)
    }
    for index, (row, expression) in enumerate(zip(block.rows, block.residuals, strict=True)):
        dimension = dimensionality_of(expression, dimensions)
        if any(name in declarations for name in row.parameters) or any(
            node.head.atom_type is QuantityType for node in subtrees(expression)
        ):
            residual_declarations[str(index)] = PortDeclaration(DIMENSIONLESS if dimension is None else dimension)

    parameters = (*block.unknowns, *sorted(block.inputs))
    name = "__bilt_nonlinear_residual"
    while name in parameters:
        name += "_"
    functions = [tree_to_functiondef(tree, name=name, parameter_order=parameters) for tree in block.residuals]
    function = functions[0]
    values = [typing.cast("ast.Return", definition.body[0]).value for definition in functions]
    function.body = [ast.Return(value=ast.Tuple(elts=typing.cast("list[ast.expr]", values), ctx=ast.Load()))]
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            node.id = mangle_name(node.id)
        elif isinstance(node, ast.arg):
            node.arg = mangle_name(node.arg)
    ast.fix_missing_locations(function)
    try:
        residual = backend.transpile(function)
    except BiltBackendError as error:
        msg = f"The backend could not compile scalar residuals: {error}"
        raise BiltNonlinearBlockError(msg) from error
    return PreparedNonlinearBlock(block, backend, residual, declarations, residual_declarations)


def _require_names(values: Mapping[str, object], expected: Iterable[str], label: str) -> None:
    if set(values) != set(expected):
        message = f"Expected {label} {sorted(expected)!r}; received {sorted(values)!r}."
        raise BiltNonlinearBlockError(message)


def _numeric(values: Mapping[str, object], declarations: dict[str, PortDeclaration]) -> dict[str, float]:
    if any(isinstance(value, Tensor) for value in values.values()):
        message = "Only scalar values are supported in a nonlinear block."
        raise BiltNonlinearBlockError(message)
    validate_inputs("nonlinear block", declarations, values)
    lowered = lower_inputs(dict(values))
    result = {}
    for name, value in lowered.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            message = f"Nonlinear-block value {name!r} must be a finite real scalar."
            raise BiltNonlinearBlockError(message)
        result[name] = float(value)
    return result


def _declarations(block: ResidualBlock) -> dict[str, PortDeclaration]:
    declarations: dict[str, PortDeclaration] = {}
    seen: dict[str, PortDeclaration | None] = {}
    for row in block.rows:
        if any(not hint.is_scalar for _, hint in row.domain_hints):
            message = f"Equation {row.label!r} declares a non-scalar domain."
            raise BiltNonlinearBlockError(message)
        prefix = ".".join((*row.path, "")) if row.path else ""
        for parameter in row.source.__bilt_parameters__:
            name = prefix + parameter
            declaration = row.source.__bilt_port_declarations__.get(parameter)
            if declaration is not None and declaration.over_a_domain:
                message = f"Port {name!r} declares a Tensor; only scalar quantities are supported."
                raise BiltNonlinearBlockError(message)
            if name in seen and seen[name] != declaration:
                message = f"Conflicting nonlinear-block quantity declarations for {name!r}."
                raise BiltNonlinearBlockError(message)
            seen[name] = declaration
            if declaration is not None:
                declarations[name] = declaration
    return declarations
