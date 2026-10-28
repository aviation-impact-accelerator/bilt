"""Lower pointwise batches and joint tensor blocks to backend linear solves."""

import ast
import collections
import dataclasses
import typing

from bilt._backend import LinearSolveBackend
from bilt._backend._errors import BiltBackendError
from bilt._backend._linear_solve import (
    LINEAR_BASIS_INTRINSIC_NAME,
    LINEAR_EXPAND_INTRINSIC_NAME,
    LINEAR_SOLVE_INTRINSIC_NAME,
)
from bilt.core._errors import BiltDomainBindingError
from bilt.core._evaluation_context import current_evaluation
from bilt.core._port import mangle_name
from bilt.core._structural_analysis._domain_inference import infer_linear_domains
from bilt.core._structural_analysis._linear_block import (
    BiltLinearBlockError,
    LinearBlock,
    extract_qualified_linear_block,
)
from bilt.core._structural_analysis._linear_domains import declared_linear_domains
from bilt.core._structural_analysis._linear_operator import (
    LinearOperatorBlock,
    extract_supported_linear_block,
    linear_operator_rows,
)
from bilt.core._structural_analysis._linear_operator_lowering import OperatorLowering, expression_ast
from bilt.core._structural_analysis._qualified_block import QualifiedBlock
from bilt.core._tensors._expression_lowering import LoweredExpression, lower_expression
from bilt.symbolic._domains import Domain, union
from bilt.symbolic._expressions import Expr

if typing.TYPE_CHECKING:
    from bilt._backend import Backend, BackendCallable
    from bilt.core._structural_analysis._plan import PlanBlock
    from bilt.symbolic._tree import Tree


type Lowered = tuple[ast.expr, Domain]


def realize_linear_block(
    block: PlanBlock | QualifiedBlock,
    backend: Backend,
    symbol_domains: dict[str, Domain],
    *,
    solution_name: str,
) -> tuple[BackendCallable, ...]:
    """Compile one solve, preserving each output's named tensor domain."""
    if not isinstance(backend, LinearSolveBackend):
        message = "The selected backend does not support linear blocks."
        raise BiltLinearBlockError(message)
    if isinstance(block, QualifiedBlock):
        if any(symbol_domains[name].dims for name in block.inputs) or any(hint.dims for _, hint in block.domain_hints):
            message = "Cross-boundary linear blocks currently require scalar inputs and unknowns."
            raise BiltLinearBlockError(message)
        linear = extract_qualified_linear_block(block)
    else:
        linear = extract_supported_linear_block(block)
    lowering = _LinearLowering(linear, symbol_domains, solution_name)
    batch_rank = len(lowering.canonical.dims)
    rhs = (
        [lowering.lower(tree, symbol_domains) for tree in lowering.linear.rhs]
        if isinstance(lowering.linear, LinearBlock)
        else []
    )
    coefficients = (
        [[lowering.lower(tree, symbol_domains) for tree in row] for row in lowering.linear.coefficients]
        if isinstance(lowering.linear, LinearBlock)
        else []
    )
    domains, pointwise = lowering.output_domains(coefficients, rhs)
    vector = ast.Tuple(elts=[_names(value, lowering.parameters) for value, _ in rhs], ctx=ast.Load())
    if pointwise:
        matrix = ast.Tuple(
            elts=[
                ast.Tuple(elts=[_names(value, lowering.parameters) for value, _ in row], ctx=ast.Load())
                for row in coefficients
            ],
            ctx=ast.Load(),
        )
        assignments = []
        keywords = [ast.keyword(arg="batch_rank", value=ast.Constant(batch_rank))] if batch_rank else []
    else:
        matrix, vector, assignments, keywords = lowering.joint(domains, rhs)
    solve = _function(
        solution_name,
        tuple(map(mangle_name, lowering.parameters)),
        ast.Call(
            func=ast.Name(id=LINEAR_SOLVE_INTRINSIC_NAME, ctx=ast.Load()),
            args=[matrix, vector],
            keywords=keywords,
        ),
    )
    solve.body[:0] = assignments
    try:
        callables = [backend.transpile_linear_solve(solve)]
        for index, output in enumerate(lowering.linear.unknowns):
            slice_ = (
                ast.Tuple(elts=[ast.Constant(Ellipsis), ast.Constant(index)], ctx=ast.Load())
                if pointwise and batch_rank
                else ast.Constant(index)
            )
            projection = _function(
                mangle_name(output),
                (solution_name,),
                ast.Subscript(value=ast.Name(id=solution_name, ctx=ast.Load()), slice=slice_, ctx=ast.Load()),
            )
            callables.append(dataclasses.replace(backend.transpile(projection), output_domain=domains[output]))
    except BiltBackendError as error:
        message = f"The backend could not compile the linear block for {lowering.linear.unknowns!r}: {error}"
        raise BiltLinearBlockError(message) from error
    return tuple(callables)


@dataclasses.dataclass
class _LinearLowering:
    linear: LinearBlock | LinearOperatorBlock
    symbol_domains: dict[str, Domain]
    solution_name: str

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(sorted(self.linear.inputs))

    @property
    def canonical(self) -> Domain:
        return current_evaluation().canonical_order

    def lowered(self, tree: Tree, domains: dict[str, Domain]) -> LoweredExpression:
        try:
            return lower_expression(Expr(tree), symbol_domains=domains, canonical_order=self.canonical)
        except BiltDomainBindingError as error:
            message = f"Cannot lower a linear block with the current coordinate bindings: {error}"
            raise BiltLinearBlockError(message) from error

    def lower(self, tree: Tree, domains: dict[str, Domain]) -> Lowered:
        lowered = self.lowered(tree, domains)
        return expression_ast(lowered.tree), lowered.domain

    def output_domains(self, coefficients: list[list[Lowered]], rhs: list[Lowered]) -> tuple[dict[str, Domain], bool]:
        known = union(*(self.symbol_domains[name] for name in self.parameters))
        if any(dim not in self.canonical.dims for dim in known.dims):
            message = "Linear-block input dimensions must be registered in the evaluation context."
            raise BiltLinearBlockError(message)
        hints = (
            dict(self.linear.block.domain_hints)
            if isinstance(self.linear.block, QualifiedBlock)
            else declared_linear_domains(self.linear.block)
        )
        for name in hints.keys() & self.symbol_domains.keys():
            if set(hints[name].dims) != set(self.symbol_domains[name].dims):
                message = f"Input {name!r} does not match its linear-block Domain hint."
                raise BiltLinearBlockError(message)
        common = union(*(domain for row in coefficients for _, domain in row), *(domain for _, domain in rhs))
        common = self.ordered(common)
        result = {}
        for name in self.linear.unknowns:
            domain = hints.get(name, common)
            if any(dim not in known.dims for dim in domain.dims):
                message = f"The domain of {name!r} must use dimensions present in the block inputs."
                raise BiltLinearBlockError(message)
            result[name] = self.ordered(domain)
        pointwise = isinstance(self.linear, LinearBlock) and all(domain == common for domain in result.values())
        if not pointwise and any(name not in hints for name in self.linear.unknowns):
            result = infer_linear_domains(
                self.linear.unknowns,
                self.linear.equations
                if isinstance(self.linear, LinearOperatorBlock)
                else tuple(zip(linear_operator_rows(self.linear), self.linear.rhs, strict=True)),
                {name: self.symbol_domains[name] for name in self.parameters},
                hints,
                self.canonical,
            )
            pointwise = isinstance(self.linear, LinearBlock) and all(domain == common for domain in result.values())
        return result, pointwise

    def ordered(self, domain: Domain) -> Domain:
        return Domain(tuple(dim for dim in self.canonical.dims if dim in domain.dims))

    def axes(self, domain: Domain) -> tuple[int, ...]:
        return tuple(i for i, dim in enumerate(self.canonical.dims) if dim in domain.dims)

    def local_name(self, suffix: str) -> str:
        name = self.solution_name + suffix
        while name in set(map(mangle_name, self.parameters)):
            name += "_"
        return name

    def operator_values(
        self, linear: LinearOperatorBlock, domains: dict[str, Domain]
    ) -> tuple[list[ast.expr], list[ast.expr], list[Domain], list[ast.stmt]]:
        builder = OperatorLowering(
            linear.unknowns, self.parameters, self.local_name("_basis"), self.local_name("_zeros")
        )
        rows: list[ast.expr] = []
        constants: list[ast.expr] = []
        row_domains = []
        for left, right in linear.equations:
            lhs = self.lowered(left, self.symbol_domains | domains)
            rhs = self.lowered(right, self.symbol_domains | domains)
            first, second = builder.split(lhs.tree), builder.split(rhs.tree)
            rows.append(ast.BinOp(left=first.linear, op=ast.Sub(), right=second.linear))
            constants.append(ast.BinOp(left=second.constant, op=ast.Sub(), right=first.constant))
            row_domains.append(union(lhs.domain, rhs.domain))
        return rows, constants, row_domains, [builder.zeros()]

    def joint(
        self, domains: dict[str, Domain], rhs: list[Lowered]
    ) -> tuple[ast.Tuple, ast.Tuple, list[ast.stmt], list[ast.keyword]]:
        basis_name = self.local_name("_basis")
        if isinstance(self.linear, LinearOperatorBlock):
            rows, constants, row_domains, extra = self.operator_values(self.linear, domains)
        else:
            lowered = [self.lower(tree, self.symbol_domains | domains) for tree in linear_operator_rows(self.linear)]
            rows = [_names(value, self.parameters, self.linear.unknowns, basis_name) for value, _ in lowered]
            row_domains = [union(domain, rhs_domain) for (_, domain), (_, rhs_domain) in zip(lowered, rhs, strict=True)]
            constants = [_names(value, self.parameters) for value, _ in rhs]
            extra = []
        if collections.Counter(domains.values()) != collections.Counter(self.ordered(domain) for domain in row_domains):
            message = "Tensor block scalar equations must have the same domain layout as its unknowns."
            raise BiltLinearBlockError(message)
        layout = (
            tuple(self.axes(domains[name]) for name in self.linear.unknowns),
            tuple(map(self.axes, row_domains)),
            len(self.canonical.dims),
        )
        layout_ast = ast.parse(repr(layout), mode="eval").body
        values = ast.Tuple(
            elts=[ast.Name(id=mangle_name(name), ctx=ast.Load()) for name in self.parameters], ctx=ast.Load()
        )
        assignments: list[ast.stmt] = [
            ast.Assign(
                targets=[ast.Name(id=basis_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id=LINEAR_BASIS_INTRINSIC_NAME, ctx=ast.Load()),
                    args=[values, layout_ast],
                    keywords=[],
                ),
            )
        ]
        assignments.extend(extra)
        matrix = ast.Tuple(elts=rows, ctx=ast.Load())
        keywords = [ast.keyword(arg="layout", value=layout_ast), ast.keyword(arg="values", value=values)]
        return matrix, ast.Tuple(elts=constants, ctx=ast.Load()), assignments, keywords


def _names(
    expression: ast.expr, parameters: tuple[str, ...], unknowns: tuple[str, ...] = (), basis: str = ""
) -> ast.expr:
    class Names(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            if node.id in unknowns:
                return ast.Subscript(
                    value=ast.Name(id=basis, ctx=ast.Load()),
                    slice=ast.Constant(unknowns.index(node.id)),
                    ctx=ast.Load(),
                )
            if node.id in parameters:
                name = ast.Name(id=mangle_name(node.id), ctx=ast.Load())
                if unknowns:
                    return ast.Call(
                        func=ast.Name(id=LINEAR_EXPAND_INTRINSIC_NAME, ctx=ast.Load()), args=[name], keywords=[]
                    )
                return name
            return node

    return typing.cast("ast.expr", Names().visit(expression))


def _function(name: str, parameters: tuple[str, ...], expression: ast.expr) -> ast.FunctionDef:
    function = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=parameter) for parameter in parameters],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=expression)],
        decorator_list=[],
    )
    ast.fix_missing_locations(function)
    return function
