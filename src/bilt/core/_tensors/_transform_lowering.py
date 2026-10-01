import ast
import inspect
import typing

from bilt.core._canonical_ordering import canonicalized
from bilt.core._evaluation_context import current_evaluation
from bilt.core._tensors._errors import BiltTensorDomainError
from bilt.core._tensors._expression_lowering import reduce_dimension
from bilt.core._to_ast import (
    MAX_INTRINSIC_NAME,
    MEAN_INTRINSIC_NAME,
    MIN_INTRINSIC_NAME,
    PROD_INTRINSIC_NAME,
    SUM_INTRINSIC_NAME,
    reduction_intrinsic_ast,
)
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import Domain, union

if typing.TYPE_CHECKING:
    import collections.abc


_REDUCTION_INTRINSICS = {
    "mean": MEAN_INTRINSIC_NAME,
    "sum": SUM_INTRINSIC_NAME,
    "min": MIN_INTRINSIC_NAME,
    "max": MAX_INTRINSIC_NAME,
    "prod": PROD_INTRINSIC_NAME,
}


def lower_transform(
    function_ast: ast.FunctionDef,
    *,
    function: collections.abc.Callable[..., object],
    symbol_domains: collections.abc.Mapping[str, Domain],
) -> tuple[ast.FunctionDef, Domain]:
    """Lower named tensor operations in a transform to backend intrinsics."""
    canonical_order = current_evaluation().canonical_order
    function_ast.decorator_list = []
    _strip_annotations(function_ast)

    parameter_names = (argument.arg for argument in _arguments(function_ast))
    fallback_domain = canonicalized(
        union(*(symbol_domains[name] for name in parameter_names)),
        canonical_order,
    ).domain

    closure = inspect.getclosurevars(function)
    bindings = {
        **closure.globals,
        **closure.nonlocals,
    }

    if len(function_ast.body) != 1 or not isinstance(function_ast.body[0], ast.Return):
        ast.fix_missing_locations(function_ast)
        return function_ast, fallback_domain

    return_statement = function_ast.body[0]

    if return_statement.value is None:
        ast.fix_missing_locations(function_ast)
        return function_ast, fallback_domain

    lowered = _lower_expression(
        return_statement.value,
        bindings=bindings,
        symbol_domains=symbol_domains,
        canonical_order=canonical_order,
    )

    if lowered is None:
        ast.fix_missing_locations(function_ast)
        return function_ast, fallback_domain

    expression, output_domain = lowered

    return_statement.value = ast.copy_location(
        expression,
        return_statement.value,
    )

    ast.fix_missing_locations(function_ast)

    return function_ast, output_domain


def _lower_expression(
    node: ast.expr,
    *,
    bindings: collections.abc.Mapping[str, object],
    symbol_domains: collections.abc.Mapping[str, Domain],
    canonical_order: Domain,
) -> tuple[ast.expr, Domain] | None:
    match node:
        case ast.Name(id=name):
            try:
                value_domain = symbol_domains[name]
            except KeyError:
                return None

            return node, value_domain

        case ast.Constant():
            return node, Domain.scalar()

        case ast.BinOp():
            return _lower_binop(
                node,
                bindings=bindings,
                symbol_domains=symbol_domains,
                canonical_order=canonical_order,
            )

        case ast.Call():
            return _lower_reduction(
                node,
                bindings=bindings,
                symbol_domains=symbol_domains,
                canonical_order=canonical_order,
            )

        case _:
            return None


def _lower_binop(
    node: ast.BinOp,
    *,
    bindings: collections.abc.Mapping[str, object],
    symbol_domains: collections.abc.Mapping[str, Domain],
    canonical_order: Domain,
) -> tuple[ast.expr, Domain] | None:
    left = _lower_expression(
        node.left,
        bindings=bindings,
        symbol_domains=symbol_domains,
        canonical_order=canonical_order,
    )
    right = _lower_expression(
        node.right,
        bindings=bindings,
        symbol_domains=symbol_domains,
        canonical_order=canonical_order,
    )

    if left is None or right is None:
        return None

    left_node, left_domain = left
    right_node, right_domain = right

    output_domain = canonicalized(
        union(left_domain, right_domain),
        canonical_order,
    ).domain

    return (
        ast.BinOp(
            left=left_node,
            op=node.op,
            right=right_node,
        ),
        output_domain,
    )


def _lower_reduction(
    node: ast.Call,
    *,
    bindings: collections.abc.Mapping[str, object],
    symbol_domains: collections.abc.Mapping[str, Domain],
    canonical_order: Domain,
) -> tuple[ast.expr, Domain] | None:
    if not (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)):
        return None

    try:
        intrinsic_name = _REDUCTION_INTRINSICS[node.func.attr]
    except KeyError:
        return None

    if len(node.args) != 1 or node.keywords:
        message = f"Tensor.{node.func.attr}() requires exactly one named dimension."
        raise NotImplementedError(message)

    value_name = node.func.value.id

    try:
        value_domain = symbol_domains[value_name]
    except KeyError:
        message = f"No domain is known for transform parameter {value_name!r}."
        raise BiltTensorDomainError(message) from None

    dim = _resolve_dimension(
        node.args[0],
        bindings=bindings,
    )

    output_domain, axis = reduce_dimension(
        value_domain,
        dim,
        canonical_order=canonical_order,
    )

    lowered = reduction_intrinsic_ast(
        intrinsic_name,
        ast.Name(
            id=value_name,
            ctx=ast.Load(),
        ),
        axis,
    )

    return ast.copy_location(lowered, node), output_domain


def _resolve_dimension(
    node: ast.expr,
    *,
    bindings: collections.abc.Mapping[str, object],
) -> Dim:
    if not isinstance(node, ast.Name):
        message = "Tensor reductions currently require a named dimension."
        raise NotImplementedError(message)

    try:
        dim = bindings[node.id]
    except KeyError:
        message = f"Cannot resolve dimension {node.id!r} in transform."
        raise NotImplementedError(message) from None

    if not isinstance(dim, Dim):
        message = f"Transform name {node.id!r} does not refer to a Bilt dimension."
        raise TypeError(message)

    return dim


def _arguments(function_ast: ast.FunctionDef) -> tuple[ast.arg, ...]:
    return (
        *function_ast.args.posonlyargs,
        *function_ast.args.args,
        *function_ast.args.kwonlyargs,
    )


def _strip_annotations(function_ast: ast.FunctionDef) -> None:
    for argument in _arguments(function_ast):
        argument.annotation = None

    if function_ast.args.vararg is not None:
        function_ast.args.vararg.annotation = None

    if function_ast.args.kwarg is not None:
        function_ast.args.kwarg.annotation = None

    function_ast.returns = None
