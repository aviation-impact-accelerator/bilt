import ast
import inspect
import typing

from bilt.core._canonical_ordering import canonicalized
from bilt.core._evaluation_context import current_evaluation
from bilt.core._tensors._errors import BiltTensorDomainError
from bilt.core._tensors._expression_lowering import reduce_dimension, selection_domain
from bilt.core._to_ast import (
    MAX_INTRINSIC_NAME,
    MEAN_INTRINSIC_NAME,
    MIN_INTRINSIC_NAME,
    PROD_INTRINSIC_NAME,
    SUM_INTRINSIC_NAME,
    isel_intrinsic_ast,
    reduction_intrinsic_ast,
)
from bilt.symbolic._atoms import ISEL, SEL
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

    closure = inspect.getclosurevars(function)
    bindings = {
        **closure.globals,
        **closure.nonlocals,
    }

    contains_selection = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"sel", "isel"}
        for node in ast.walk(function_ast)
    )
    if contains_selection:
        return _lower_selection_body(
            function_ast,
            bindings=bindings,
            symbol_domains=symbol_domains,
            canonical_order=canonical_order,
        )

    parameter_names = (argument.arg for argument in _arguments(function_ast))
    output_domain = canonicalized(
        union(*(symbol_domains[name] for name in parameter_names)),
        canonical_order,
    ).domain
    match function_ast.body:
        case [ast.Return(value=ast.expr() as expression) as return_statement]:
            lowered = _lower_expression(
                expression,
                bindings=bindings,
                symbol_domains=symbol_domains,
                canonical_order=canonical_order,
            )
            if lowered is not None:
                lowered_expression, output_domain = lowered
                return_statement.value = ast.copy_location(lowered_expression, expression)

    ast.fix_missing_locations(function_ast)

    return function_ast, output_domain


def _lower_selection_body(
    function_ast: ast.FunctionDef,
    *,
    bindings: collections.abc.Mapping[str, object],
    symbol_domains: collections.abc.Mapping[str, Domain],
    canonical_order: Domain,
) -> tuple[ast.FunctionDef, Domain]:
    """Track domains through straight-line selections, rejecting unsupported flow."""
    statements = function_ast.body
    if ast.get_docstring(function_ast) is not None:
        statements = statements[1:]
    if not statements or not isinstance(statements[-1], ast.Return):
        message = "A transform containing sel/isel must end with a return."
        raise BiltTensorDomainError(message)
    working_domains = dict(symbol_domains)
    output_domain = Domain.scalar()
    for statement in statements:
        match statement:
            case ast.Assign(targets=[ast.Name(id=target)], value=expression):
                pass
            case ast.Return(value=ast.expr() as expression) if statement is statements[-1]:
                target = None
            case _:
                message = "Transforms containing sel/isel require straight-line assignments and a final return."
                raise BiltTensorDomainError(message)
        lowered = _lower_expression(
            expression,
            bindings=bindings,
            symbol_domains=working_domains,
            canonical_order=canonical_order,
        )
        if lowered is None:
            message = "Cannot infer the domain of an expression in a transform containing sel/isel."
            raise BiltTensorDomainError(message)
        statement.value, output_domain = lowered
        if target is not None:
            working_domains[target] = output_domain
    ast.fix_missing_locations(function_ast)
    return function_ast, output_domain


def _lower_expression(  # noqa: PLR0911
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
                bound_value = bindings.get(name)
                if isinstance(bound_value, (int, float)):
                    return ast.Constant(value=bound_value), Domain.scalar()
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

        case ast.UnaryOp(op=ast.USub() | ast.UAdd()):
            operand = _lower_expression(
                node.operand,
                bindings=bindings,
                symbol_domains=symbol_domains,
                canonical_order=canonical_order,
            )
            if operand is None:
                return None
            operand_ast, operand_domain = operand
            return ast.UnaryOp(op=node.op, operand=operand_ast), operand_domain

        case ast.Call():
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"sel", "isel"}:
                return _lower_selection(
                    node,
                    bindings=bindings,
                    symbol_domains=symbol_domains,
                    canonical_order=canonical_order,
                )
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


def _lower_selection(
    node: ast.Call,
    *,
    bindings: collections.abc.Mapping[str, object],
    symbol_domains: collections.abc.Mapping[str, Domain],
    canonical_order: Domain,
) -> tuple[ast.expr, Domain]:
    attribute = typing.cast("ast.Attribute", node.func)
    expected_arguments = 2
    if len(node.args) != expected_arguments or node.keywords:
        message = f"Tensor.{attribute.attr}() requires a dimension and a scalar selector as positional arguments."
        raise BiltTensorDomainError(message)
    dim = _resolve_dimension(node.args[0], bindings=bindings)
    value = _lower_expression(
        attribute.value,
        bindings=bindings,
        symbol_domains=symbol_domains,
        canonical_order=canonical_order,
    )
    if value is None:
        message = "Cannot infer the domain of the value being selected."
        raise BiltTensorDomainError(message)
    value_ast, value_domain = value
    if attribute.attr == "sel":
        selector = node.args[1]
        try:
            coordinate = bindings[selector.id] if isinstance(selector, ast.Name) else ast.literal_eval(selector)
        except (KeyError, ValueError, TypeError) as error:
            message = "sel requires a literal coordinate or a bound coordinate constant."
            raise BiltTensorDomainError(message) from error
        output_domain, axis = selection_domain(
            value_domain,
            dim,
            Domain.scalar(),
            operator=SEL,
            canonical_order=canonical_order,
        )
        selector_ast = ast.Constant(value=current_evaluation().domains.position(dim, coordinate))
    else:
        index = _lower_expression(
            node.args[1],
            bindings=bindings,
            symbol_domains=symbol_domains,
            canonical_order=canonical_order,
        )
        if index is None:
            message = "Cannot infer the domain of the positional selector."
            raise BiltTensorDomainError(message)
        selector_ast, selector_domain = index
        output_domain, axis = selection_domain(
            value_domain,
            dim,
            selector_domain,
            operator=ISEL,
            canonical_order=canonical_order,
        )
    return isel_intrinsic_ast(value_ast, selector_ast, axis), output_domain


def _lower_reduction(
    node: ast.Call,
    *,
    bindings: collections.abc.Mapping[str, object],
    symbol_domains: collections.abc.Mapping[str, Domain],
    canonical_order: Domain,
) -> tuple[ast.expr, Domain] | None:
    if not isinstance(node.func, ast.Attribute):
        return None

    try:
        intrinsic_name = _REDUCTION_INTRINSICS[node.func.attr]
    except KeyError:
        return None

    if len(node.args) != 1 or node.keywords:
        message = f"Tensor.{node.func.attr}() requires exactly one named dimension."
        raise NotImplementedError(message)

    value = _lower_expression(
        node.func.value,
        bindings=bindings,
        symbol_domains=symbol_domains,
        canonical_order=canonical_order,
    )
    if value is None:
        if isinstance(node.func.value, ast.Name):
            message = f"No domain is known for transform parameter {node.func.value.id!r}."
            raise BiltTensorDomainError(message)
        message = "No domain is known for the transform reduction operand."
        raise BiltTensorDomainError(message)
    value_node, value_domain = value

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
        value_node,
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
