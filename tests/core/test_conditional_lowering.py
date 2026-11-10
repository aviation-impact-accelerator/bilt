# ruff: noqa: SIM108
import ast
import typing

import jax
import jax.numpy as jnp
import pytest

from bilt._backend._jax import JAX_BACKEND
from bilt._backend._python import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._tensors._expression_lowering import lower_expression
from bilt.core._to_ast import tree_to_functiondef
from bilt.errors import BiltTensorDomainError
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import Domain, domain
from bilt.symbolic._expressions import Expr, conditional_expr
from bilt.symbolic._isolation._isolate import isolate

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


def _expression(x: Expr, a: Expr, b: Expr) -> Expr:
    return conditional_expr(x > 0, a, b)


def test_scalar_conditional_lowers_with_scalar_domain() -> None:
    x, a, b = (Expr.symbol(name) for name in ("x", "a", "b"))
    expression = _expression(x, a, b)
    lowered = lower_expression(
        expression,
        symbol_domains={"x": domain(), "a": domain(), "b": domain()},
        canonical_order=domain(),
    )
    assert lowered.domain.is_scalar
    assert lowered.tree is expression.tree


@pytest.mark.parametrize("non_scalar", ("x", "a", "b"))
def test_scalar_conditional_rejects_tensor_predicate_or_branch(non_scalar: str) -> None:
    x, a, b = (Expr.symbol(name) for name in ("x", "a", "b"))
    dimension = Dim("time")
    symbol_domains = dict.fromkeys(("x", "a", "b"), domain())
    symbol_domains[non_scalar] = domain(dimension)
    with pytest.raises(BiltTensorDomainError, match="scalar"):
        lower_expression(
            _expression(x, a, b),
            symbol_domains=symbol_domains,
            canonical_order=domain(dimension),
        )


def test_relation_predicate_with_arithmetic_lowers() -> None:
    x, threshold, a, b = (Expr.symbol(name) for name in ("x", "threshold", "a", "b"))
    expression = conditional_expr(x + 1 > threshold * 2, a, b)
    lowered = lower_expression(
        expression,
        symbol_domains=dict.fromkeys(("x", "threshold", "a", "b"), domain()),
        canonical_order=domain(),
    )
    assert lowered.domain.is_scalar
    assert lowered.tree is expression.tree


def test_python_conditional_only_evaluates_selected_branch() -> None:
    x = Expr.symbol("x")
    expression = conditional_expr(x > 0, 1 / x, -x)
    lowered = lower_expression(
        expression,
        symbol_domains={"x": Domain.scalar()},
        canonical_order=Domain.scalar(),
    )
    function_ast = tree_to_functiondef(lowered.tree, name="result", parameter_order=("x",))
    assert isinstance(function_ast.body[0], ast.Return)
    assert isinstance(function_ast.body[0].value, ast.IfExp)
    compiled = PYTHON_BACKEND.transpile(function_ast)
    assert compiled.callable(0.0) == pytest.approx(0.0)
    assert compiled.callable(2.0) == pytest.approx(0.5)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_conditional_equation_executes_in_both_backends(backend: Backend) -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        if x > 0:
            result = x**2
        else:
            result = -x
        return y == result

    isolated = isolate(relation.__bilt_equation_tree__, Expr.symbol("y"))
    lowered = lower_expression(
        isolated.rhs,
        symbol_domains={"x": Domain.scalar()},
        canonical_order=Domain.scalar(),
    )
    compiled = backend.transpile(
        tree_to_functiondef(lowered.tree, name="y", parameter_order=relation.__bilt_parameters__)
    )
    assert float(compiled.callable(2.0)) == pytest.approx(4.0)
    assert float(compiled.callable(-3.0)) == pytest.approx(3.0)


def test_jax_compilation_uses_lazy_conditional() -> None:
    x = Expr.symbol("x")
    expression = conditional_expr(x > 0, x**2, -x)
    function_ast = tree_to_functiondef(expression.tree, name="value", parameter_order=("x",))
    compiled = JAX_BACKEND.transpile(function_ast)
    assert float(compiled.callable(jnp.asarray(2.0))) == pytest.approx(4.0)
    assert float(compiled.callable(jnp.asarray(-3.0))) == pytest.approx(3.0)
    assert "cond" in str(jax.make_jaxpr(compiled.callable)(jnp.asarray(2.0)))


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_conditional_equation_model_evaluation(backend: Backend) -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        if x > 0:
            value = x**2
        else:
            value = -x
        return y == value

    analyser = CausalizationStructuralAnalyzer([relation], backend)
    request = EvaluationRequest(output="y", input_domains={"x": Domain.scalar()})
    with EvaluationContext():
        result = analyser.analyse(request)
    compiled = backend.chain_case(result)
    assert float(compiled.callable(x=2.0)) == pytest.approx(4.0)
    assert float(compiled.callable(x=-3.0)) == pytest.approx(3.0)
