"""Parity and structure checks for the experimental neutral equation IR."""

import ast

import jax.numpy as jnp
import pytest

from bilt._backend._jax import JAX_BACKEND
from bilt._backend._python import PYTHON_BACKEND
from bilt.core._ir import ir_to_functiondef, tree_to_ir
from bilt.core._tensors._expression_lowering import lower_expression
from bilt.symbolic._atoms import CONDITIONAL, WHERE
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import Domain, domain
from bilt.symbolic._expressions import Expr, conditional_expr


@pytest.mark.parametrize("backend_name", ("python", "jax"))
def test_arithmetic_ir_compiles_on_existing_backends(backend_name: str) -> None:
    a, b, c = (Expr.symbol(name) for name in ("a", "b", "c"))
    lowered = lower_expression(
        a + b * c,
        symbol_domains=dict.fromkeys(("a", "b", "c"), Domain.scalar()),
        canonical_order=Domain.scalar(),
    )
    ir = tree_to_ir(
        lowered.tree,
        name="result",
        parameter_order=("a", "b", "c", "result"),
        output_domain=lowered.domain,
    )
    assert ir.parameters == ("a", "b", "c")
    assert ir.output_domain == Domain.scalar()
    assert all(not isinstance(op, ast.AST) for op in ir.body.operations)
    assert {op.opcode for op in ir.body.operations} >= {"arith.add", "arith.mul"}

    backend = PYTHON_BACKEND if backend_name == "python" else JAX_BACKEND
    fn = backend.transpile(ir_to_functiondef(ir))
    assert float(fn.callable(2.0, 3.0, 4.0)) == pytest.approx(14.0)


@pytest.mark.parametrize("backend_name", ("python", "jax"))
def test_conditional_ir_has_lazy_nested_regions(backend_name: str) -> None:
    x = Expr.symbol("x")
    expression = conditional_expr(x > 0, 1 / x, -x)
    lowered = lower_expression(
        expression,
        symbol_domains={"x": Domain.scalar()},
        canonical_order=Domain.scalar(),
    )
    assert lowered.tree.head is CONDITIONAL
    ir = tree_to_ir(lowered.tree, name="answer", parameter_order=("x",))
    condition = next(op for op in ir.body.operations if op.opcode == "scf.if")
    assert len(condition.regions) == 2
    assert all(region.operations for region in condition.regions)
    assert all(not isinstance(op, ast.AST) for region in condition.regions for op in region.operations)

    backend = PYTHON_BACKEND if backend_name == "python" else JAX_BACKEND
    fn = backend.transpile(ir_to_functiondef(ir))
    assert float(fn.callable(2.0)) == pytest.approx(0.5)
    # JAX traces both regions, so only numerical parity (not Python's
    # short-circuit behaviour) is asserted for the zero branch.
    if backend_name == "python":
        assert fn.callable(0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("backend_name", ("python", "jax"))
def test_tensor_where_is_not_mistaken_for_lazy_scalar_branch(backend_name: str) -> None:
    x, a, b = (Expr.symbol(name) for name in ("x", "a", "b"))
    time = Dim("time")
    lowered = lower_expression(
        conditional_expr(x > 0, a, b),
        symbol_domains={"x": domain(time), "a": domain(time), "b": domain(time)},
        canonical_order=domain(time),
    )
    assert lowered.tree.head is WHERE
    ir = tree_to_ir(lowered.tree, name="answer", parameter_order=("x", "a", "b"), output_domain=lowered.domain)
    assert "bilt.where" in {op.opcode for op in ir.body.operations}
    assert "scf.if" not in {op.opcode for op in ir.body.operations}

    backend = PYTHON_BACKEND if backend_name == "python" else JAX_BACKEND
    fn = backend.transpile(ir_to_functiondef(ir))
    arr = jnp.asarray if backend_name == "jax" else __import__("numpy").asarray
    result = fn.callable(arr([-1.0, 2.0]), arr([10.0, 20.0]), arr([30.0, 40.0]))
    assert list(result) == pytest.approx([30.0, 20.0])


@pytest.mark.parametrize("backend_name", ("python", "jax"))
def test_named_reduction_becomes_positional_ir_operation(backend_name: str) -> None:
    time = Dim("time")
    x = Expr.symbol("x")
    lowered = lower_expression(
        x.sum(time),
        symbol_domains={"x": domain(time)},
        canonical_order=domain(time),
    )
    ir = tree_to_ir(lowered.tree, name="answer", parameter_order=("x",), output_domain=lowered.domain)
    reduction = next(op for op in ir.body.operations if op.opcode == "bilt.sum")
    assert dict(reduction.attributes) == {"axis": 0, "keepdims": True}
    assert ir.output_domain == lowered.domain
    backend = PYTHON_BACKEND if backend_name == "python" else JAX_BACKEND
    fn = backend.transpile(ir_to_functiondef(ir))
    arr = jnp.asarray if backend_name == "jax" else __import__("numpy").asarray
    result = fn.callable(arr([1.0, 2.0, 3.0]))
    assert list(result) == pytest.approx([6.0])
