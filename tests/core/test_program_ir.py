"""Tests for neutral IR conversion, validation and scheduling."""

import ast

import jax.numpy as jnp
import pytest

from bilt._backend._case import BackendCase
from bilt._backend._jax import JAX_BACKEND
from bilt._backend._python import PYTHON_BACKEND
from bilt.core._ast_to_ir import ast_to_ir
from bilt.core._ir import Block, FunctionIR, Operation, Value, ir_to_functiondef
from bilt.core._ir_validation import validate_function_ir
from bilt.core._program_ir import CallStep, ProgramIR, program_from_case
from bilt.symbolic._domains import Domain


@pytest.mark.parametrize("backend_name", ("python", "jax"))
def test_pure_transform_with_local_assignments_and_scalar_branch(backend_name: str) -> None:
    original = ast.parse(
        "def output(x, y):\n"
        "    scaled = x * 2\n"
        "    adjusted = scaled + y\n"
        "    return adjusted if x > 0 else -adjusted\n"
    ).body[0]
    assert isinstance(original, ast.FunctionDef)
    ir = ast_to_ir(original, output_domain=Domain.scalar())
    assert ir is not None
    assert {op.opcode for op in ir.body.operations} >= {"arith.mul", "arith.add", "scf.if"}
    validate_function_ir(ir)
    backend = PYTHON_BACKEND if backend_name == "python" else JAX_BACKEND
    compiled = backend.transpile(ir_to_functiondef(ir))
    x = jnp.asarray(2.0) if backend_name == "jax" else 2.0
    assert float(compiled.callable(x, 3.0)) == pytest.approx(7.0)
    x = jnp.asarray(-2.0) if backend_name == "jax" else -2.0
    assert float(compiled.callable(x, 3.0)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "source",
    (
        "def output(x):\n    return abs(x)\n",
        "def output(x):\n    return x[0]\n",
        "def output(x):\n    for i in range(2):\n        x += i\n    return x\n",
    ),
)
def test_unsupported_python_constructs_keep_existing_backend_route(source: str) -> None:
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    assert ast_to_ir(function) is None


def test_ir_validator_rejects_undefined_operand() -> None:
    function = FunctionIR(
        name="answer",
        parameters=("x",),
        body=Block(
            operations=(Operation(Value(0), "arith.add", (Value("x"), Value(999))),),
            yield_value=Value(0),
        ),
    )
    with pytest.raises(ValueError, match="not defined"):
        validate_function_ir(function)


def test_ir_validator_rejects_bad_region_and_tensor_axis() -> None:
    bad_region = FunctionIR("answer", ("x",), Block((Operation(Value(0), "scf.if", (Value("x"),)),), Value(0)))
    with pytest.raises(ValueError, match="regions"):
        validate_function_ir(bad_region)
    bad_axis = FunctionIR(
        "answer", ("x",), Block((Operation(Value(0), "bilt.sum", (Value("x"),), (("axis", -1),)),), Value(0))
    )
    with pytest.raises(ValueError, match="nonnegative"):
        validate_function_ir(bad_axis)


def test_program_ir_validates_producer_dependencies() -> None:
    program = ProgramIR(
        parameters=("a",),
        steps=(
            CallStep("b", ("a",), Domain.scalar()),
            CallStep("result", ("b",), Domain.scalar()),
        ),
        result="result",
    )
    program.verify()
    with pytest.raises(ValueError, match="unavailable inputs"):
        ProgramIR(("a",), (CallStep("result", ("missing",)),), "result").verify()


def test_case_schedule_describes_compiled_functions_without_callables() -> None:
    first = PYTHON_BACKEND.transpile(ast.parse("def doubled(a):\n    return a * 2\n").body[0])
    second = PYTHON_BACKEND.transpile(ast.parse("def result(doubled):\n    return doubled + 1\n").body[0])
    case = BackendCase(name="example", parameters=("a",), intermediates=[first, second], return_name="result")
    program = program_from_case(case)
    assert tuple(step.output for step in program.steps) == ("doubled", "result")
    assert program.steps[0].inputs == ("a",)
    expected = 7.0
    assert PYTHON_BACKEND.chain_case(case).callable(a=3.0) == expected
