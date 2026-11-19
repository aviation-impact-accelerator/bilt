"""Contract tests independent of any compilation backend."""

import dataclasses

import pytest

from bilt.core._ir import (
    Binding,
    Block,
    ChildStep,
    FunctionIR,
    FunctionStep,
    IRValidationError,
    LinearSolveStep,
    Opcode,
    Operation,
    ProgramIR,
    ProjectionStep,
    Source,
    Value,
    ValueKind,
    ValueType,
    validate_function,
    validate_program,
)
from bilt.symbolic._dimensions import dim
from bilt.symbolic._domains import Domain, domain
from bilt.units._metadata import Metadata
from bilt.units.dimensions._dimensions import ENERGY, POWER, TIME

FLOAT = ValueType(ValueKind.FLOAT)
BOOL = ValueType(ValueKind.BOOLEAN)
OPAQUE = ValueType(ValueKind.OPAQUE)


def example_function() -> FunctionIR:
    """A directed equation, isolated into scalar operations."""
    return FunctionIR(
        "sum",
        (Value("a", FLOAT), Value("b", FLOAT)),
        Block((Operation(Value("sum", FLOAT), Opcode.ADD, ("a", "b")),), "sum"),
        Source(("engine",), "sum_equation"),
    )


def test_golden_ir_contract_for_isolated_equation() -> None:
    function = example_function()
    validate_function(function)
    assert (
        function.name,
        tuple(value.name for value in function.parameters),
        tuple((op.result.name, op.opcode.value, op.operands) for op in function.body.operations),
        function.body.yield_name,
        function.source,
    ) == (
        "sum",
        ("a", "b"),
        (("sum", "arith.add", ("a", "b")),),
        "sum",
        Source(("engine",), "sum_equation"),
    )


def test_immutable_types_preserve_scientific_dimensions_and_metadata() -> None:
    time = dim("time")
    typed = ValueType(ValueKind.FLOAT, domain(time), POWER, Metadata("co2"))
    same = ValueType(ValueKind.FLOAT, domain(time), POWER, Metadata("co2"))
    assert typed == same
    assert typed.dimension != ENERGY
    assert typed.domain != Domain.scalar()
    with pytest.raises(dataclasses.FrozenInstanceError):
        typed.__setattr__("kind", ValueKind.BOOLEAN)


def test_scalar_conditional_has_lexical_regions() -> None:
    function = FunctionIR(
        "sign",
        (Value("predicate", BOOL), Value("x", FLOAT)),
        Block(
            (
                Operation(
                    Value("result", FLOAT),
                    Opcode.IF,
                    ("predicate",),
                    regions=(
                        Block((Operation(Value("positive", FLOAT), Opcode.NEG, ("x",)),), "positive"),
                        Block((), "x"),
                    ),
                ),
            ),
            "result",
        ),
    )
    validate_function(function)


def test_tensor_reductions_and_elementwise_where_have_distinct_contracts() -> None:
    time = dim("time")
    tensor = ValueType(ValueKind.FLOAT, domain(time), POWER)
    mask = ValueType(ValueKind.BOOLEAN, domain(time))
    function = FunctionIR(
        "total",
        (Value("x", tensor), Value("y", tensor), Value("select", mask)),
        Block(
            (
                Operation(Value("chosen", tensor), Opcode.WHERE, ("select", "x", "y")),
                Operation(Value("sum", tensor), Opcode.SUM, ("chosen",), attributes=(("axis", 0),)),
            ),
            "sum",
        ),
    )
    validate_function(function)


@pytest.mark.parametrize(
    ("function", "message"),
    (
        (
            FunctionIR(
                "invalid", (Value("x", FLOAT),), Block((Operation(Value("y", FLOAT), Opcode.ADD, ("x", "z")),), "y")
            ),
            "Undefined operand",
        ),
        (
            FunctionIR(
                "invalid", (Value("x", FLOAT),), Block((Operation(Value("x", FLOAT), Opcode.NEG, ("x",)),), "x")
            ),
            "Duplicate SSA",
        ),
        (
            FunctionIR(
                "invalid",
                (Value("x", FLOAT),),
                Block(
                    (Operation(Value("y", FLOAT), Opcode.CONSTANT, attributes=(("value", 1.0), ("value", 2.0))),), "y"
                ),
            ),
            "Duplicate operation attribute",
        ),
        (
            FunctionIR(
                "invalid",
                (Value("x", FLOAT),),
                Block((Operation(Value("y", FLOAT), Opcode.IF, ("x",), regions=(Block((), "x"),)),), "y"),
            ),
            "two regions",
        ),
        (
            FunctionIR(
                "invalid",
                (Value("x", FLOAT),),
                Block((Operation(Value("y", FLOAT), Opcode.NEG, ("x",)),), "missing"),
            ),
            "Undefined yield",
        ),
    ),
)
def test_function_validation_rejects_malformed_ir(function: FunctionIR, message: str) -> None:
    with pytest.raises(IRValidationError, match=message):
        validate_function(function)


def test_region_result_does_not_escape_its_scope() -> None:
    function = FunctionIR(
        "invalid",
        (Value("flag", BOOL), Value("x", FLOAT)),
        Block(
            (
                Operation(
                    Value("result", FLOAT),
                    Opcode.IF,
                    ("flag",),
                    regions=(
                        Block((Operation(Value("private", FLOAT), Opcode.NEG, ("x",)),), "private"),
                        Block((), "x"),
                    ),
                ),
                Operation(Value("leak", FLOAT), Opcode.NEG, ("private",)),
            ),
            "leak",
        ),
    )
    with pytest.raises(IRValidationError, match="Undefined operand"):
        validate_function(function)


def test_function_call_program_is_complete_and_ordered() -> None:
    function = example_function()
    program = ProgramIR(
        (Value("left", FLOAT), Value("right", FLOAT)),
        (
            FunctionStep(
                Value("answer", FLOAT),
                function,
                (Binding("left", "a"), Binding("right", "b")),
            ),
        ),
        "answer",
    )
    validate_program(program)
    assert isinstance(program.steps[0], FunctionStep)
    assert program.steps[0].output.name == program.return_name


def test_child_call_uses_explicit_qualified_bindings() -> None:
    program = ProgramIR(
        (Value("source", FLOAT),),
        (
            ChildStep(
                Value("unit__value", FLOAT),
                ("unit",),
                "value",
                (Value("source", FLOAT),),
                (Binding("source", "source"),),
            ),
        ),
        "unit__value",
    )
    validate_program(program)


def test_coupled_solve_and_projection_are_explicit() -> None:
    coefficients = FunctionIR("coefficients", (Value("known", FLOAT),), Block((), "known"))
    rhs = FunctionIR("rhs", (Value("known", FLOAT),), Block((), "known"))
    program = ProgramIR(
        (Value("input", FLOAT),),
        (
            LinearSolveStep(
                Value("solution", OPAQUE),
                (Value("x", FLOAT), Value("y", FLOAT)),
                (Value("known", FLOAT),),
                (Binding("input", "known"),),
                coefficients,
                rhs,
            ),
            ProjectionStep(Value("x", FLOAT), "solution", "x"),
            ProjectionStep(Value("y", FLOAT), "solution", "y"),
        ),
        "x",
    )
    validate_program(program)


def test_rejected_undefined_program_input_and_wrong_projection() -> None:
    bad = ProgramIR(
        (Value("x", FLOAT),),
        (FunctionStep(Value("y", FLOAT), example_function(), (Binding("x", "a"), Binding("missing", "b"))),),
        "y",
    )
    with pytest.raises(IRValidationError, match="Undefined argument"):
        validate_program(bad)

    bad_projection = ProgramIR(
        (),
        (ProjectionStep(Value("x", FLOAT), "not_a_solution", "x"),),
        "x",
    )
    with pytest.raises(IRValidationError, match="Unknown linear solution"):
        validate_program(bad_projection)


def test_physical_dimension_mismatch_and_axis_errors() -> None:
    mismatch = FunctionIR(
        "invalid",
        (
            Value("energy", ValueType(ValueKind.FLOAT, Domain.scalar(), ENERGY)),
            Value("time", ValueType(ValueKind.FLOAT, Domain.scalar(), TIME)),
        ),
        Block(
            (
                Operation(
                    Value("result", ValueType(ValueKind.FLOAT, Domain.scalar(), ENERGY)), Opcode.ADD, ("energy", "time")
                ),
            ),
            "result",
        ),
    )
    with pytest.raises(IRValidationError, match="incompatible value types"):
        validate_function(mismatch)

    time = dim("time")
    tensor = ValueType(ValueKind.FLOAT, domain(time))
    bad_axis = FunctionIR(
        "invalid",
        (Value("x", tensor),),
        Block((Operation(Value("y", tensor), Opcode.SUM, ("x",), attributes=(("axis", 2),)),), "y"),
    )
    with pytest.raises(IRValidationError, match="Invalid positional tensor axis"):
        validate_function(bad_axis)
