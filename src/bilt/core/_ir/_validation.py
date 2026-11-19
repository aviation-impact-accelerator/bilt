"""Verify the backend-neutral IR before a backend is allowed to lower it."""

import typing

from bilt.core._ir._nodes import (
    Binding,
    Block,
    ChildStep,
    FunctionIR,
    FunctionStep,
    LinearSolveStep,
    Opcode,
    Operation,
    ProgramIR,
    Value,
    ValueKind,
    ValueType,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class IRValidationError(ValueError):
    """A malformed or inconsistent backend-neutral computation."""


_ARITY: dict[Opcode, int] = {
    Opcode.CONSTANT: 0,
    Opcode.ADD: 2,
    Opcode.SUB: 2,
    Opcode.MUL: 2,
    Opcode.DIV: 2,
    Opcode.POW: 2,
    Opcode.NEG: 1,
    Opcode.INV: 1,
    Opcode.EQ: 2,
    Opcode.NE: 2,
    Opcode.LT: 2,
    Opcode.LE: 2,
    Opcode.GT: 2,
    Opcode.GE: 2,
    Opcode.WHERE: 3,
    Opcode.MASK_AND: 2,
    Opcode.MASK_OR: 2,
    Opcode.MASK_NOT: 1,
    Opcode.SUM: 1,
    Opcode.MEAN: 1,
    Opcode.MIN: 1,
    Opcode.MAX: 1,
    Opcode.PROD: 1,
    Opcode.ISEL: 2,
    Opcode.IF: 1,
}
_COMPARE = {Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.LE, Opcode.GT, Opcode.GE}
_MASK = {Opcode.MASK_AND, Opcode.MASK_OR, Opcode.MASK_NOT}
_ADDITIVE = {Opcode.ADD, Opcode.SUB}
_MULTIPLICATIVE = {Opcode.MUL, Opcode.DIV}
_REDUCE = {Opcode.SUM, Opcode.MEAN, Opcode.MIN, Opcode.MAX, Opcode.PROD}
_NUMERIC = {ValueKind.FLOAT, ValueKind.INTEGER}
_BRANCH_COUNT = 2


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise IRValidationError(message)


def _definitions(values: Iterable[Value], *, context: str) -> dict[str, ValueType]:
    names: dict[str, ValueType] = {}
    for value in values:
        _require(bool(value.name), f"{context}: empty value name")
        _require(value.name not in names, f"{context}: duplicate value {value.name!r}")
        names[value.name] = value.type
    return names


def _bindings(bindings: tuple[Binding, ...], parameters: tuple[Value, ...], available: Mapping[str, ValueType]) -> None:
    expected = _definitions(parameters, context="callee parameters")
    mapped: set[str] = set()
    for binding in bindings:
        _require(binding.parameter in expected, f"Unknown callee parameter {binding.parameter!r}")
        _require(binding.parameter not in mapped, f"Duplicate binding for {binding.parameter!r}")
        _require(binding.value in available, f"Undefined argument {binding.value!r}")
        _require(
            available[binding.value] == expected[binding.parameter],
            f"Type mismatch for parameter {binding.parameter!r}",
        )
        mapped.add(binding.parameter)
    _require(mapped == expected.keys(), "Not all callee parameters are bound")


def _same_types(actual: ValueType, expected: ValueType, context: str) -> None:
    _require(actual == expected, f"{context}: incompatible value types")


def _numeric(value: ValueType, operation: Opcode) -> None:
    _require(value.kind in _NUMERIC, f"{operation.value}: expected numeric value")


def _boolean(value: ValueType, operation: Opcode) -> None:
    _require(value.kind is ValueKind.BOOLEAN, f"{operation.value}: expected boolean value")
    _require(value.dimension is None, f"{operation.value}: predicate has a physical dimension")


def validate_function(function: FunctionIR) -> None:
    """Verify op arity, region dominance, type consistency and SSA results."""
    _require(bool(function.name), "Function must have a name")
    scope = _definitions(function.parameters, context=f"function {function.name}")
    defined = set(scope)
    _validate_block(function.body, scope, defined)


def _validate_block(block: Block, parent: Mapping[str, ValueType], defined: set[str]) -> ValueType:
    scope = dict(parent)
    for operation in block.operations:
        _require(bool(operation.result.name), "IR result name cannot be empty")
        _require(operation.result.name not in defined, f"Duplicate SSA definition {operation.result.name!r}")
        expected_arity = _ARITY.get(operation.opcode)
        _require(expected_arity is not None, f"Unknown IR operation {operation.opcode!r}")
        _require(len(operation.operands) == expected_arity, f"{operation.opcode.value}: wrong number of operands")
        args: list[ValueType] = []
        for name in operation.operands:
            _require(name in scope, f"Undefined operand {name!r}")
            args.append(scope[name])
        attrs = dict(operation.attributes)
        _require(len(attrs) == len(operation.attributes), "Duplicate operation attribute")
        _verify_operation(operation, args, attrs, scope, defined)
        defined.add(operation.result.name)
        scope[operation.result.name] = operation.result.type
    _require(block.yield_name in scope, f"Undefined yield {block.yield_name!r}")
    return scope[block.yield_name]


def _verify_operation(  # noqa: C901, PLR0912, PLR0915
    op: Operation,
    args: list[ValueType],
    attrs: dict[str, bool | int | float | str],
    scope: Mapping[str, ValueType],
    defined: set[str],
) -> None:
    code = op.opcode
    result = op.result.type
    if code is Opcode.IF:
        _require(not attrs, "scf.if cannot have attributes")
        _require(len(op.regions) == _BRANCH_COUNT, "scf.if requires two regions")
        _boolean(args[0], code)
        _require(args[0].domain.is_scalar, "scf.if requires a scalar predicate")
        for region in op.regions:
            _same_types(_validate_block(region, scope, defined), result, "scf.if yield")
        return
    _require(not op.regions, f"{code.value}: unexpected regions")
    if code is Opcode.CONSTANT:
        _require(set(attrs) == {"value"}, "arith.constant requires only a value")
        value = attrs["value"]
        expected = {ValueKind.BOOLEAN: bool, ValueKind.INTEGER: int, ValueKind.FLOAT: float}
        _require(result.kind in expected and type(value) is expected[result.kind], "Invalid constant type")
        _require(result.domain.is_scalar, "Constant must have a scalar domain")
    elif code in _ADDITIVE:
        for arg in args:
            _numeric(arg, code)
        _same_types(args[0], args[1], code.value)
        _same_types(result, args[0], code.value)
    elif code in _MULTIPLICATIVE:
        for arg in args:
            _numeric(arg, code)
        _require(args[0].kind == args[1].kind == result.kind, f"{code.value}: incompatible element kinds")
        _require(args[0].domain == args[1].domain == result.domain, f"{code.value}: domain mismatch")
        if args[0].dimension is not None and args[1].dimension is not None:
            expected = (
                args[0].dimension * args[1].dimension
                if code is Opcode.MUL
                else args[0].dimension / args[1].dimension
            )
            _require(result.dimension == expected, f"{code.value}: physical dimension mismatch")
    elif code is Opcode.POW:
        _numeric(args[0], code)
        _numeric(args[1], code)
        _require(args[1].domain.is_scalar, "arith.pow: exponent must be scalar")
        _require(args[1].dimension is None or args[1].dimension.is_dimensionless, "arith.pow: dimensioned exponent")
        _require(result.domain == args[0].domain, "arith.pow: result domain mismatch")
        _require(result.kind == args[0].kind, "arith.pow: result kind mismatch")
    elif code in (Opcode.NEG, Opcode.INV):
        _numeric(args[0], code)
        if code is Opcode.NEG:
            _same_types(result, args[0], code.value)
        else:
            _require(result.kind == args[0].kind and result.domain == args[0].domain,
                     "arith.inv: result kind or domain mismatch")
            if args[0].dimension is not None:
                _require(result.dimension == args[0].dimension**-1, "arith.inv: physical dimension mismatch")
    elif code in _COMPARE:
        for arg in args:
            _numeric(arg, code)
        _same_types(args[0], args[1], code.value)
        _require(result.kind is ValueKind.BOOLEAN, f"{code.value}: comparison result is not boolean")
        _require(result.domain == args[0].domain, f"{code.value}: comparison domain mismatch")
        _require(result.dimension is None, f"{code.value}: comparison has physical dimensions")
    elif code in _MASK:
        for arg in args:
            _boolean(arg, code)
            _require(arg.domain == result.domain, f"{code.value}: mask domain mismatch")
        _boolean(result, code)
    elif code is Opcode.WHERE:
        _boolean(args[0], code)
        _same_types(args[1], args[2], "tensor.where branches")
        _same_types(result, args[1], "tensor.where result")
        _require(args[0].domain == result.domain, "tensor.where: mask domain mismatch")
    elif code in _REDUCE | {Opcode.ISEL}:
        _numeric(args[0], code)
        _require(set(attrs) == {"axis"}, f"{code.value}: expected positional axis")
        axis = attrs["axis"]
        _require(type(axis) is int and 0 <= axis < len(args[0].domain.dims), "Invalid positional tensor axis")
        axis = typing.cast("int", axis)
        _require(result.kind == args[0].kind, f"{code.value}: result kind mismatch")
        _require(result.dimension == args[0].dimension, f"{code.value}: physical dimension mismatch")
        if code in _REDUCE:
            _require(result.domain == args[0].domain, "Reduction must retain its named domain")
        else:
            _require(args[1].kind is ValueKind.INTEGER, "tensor.isel requires an integer index")
            _require(args[1].domain.is_scalar, "tensor.isel requires a scalar index")
            _require(
                result.domain.dims == args[0].domain.dims[:axis] + args[0].domain.dims[axis + 1 :],
                "tensor.isel must remove the selected axis",
            )


def validate_program(program: ProgramIR) -> None:
    """Verify dependency order, typed bindings, solves, and final return."""
    available = _definitions(program.inputs, context="program inputs")
    solutions: dict[str, dict[str, ValueType]] = {}
    for step in program.steps:
        if isinstance(step, FunctionStep):
            validate_function(step.function)
            _bindings(step.bindings, step.function.parameters, available)
            output_type = _function_result(step.function)
            _same_types(step.output.type, output_type, "Function result")
            produced = step.output
        elif isinstance(step, ChildStep):
            _require(bool(step.child_path) and all(step.child_path), "Child path must be qualified")
            _require(bool(step.child_output), "Child output cannot be empty")
            _bindings(step.bindings, step.parameters, available)
            produced = step.output
        elif isinstance(step, LinearSolveStep):
            _require(step.solution.type.kind is ValueKind.OPAQUE, "Solve result must be opaque")
            _require(bool(step.unknowns), "Linear solve requires unknowns")
            unknowns = _definitions(step.unknowns, context="linear unknowns")
            validate_function(step.coefficients)
            validate_function(step.rhs)
            _bindings(step.bindings, step.parameters, available)
            _require(
                step.coefficients.parameters == step.parameters and step.rhs.parameters == step.parameters,
                "Linear kernels must take the solve parameters",
            )
            _numeric(_function_result(step.coefficients), Opcode.MUL)
            _numeric(_function_result(step.rhs), Opcode.MUL)
            solutions[step.solution.name] = unknowns
            produced = step.solution
        else:
            _require(step.solution in solutions, f"Unknown linear solution {step.solution!r}")
            _require(step.unknown in solutions[step.solution], f"Unknown projection {step.unknown!r}")
            _same_types(step.output.type, solutions[step.solution][step.unknown], "Projection type")
            produced = step.output
        _require(bool(produced.name), "Program step output cannot be empty")
        _require(produced.name not in available, f"Duplicate program producer {produced.name!r}")
        available[produced.name] = produced.type
    _require(program.return_name in available, f"Undefined program return {program.return_name!r}")


def _function_result(function: FunctionIR) -> ValueType:
    scope = _definitions(function.parameters, context="function parameters")
    return _validate_block(function.body, scope, set(scope))
