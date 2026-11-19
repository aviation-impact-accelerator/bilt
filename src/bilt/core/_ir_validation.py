"""Structural validation for realised function IR before backend emission."""

import typing

from bilt.core._ir import Block, FunctionIR, Value

if typing.TYPE_CHECKING:
    from collections.abc import Mapping


_ARITY: Mapping[str, int] = {
    "arith.constant": 0,
    "arith.neg": 1,
    "arith.pos": 1,
    "arith.inv": 1,
    "arith.add": 2,
    "arith.sub": 2,
    "arith.mul": 2,
    "arith.div": 2,
    "arith.pow": 2,
    "cmp.eq": 2,
    "cmp.ne": 2,
    "cmp.lt": 2,
    "cmp.le": 2,
    "cmp.gt": 2,
    "cmp.ge": 2,
    "bilt.where": 3,
    "bilt.mask_and": 2,
    "bilt.mask_or": 2,
    "bilt.mask_not": 1,
    "bilt.mean": 1,
    "bilt.sum": 1,
    "bilt.min": 1,
    "bilt.max": 1,
    "bilt.prod": 1,
    "bilt.isel": 2,
    "scf.if": 1,
}


def validate_function_ir(function: FunctionIR) -> None:
    """Verify operands, SSA identity, region dominance and operation shapes."""
    if len(function.parameters) != len(set(function.parameters)):
        message = "IR function parameters must be unique."
        raise ValueError(message)
    if len({name for name, _ in function.parameter_types}) != len(function.parameter_types):
        message = "IR parameter type names must be unique."
        raise ValueError(message)
    if any(name not in function.parameters for name, _ in function.parameter_types):
        message = "IR parameter types must belong to parameters."
        raise ValueError(message)
    known = {Value(name) for name in function.parameters}
    _validate_block(function.body, known, set())


def _validate_block(block: Block, visible: set[Value], defined: set[Value]) -> None:  # noqa: C901, PLR0912
    scope = visible.copy()
    for operation in block.operations:
        if operation.result in defined or operation.result in scope:
            message = f"IR value {operation.result.key!r} defined more than once."
            raise ValueError(message)
        expected = _ARITY.get(operation.opcode)
        if expected is None:
            message = f"Unsupported IR operation {operation.opcode!r}."
            raise ValueError(message)
        if len(operation.operands) != expected:
            message = f"IR operation {operation.opcode!r} expects {expected} operands."
            raise ValueError(message)
        for operand in operation.operands:
            if operand not in scope:
                message = f"IR operand {operand.key!r} is not defined in this scope."
                raise ValueError(message)
        region_count = 2 if operation.opcode == "scf.if" else 0
        if len(operation.regions) != region_count:
            message = f"IR operation {operation.opcode!r} expects {region_count} regions."
            raise ValueError(message)
        attributes = dict(operation.attributes)
        if len(attributes) != len(operation.attributes):
            message = "IR operation attributes must be unique."
            raise ValueError(message)
        if operation.opcode == "arith.constant" and "value" not in attributes:
            message = "IR constants require a value."
            raise ValueError(message)
        if operation.opcode in (
            "bilt.sum",
            "bilt.mean",
            "bilt.min",
            "bilt.max",
            "bilt.prod",
            "bilt.isel",
        ) and (type(attributes.get("axis")) is not int or attributes["axis"] < 0):
            message = "IR tensor operations require a nonnegative positional axis."
            raise ValueError(message)
        defined.add(operation.result)
        for region in operation.regions:
            _validate_block(region, scope, defined)
        scope.add(operation.result)
    if block.yield_value not in scope:
        message = f"IR yielded value {block.yield_value.key!r} is not defined."
        raise ValueError(message)
