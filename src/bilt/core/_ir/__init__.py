"""Backend-neutral, typed, verified representation of realised computations.

This is an internal compiler contract. Runtime lowering is deliberately unchanged.
"""

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
    ProjectionStep,
    Source,
    Value,
    ValueKind,
    ValueType,
)
from bilt.core._ir._validation import IRValidationError, validate_function, validate_program

__all__ = [
    "Binding",
    "Block",
    "ChildStep",
    "FunctionIR",
    "FunctionStep",
    "IRValidationError",
    "LinearSolveStep",
    "Opcode",
    "Operation",
    "ProgramIR",
    "ProjectionStep",
    "Source",
    "Value",
    "ValueKind",
    "ValueType",
    "validate_function",
    "validate_program",
]
