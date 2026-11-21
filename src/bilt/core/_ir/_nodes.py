"""Backend-neutral IR contracts for realised Bilt computations.

These immutable nodes describe computations, not acausal symbolic equations.
They deliberately carry no Python AST, backend callables, or numerical engines.
They do not participate in runtime lowering until follow-on PRs.
"""

import dataclasses
import enum
import typing

from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from bilt.units._metadata import Metadata
    from bilt.units.dimensions._dimensions import Dimension


class ValueKind(enum.StrEnum):
    """Scalar element kind, independent of tensor rank and physical units."""

    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    OPAQUE = "opaque"


class Opcode(enum.StrEnum):
    """Operations supported by the first backend-neutral function contract."""

    CONSTANT = "arith.constant"
    ADD = "arith.add"
    SUB = "arith.sub"
    MUL = "arith.mul"
    DIV = "arith.div"
    POW = "arith.pow"
    NEG = "arith.neg"
    INV = "arith.inv"
    EQ = "cmp.eq"
    NE = "cmp.ne"
    LT = "cmp.lt"
    LE = "cmp.le"
    GT = "cmp.gt"
    GE = "cmp.ge"
    WHERE = "tensor.where"
    MASK_AND = "tensor.mask_and"
    MASK_OR = "tensor.mask_or"
    MASK_NOT = "tensor.mask_not"
    SUM = "tensor.sum"
    MEAN = "tensor.mean"
    MIN = "tensor.min"
    MAX = "tensor.max"
    PROD = "tensor.prod"
    ISEL = "tensor.isel"
    IF = "scf.if"


type AttributeValue = bool | int | float | str


@dataclasses.dataclass(frozen=True, slots=True)
class ValueType:
    """Type of one SSA value.

    `domain` refers to named tensor dimensions; `dimension` refers to
    physical dimensions. A missing physical dimension means *unknown*, not
    dimensionless. Tagged physical dimensions retain their scientific meaning.
    """

    kind: ValueKind
    domain: Domain = dataclasses.field(default_factory=Domain.scalar)
    dimension: Dimension | None = None
    metadata: Metadata | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Value:
    """A named SSA definition or function argument, with an explicit type."""

    name: str
    type: ValueType


@dataclasses.dataclass(frozen=True, slots=True)
class Source:
    """Logical model provenance, excluding runtime function/callable objects."""

    path: tuple[str, ...]
    name: str


@dataclasses.dataclass(frozen=True, slots=True)
class Operation:
    """Pure function operation yielding exactly one SSA value."""

    result: Value
    opcode: Opcode
    operands: tuple[str, ...] = ()
    attributes: tuple[tuple[str, AttributeValue], ...] = ()
    regions: tuple[Block, ...] = ()
    source: Source | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Block:
    """Lexical SSA region with an explicit yielded value."""

    operations: tuple[Operation, ...]
    yield_name: str


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionIR:
    """A directed numerical function, after equation isolation and inference."""

    name: str
    parameters: tuple[Value, ...]
    body: Block
    source: Source | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Binding:
    """Pass the caller's named value into a named callee parameter."""

    value: str
    parameter: str


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionStep:
    """Evaluate a self-contained function and bind the returned value."""

    output: Value
    function: FunctionIR
    bindings: tuple[Binding, ...]
    source: Source | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ChildStep:
    """Call an identified child model route using qualified port bindings."""

    output: Value
    child_path: tuple[str, ...]
    child_output: str
    parameters: tuple[Value, ...]
    bindings: tuple[Binding, ...]
    source: Source | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class LinearSolveStep:
    """Represent one coupled solve and the names/types of its unknowns.

    The matrix and right-hand-side construction are explicit FunctionIRs.
    An opaque solution value is projected by following ProjectionSteps.
    The numerical method is intentionally not selected here.
    """

    solution: Value
    unknowns: tuple[Value, ...]
    parameters: tuple[Value, ...]
    bindings: tuple[Binding, ...]
    coefficients: FunctionIR
    rhs: FunctionIR
    source: Source | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ProjectionStep:
    """Extract a named unknown from the result of a coupled solve."""

    output: Value
    solution: str
    unknown: str
    source: Source | None = None


type ProgramStep = FunctionStep | ChildStep | LinearSolveStep | ProjectionStep


@dataclasses.dataclass(frozen=True, slots=True)
class ProgramIR:
    """A dependency-ordered, backend-neutral realised computation."""

    inputs: tuple[Value, ...]
    steps: tuple[ProgramStep, ...]
    return_name: str
    source: Source | None = None
