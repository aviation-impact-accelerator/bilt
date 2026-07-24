from __future__ import annotations

__all__ = [
    "BiltSystemEvaluationError",
    "encode_inputs",
    "find_successful_case",
    "flatten_inputs",
]

import ast
import collections.abc
import dataclasses
import inspect
import textwrap
import typing

from bilt._backend import BackendCallable, BackendCase
from bilt.core._error import BiltCoreError
from bilt.core._transform import BasicTransform

from ._interface import Interface
from ._sentinel import MISSING

if typing.TYPE_CHECKING:
    from bilt._backend import Backend

    from ._implementation import Implementation


class BiltSystemEvaluationError(BiltCoreError):
    pass


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class _FieldRef:
    path: tuple[str, ...]
    name: str

    @classmethod
    def parse(cls, value: str) -> typing.Self:
        parts = tuple(part for part in value.split(".") if part)

        if not parts:
            message = "A field reference cannot be empty."
            raise ValueError(message)

        return cls(path=parts[:-1], name=parts[-1])

    @property
    def dotted(self) -> str:
        return ".".join((*self.path, self.name))

    @property
    def safe_name(self) -> str:
        return "__".join((*self.path, self.name))

    def relative_to(self, scope: tuple[str, ...]) -> _FieldRef | None:
        if self.path[: len(scope)] != scope:
            return None

        return _FieldRef(
            path=self.path[len(scope) :],
            name=self.name,
        )

    def rebase(self, scope: tuple[str, ...]) -> _FieldRef:
        return _FieldRef(
            path=(*scope, *self.path),
            name=self.name,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _Dependency:
    parameter: str
    attribute: str | None
    field: _FieldRef


@dataclasses.dataclass(frozen=True, slots=True)
class _BoundTransform:
    operation: BasicTransform
    output: _FieldRef
    dependencies: tuple[_Dependency, ...]


def flatten_inputs(
    inputs: collections.abc.Mapping[str, object],
) -> dict[str, object]:
    """Flatten nested system instances into dotted leaf-field inputs."""

    flattened: dict[str, object] = {}

    def add(path: tuple[str, ...], value: object) -> None:
        variables = getattr(
            value,
            "__bilt_system_instance_variables__",
            None,
        )

        if isinstance(variables, collections.abc.Mapping):
            for name, child_value in variables.items():
                if child_value is MISSING:
                    continue

                add((*path, name), child_value)

            return

        if value is MISSING:
            return

        flattened[".".join(path)] = value

    for name, value in inputs.items():
        if "." in name:
            if value is not MISSING:
                flattened[name] = value
            continue

        add((name,), value)

    return flattened


def encode_inputs(
    inputs: collections.abc.Mapping[str, object],
    parameters: collections.abc.Iterable[str],
) -> dict[str, object]:
    """Select known inputs required by a generated backend callable."""

    flattened = flatten_inputs(inputs)
    encoded = {
        _FieldRef.parse(name).safe_name: value
        for name, value in flattened.items()
    }

    try:
        return {
            parameter: encoded[parameter]
            for parameter in parameters
        }
    except KeyError as error:
        parameter = typing.cast("str", error.args[0])
        message = f"Required backend input {parameter!r} is not available."
        raise BiltSystemEvaluationError(message) from None


def find_successful_case(
    implementation: Implementation,
    inputs: dict[str, object],
    output: str,
    backend: Backend,
) -> BackendCase:
    """Find and lower a transform path, querying child systems lazily."""

    flattened = flatten_inputs(inputs)
    known = {
        _FieldRef.parse(name)
        for name in flattened
    }
    target = _FieldRef.parse(output)

    transforms, required_inputs = _resolve_output(
        root=implementation,
        known=known,
        output=target,
        visiting=(),
    )

    return BackendCase(
        name=target.safe_name,
        parameters=tuple(
            field.safe_name
            for field in required_inputs
        ),
        intermediates=[
            _operation_to_callable(
                transform,
                backend=backend,
            )
            for transform in transforms
        ],
        return_name=target.safe_name,
    )


def _resolve_output(
    *,
    root: Implementation,
    known: set[_FieldRef],
    output: _FieldRef,
    visiting: tuple[_FieldRef, ...],
) -> tuple[list[_BoundTransform], list[_FieldRef]]:
    if output in known:
        return [], [output]

    if output in visiting:
        start = visiting.index(output)
        cycle = (*visiting[start:], output)
        message = (
            "Cyclic transform dependency: "
            + " -> ".join(field.dotted for field in cycle)
        )
        raise BiltSystemEvaluationError(message)

    failures: list[str] = []

    for candidate in _candidates_for(
        implementation=root,
        target=output,
        scope=(),
    ):
        operations: list[_BoundTransform] = []
        required_inputs: list[_FieldRef] = []

        try:
            for dependency in candidate.dependencies:
                child_operations, child_inputs = _resolve_output(
                    root=root,
                    known=known,
                    output=dependency.field,
                    visiting=(*visiting, output),
                )
                operations.extend(child_operations)
                _extend_unique(required_inputs, child_inputs)

        except BiltSystemEvaluationError as error:
            failures.append(str(error))
            continue

        operations.append(candidate)
        return _deduplicate_operations(operations), required_inputs

    detail = f" Tried: {'; '.join(failures)}" if failures else ""
    message = (
        f"No transform path found for output {output.dotted!r}."
        f"{detail}"
    )
    raise BiltSystemEvaluationError(message)


def _candidates_for(
    *,
    implementation: Implementation,
    target: _FieldRef,
    scope: tuple[str, ...],
) -> typing.Iterator[_BoundTransform]:
    """Try a transform owned by this level before inspecting the child."""

    relative_target = target.relative_to(scope)

    if relative_target is None:
        return

    operation = implementation._components.get(  # noqa: SLF001
        relative_target.dotted
    )

    if operation is not None:
        if not isinstance(operation, BasicTransform):
            message = (
                f"Only transforms are currently supported, but "
                f"{relative_target.dotted!r} is registered to "
                f"{type(operation).__name__}."
            )
            raise BiltSystemEvaluationError(message)

        yield _bind_transform(
            implementation=implementation,
            operation=operation,
            output=target,
            scope=scope,
        )

    if not relative_target.path:
        return

    child_name = relative_target.path[0]
    child_implementation = implementation._children.get(  # noqa: SLF001
        child_name
    )

    if child_implementation is None:
        child_implementation = implementation._internal_systems.get(  # noqa: SLF001
            child_name
        )

    if child_implementation is None:
        return

    yield from _candidates_for(
        implementation=child_implementation,
        target=target,
        scope=(*scope, child_name),
    )


def _bind_transform(
    *,
    implementation: Implementation,
    operation: BasicTransform,
    output: _FieldRef,
    scope: tuple[str, ...],
) -> _BoundTransform:
    return _BoundTransform(
        operation=operation,
        output=output,
        dependencies=_infer_dependencies(
            implementation=implementation,
            operation=operation,
            scope=scope,
        ),
    )


def _infer_dependencies(
    *,
    implementation: Implementation,
    operation: BasicTransform,
    scope: tuple[str, ...],
) -> tuple[_Dependency, ...]:
    function_ast = _function_ast(operation)
    parameters = tuple(operation.__bilt_parameters__)
    child_parameters = _system_parameter_names(implementation)

    visitor = _DependencyVisitor(
        parameters=set(parameters),
        child_parameters=child_parameters,
    )
    visitor.visit(function_ast)

    unsupported = child_parameters & visitor.plain_child_uses

    if unsupported:
        names = ", ".join(sorted(unsupported))
        message = (
            "System-valued transform parameters must be accessed through "
            f"fields rather than used directly: {names}."
        )
        raise BiltSystemEvaluationError(message)

    dependencies: list[_Dependency] = []

    for parameter in parameters:
        if parameter not in child_parameters:
            dependencies.append(
                _Dependency(
                    parameter=parameter,
                    attribute=None,
                    field=_FieldRef.parse(parameter).rebase(scope),
                )
            )
            continue

        for attribute in visitor.attributes.get(parameter, ()):
            dependencies.append(
                _Dependency(
                    parameter=parameter,
                    attribute=attribute,
                    field=_FieldRef.parse(
                        f"{parameter}.{attribute}"
                    ).rebase(scope),
                )
            )

    return tuple(dependencies)


def _system_parameter_names(
    implementation: Implementation,
) -> set[str]:
    public_children = {
        name
        for name, attribute
        in implementation.system.__bilt_system_attributes__.items()
        if isinstance(attribute.annotation, Interface)
    }
    return public_children | set(implementation._internal_systems)  # noqa: SLF001


class _DependencyVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        parameters: set[str],
        child_parameters: set[str],
    ) -> None:
        self._parameters = parameters
        self._child_parameters = child_parameters
        self.attributes: dict[str, list[str]] = {}
        self.plain_child_uses: set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        reference = _attribute_reference(node)

        if (
            reference is not None
            and reference[0] in self._child_parameters
        ):
            parameter, attribute = reference
            attributes = self.attributes.setdefault(parameter, [])

            if attribute not in attributes:
                attributes.append(attribute)

            return

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            node.id in self._parameters
            and node.id in self._child_parameters
        ):
            self.plain_child_uses.add(node.id)


def _attribute_reference(
    node: ast.Attribute,
) -> tuple[str, str] | None:
    attributes = [node.attr]
    value = node.value

    while isinstance(value, ast.Attribute):
        attributes.append(value.attr)
        value = value.value

    if not isinstance(value, ast.Name):
        return None

    return value.id, ".".join(reversed(attributes))


def _deduplicate_operations(
    operations: list[_BoundTransform],
) -> list[_BoundTransform]:
    result: list[_BoundTransform] = []
    seen: set[_FieldRef] = set()

    for operation in operations:
        if operation.output in seen:
            continue

        seen.add(operation.output)
        result.append(operation)

    return result


def _extend_unique(
    target: list[_FieldRef],
    values: collections.abc.Iterable[_FieldRef],
) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _function_ast(
    operation: BasicTransform,
) -> ast.FunctionDef:
    function_ast = typing.cast(
        "ast.FunctionDef",
        ast.parse(
            textwrap.dedent(
                inspect.getsource(operation._function)  # noqa: SLF001
            )
        ).body[0],
    )
    function_ast.decorator_list = [
        decorator
        for decorator in function_ast.decorator_list
        if not _is_transform_decorator(decorator)
    ]
    return function_ast


def _is_transform_decorator(
    decorator: ast.expr,
) -> bool:
    if isinstance(decorator, ast.Call):
        return _is_transform_decorator(decorator.func)

    if isinstance(decorator, ast.Name):
        return decorator.id == "transform"

    return (
        isinstance(decorator, ast.Attribute)
        and decorator.attr == "transform"
    )


class _RewriteDependencies(ast.NodeTransformer):
    def __init__(
        self,
        dependencies: tuple[_Dependency, ...],
    ) -> None:
        self._names = {
            dependency.parameter: dependency.field.safe_name
            for dependency in dependencies
            if dependency.attribute is None
        }
        self._attributes = {
            (dependency.parameter, dependency.attribute):
            dependency.field.safe_name
            for dependency in dependencies
            if dependency.attribute is not None
        }

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        reference = _attribute_reference(node)

        if reference is not None:
            replacement = self._attributes.get(reference)

            if replacement is not None:
                return ast.copy_location(
                    ast.Name(
                        id=replacement,
                        ctx=node.ctx,
                    ),
                    node,
                )

        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self._names.get(node.id)

        if replacement is None:
            return node

        return ast.copy_location(
            ast.Name(
                id=replacement,
                ctx=node.ctx,
            ),
            node,
        )


def _operation_to_callable(
    transform: _BoundTransform,
    *,
    backend: Backend,
) -> BackendCallable:
    function_ast = _function_ast(transform.operation)

    if (
        function_ast.args.posonlyargs
        or function_ast.args.vararg is not None
        or function_ast.args.kwonlyargs
        or function_ast.args.kwarg is not None
        or function_ast.args.defaults
    ):
        message = (
            "Hierarchical transform lowering currently supports only "
            "ordinary positional parameters without defaults."
        )
        raise BiltSystemEvaluationError(message)

    function_ast.args.args = [
        ast.arg(
            arg=dependency.field.safe_name,
            annotation=None,
        )
        for dependency in transform.dependencies
    ]
    function_ast.name = transform.output.safe_name

    function_ast = typing.cast(
        "ast.FunctionDef",
        _RewriteDependencies(
            transform.dependencies
        ).visit(function_ast),
    )
    ast.fix_missing_locations(function_ast)

    return backend.transpile(function_ast)
