import types
import typing

import pytest

import bilt.core._system._implementation as implementation_module
from bilt.core._system._implementation import Implementation

if typing.TYPE_CHECKING:
    from bilt._backend import Backend, BackendCallable
    from bilt.core._interface import Interface


class _System:
    def __init__(self) -> None:
        self.__bilt_system_registry__: set[Implementation] = set()
        self.__bilt_system_implementation_default__: list[Implementation] = []


class _Domains:
    def __init__(self) -> None:
        self.bound_inputs: tuple[object, ...] = ()
        self.bound_output: object | None = None

    def bind_all(self, values: typing.Iterable[object]) -> None:
        self.bound_inputs = tuple(values)

    def bind(self, value: object) -> None:
        self.bound_output = value


class _EvaluationContext:
    def __init__(self) -> None:
        self.domains = _Domains()

    def __enter__(self) -> typing.Self:
        return self

    def __exit__(self, *_: object) -> None:
        pass


@pytest.fixture
def evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> _EvaluationContext:
    evaluation = _EvaluationContext()

    monkeypatch.setattr(
        implementation_module,
        "EvaluationContext",
        lambda: evaluation,
    )

    return evaluation


def test_evaluate_binds_input_output(
    evaluation: _EvaluationContext,
) -> None:
    implementation = Implementation(
        typing.cast("Interface", _System()),
        [],
    )
    backend = typing.cast("Backend", object())
    value = object()

    result = implementation.evaluate(
        inputs={"result": value},
        output="result",
        backend=backend,
    )

    assert result is value
    assert evaluation.domains.bound_inputs == (value,)
    assert evaluation.domains.bound_output is value


@pytest.mark.xfail(reason="Internal canonical domain ordering requires evaluation context manager, would need all that implemented on _EvaluationContext too to make this pass")
def test_evaluate_binds_computed_output(
    evaluation: _EvaluationContext,
) -> None:
    implementation = Implementation(
        typing.cast("Interface", _System()),
        [],
    )
    backend = typing.cast("Backend", object())
    input_value = object()
    output_value = object()

    cached = typing.cast(
        "BackendCallable",
        types.SimpleNamespace(
            parameters=("source",),
            callable=lambda source: output_value,  # noqa: ARG005
        ),
    )
    implementation._callable_cache[  # noqa: SLF001
        ("result", frozenset({"source"}), backend)
    ] = cached

    result = implementation.evaluate(
        inputs={"source": input_value},
        output="result",
        backend=backend,
    )

    assert result is output_value
    assert evaluation.domains.bound_inputs == (input_value,)
    assert evaluation.domains.bound_output is output_value
