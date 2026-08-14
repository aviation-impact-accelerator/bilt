from __future__ import annotations

import types
import typing
from unittest.mock import Mock

import bilt.core._system._implementation as implementation_module
from bilt.core._system._implementation import Implementation

if typing.TYPE_CHECKING:
    import pytest

    from bilt._backend import Backend
    from bilt.core._interface import Interface
    from bilt.core._model_constituent import BiltModelConstituent


class FakeSystem:
    def __init__(self) -> None:
        self.__bilt_system_registry__: set[Implementation] = set()
        self.__bilt_system_implementation_default__: list[Implementation] = []


class FakeComponent:
    __bilt_name__ = "component"


class FakeBackend:
    def __init__(self) -> None:
        self.chain_calls = 0

    def chain_case(self, case: object) -> object:
        self.chain_calls += 1

        return types.SimpleNamespace(
            parameters=("x",),
            callable=case,
        )


def test_evaluate_returns_available_output_without_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structural_analysis = Mock()

    monkeypatch.setattr(
        implementation_module,
        "get_current_structural_analysis_backend",
        structural_analysis,
    )

    system = typing.cast("Interface", FakeSystem())
    components = typing.cast(
        "list[BiltModelConstituent]",
        [FakeComponent()],
    )
    backend = typing.cast("Backend", FakeBackend())

    implementation = Implementation(system, components)

    expected = 1

    actual = implementation.evaluate(
        {"x": expected},
        "x",
        backend,
    )

    assert actual == expected
    structural_analysis.assert_not_called()


def test_evaluate_reuses_cached_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structural_analysis_calls = 0

    def structural_analysis(*_: object) -> object:
        nonlocal structural_analysis_calls
        structural_analysis_calls += 1

        def increment(x: int) -> int:
            return x + 1

        return increment

    def get_current_structural_analysis_backend() -> object:
        return structural_analysis

    monkeypatch.setattr(
        implementation_module,
        "get_current_structural_analysis_backend",
        get_current_structural_analysis_backend,
    )

    fake_backend = FakeBackend()
    backend = typing.cast("Backend", fake_backend)
    system = typing.cast("Interface", FakeSystem())
    components = typing.cast(
        "list[BiltModelConstituent]",
        [FakeComponent()],
    )

    implementation = Implementation(system, components)

    first_input = 1
    second_input = 2

    assert (
        implementation.evaluate(
            {"x": first_input},
            "y",
            backend,
        )
        == first_input + 1
    )
    assert (
        implementation.evaluate(
            {"x": second_input},
            "y",
            backend,
        )
        == second_input + 1
    )

    expected_calls = 1

    assert structural_analysis_calls == expected_calls
    assert fake_backend.chain_calls == expected_calls


def test_cache_is_specific_to_structural_analysis_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def first_structural_analysis(*_: object) -> object:
        def increment(x: int) -> int:
            return x + 1

        return increment

    def second_structural_analysis(*_: object) -> object:
        def increment(x: int) -> int:
            return x + 2

        return increment

    current = first_structural_analysis

    def get_current_structural_analysis_backend() -> object:
        return current

    monkeypatch.setattr(
        implementation_module,
        "get_current_structural_analysis_backend",
        get_current_structural_analysis_backend,
    )

    fake_backend = FakeBackend()
    backend = typing.cast("Backend", fake_backend)
    system = typing.cast("Interface", FakeSystem())
    components = typing.cast(
        "list[BiltModelConstituent]",
        [FakeComponent()],
    )

    implementation = Implementation(system, components)

    value = 1

    assert (
        implementation.evaluate(
            {"x": value},
            "y",
            backend,
        )
        == value + 1
    )

    current = second_structural_analysis

    assert (
        implementation.evaluate(
            {"x": value},
            "y",
            backend,
        )
        == value + 2
    )

    expected_chain_calls = 2
    assert fake_backend.chain_calls == expected_chain_calls
