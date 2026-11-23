"""Categorical diagnostics reported when importing models with bilt check."""

import subprocess
import sys
import textwrap
import typing

import pytest

if typing.TYPE_CHECKING:
    from pathlib import Path


def _check_model(root: Path, source: str) -> subprocess.CompletedProcess[str]:
    (root / "model.py").write_text(textwrap.dedent(source), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "bilt", "check", "."],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_collects_unsupported_categorical_port_declarations(tmp_path: Path) -> None:
    result = _check_model(
        tmp_path,
        """
        from enum import Enum, Flag, IntFlag, auto
        from typing import Annotated
        from bilt import Interface, equation, field, system, transform
        from bilt.units.dimensions import LENGTH

        class Flags(Flag):
            ON = auto()

        class Options(IntFlag):
            ON = auto()

        class Mode(Enum):
            ON = auto()

        @system
        class Bad(metaclass=Interface):
            flags: Flags = field()
            options: Options = field()
            optional: Mode | None = field()
            dimensional: Annotated[Mode, LENGTH] = field()
            abstract: Enum = field()

        @equation
        def bad_equation(mode: Options, result: float) -> bool:
            return result == 0.0

        @transform
        def result(mode: Mode) -> float:
            return 0.0

        @transform
        def mode(value: float) -> Mode:
            return Mode.ON
        """,
    )
    assert result.returncode == 1
    assert "BILT999" not in result.stderr
    for message in (
        "Port 'flags' of 'Bad' declares Flags. Flag and IntFlag are unsupported",
        "Port 'options' of 'Bad' declares Options. Flag and IntFlag are unsupported",
        "Port 'optional' of 'Bad' must use a concrete enum class directly",
        "Port 'dimensional' of 'Bad' declares units, dimensionality or a tensor domain",
        "Port 'abstract' of 'Bad' declares Enum, which has no enum members",
        "Port 'mode' of 'bad_equation' declares Options. Flag and IntFlag are unsupported",
    ):
        assert any(
            "model.py:" in line and "error: BILT004 " in line and message in line for line in result.stderr.splitlines()
        )


@pytest.mark.parametrize(
    ("definition", "message"),
    (
        (
            """
            @equation
            def output(mode: Mode, result: float) -> bool:
                match mode:
                    case Other.ON:
                        selected = 1.0
                    case _:
                        selected = 0.0
                return result == selected
            """,
            "enum cases must name members of the match subject's enum type",
        ),
        (
            """
            @equation
            def output(mode: float, result: float) -> bool:
                match mode:
                    case Mode.ON:
                        selected = 1.0
                    case _:
                        selected = 0.0
                return result == selected
            """,
            "enum patterns require a match subject declared with that enum type",
        ),
        (
            """
            @equation
            def output(mode: Mode, result: float) -> bool:
                match mode:
                    case Mode.ON:
                        selected = 1.0
                return result == selected
            """,
            "match requires literal cases and a final case _",
        ),
        (
            """
            @equation
            def output(mode: Mode, result: float) -> bool:
                match mode:
                    case Mode.ON if result > 0:
                        selected = 1.0
                    case _:
                        selected = 0.0
                return result == selected
            """,
            "match guards are not supported",
        ),
        (
            """
            @equation
            def output(mode: Mode, result: float) -> bool:
                if mode:
                    selected = 1.0
                else:
                    selected = 0.0
                return result == selected
            """,
            "enum parameters may only be used as match subjects",
        ),
        (
            """
            @equation
            def output(result: float) -> bool:
                return result == Mode.ON + 1
            """,
            "enum members may only be used in match case patterns",
        ),
        (
            """
            def member():
                return Mode.ON

            @equation
            def output(result: float) -> bool:
                return result == member()
            """,
            "Enum member Mode.ON may only be used in match case patterns",
        ),
        (
            """
            @system
            class Child(metaclass=Interface):
                mode: Mode = field()

            @equation
            def output(child: Child, result: float) -> bool:
                return result == child.mode * 2
            """,
            "Found 'child.mode'",
        ),
    ),
    ids=(
        "wrong-member-class",
        "numeric-subject",
        "missing-wildcard",
        "guarded-case",
        "boolean-selector",
        "member-arithmetic",
        "member-from-helper",
        "child-arithmetic",
    ),
)
def test_check_reports_invalid_enum_matching(tmp_path: Path, definition: str, message: str) -> None:
    common = textwrap.dedent("""
        from enum import IntEnum
        from bilt import Interface, equation, field, system

        class Mode(IntEnum):
            ON = 1

        class Other(IntEnum):
            ON = 1
        """)
    result = _check_model(tmp_path, common + textwrap.dedent(definition))
    assert result.returncode == 1
    assert "model.py:" in result.stderr
    assert "error: BILT004 " in result.stderr
    assert message in result.stderr


@pytest.mark.parametrize(("child_type", "declaration"), (("Other", "Other"), ("float", "a non-enum port")))
def test_check_reports_qualified_enum_type_disagreement(tmp_path: Path, child_type: str, declaration: str) -> None:
    result = _check_model(
        tmp_path,
        f"""
        from enum import IntEnum
        from bilt import Implementation, Interface, equation, field, system

        class Mode(IntEnum):
            ON = 1

        class Other(IntEnum):
            ON = 1

        @system
        class Expected(metaclass=Interface):
            mode: Mode = field()

        @system
        class Supplied(metaclass=Interface):
            mode: {child_type} = field()

        @system
        class Parent(metaclass=Interface):
            result: float = field()

        @equation
        def result(child: Expected, result: float) -> bool:
            match child.mode:
                case Mode.ON:
                    selected = 1.0
                case _:
                    selected = 0.0
            return result == selected

        child = Implementation(Supplied, [])
        parent = Implementation(Parent, [result], internal_implementations={{"child": child}})
        """,
    )
    assert result.returncode == 1
    assert "error: BILT004 " in result.stderr
    assert "categorical port 'child.mode' inconsistent enum types" in result.stderr
    assert "'Expected' declares Mode" in result.stderr
    assert f"'Supplied' declares {declaration}" in result.stderr


def test_check_reports_direct_enum_type_disagreement(tmp_path: Path) -> None:
    result = _check_model(
        tmp_path,
        """
        from enum import IntEnum
        from bilt import Implementation, Interface, equation, field, system

        class Mode(IntEnum):
            ON = 1

        class Other(IntEnum):
            ON = 1

        @system
        class Model(metaclass=Interface):
            mode: Mode = field()
            result: float = field()

        @equation
        def result(mode: Other, result: float) -> bool:
            match mode:
                case Other.ON:
                    selected = 1.0
                case _:
                    selected = 0.0
            return result == selected

        model = Implementation(Model, [result])
        """,
    )
    assert result.returncode == 1
    assert "error: BILT004 " in result.stderr
    assert "categorical port 'mode' inconsistent enum types" in result.stderr
    assert "'Model' declares Mode" in result.stderr
    assert "'result' declares Other" in result.stderr


def test_check_accepts_nested_enum_matching_without_backend_compilation(tmp_path: Path) -> None:
    result = _check_model(
        tmp_path,
        """
        from enum import Enum
        from bilt import Implementation, Interface, equation, field, system

        class Mode(Enum):
            OFF = "off"
            ON = "on"

        @system
        class Child(metaclass=Interface):
            mode: Mode = field()

        @system
        class Parent(metaclass=Interface):
            result: float = field()

        @equation
        def result(child: Child, result: float) -> bool:
            match child.mode:
                case Mode.OFF:
                    selected = 0
                case _:
                    selected = 1.5
            return result == selected

        child = Implementation(Child, [])
        parent = Implementation(Parent, [result], internal_implementations={"child": child})
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "2 system(s), 2 implementation(s), 1 equation(s)" in result.stdout


def test_check_keeps_numeric_match_syntax_outside_categorical_diagnostics(tmp_path: Path) -> None:
    result = _check_model(
        tmp_path,
        """
        from bilt import equation

        @equation
        def output(mode: float, result: float) -> bool:
            match mode:
                case 0:
                    selected = 1.0
            return result == selected
        """,
    )
    assert result.returncode == 1
    assert "error: BILT999 BiltEquationSyntaxError" in result.stderr
    assert "match requires literal cases and a final case _" in result.stderr
