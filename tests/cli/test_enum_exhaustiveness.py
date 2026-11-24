"""Check that categorical matches explicitly enumerate enum alternatives."""

import subprocess
import sys
import textwrap
import typing

import pytest

if typing.TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("kind", ("equation", "transform"))
def test_bilt_check_flags_wildcard_masking_missing_enum_members(tmp_path: Path, kind: str) -> None:
    if kind == "equation":
        member = """
            @equation
            def price(scenario: Scenario, price: float) -> bool:
                match scenario:
                    case Scenario.MINIMUM:
                        selected = 0.5
                    case _:
                        selected = 1.5
                return price == selected
        """
    else:
        member = """
            @transform
            def price(scenario: Scenario) -> float:
                match scenario:
                    case Scenario.MINIMUM:
                        return 0.5
                    case _:
                        return 1.5
        """
    prefix = """
        import enum
        from bilt import equation, transform

        class Scenario(enum.Enum):
            MINIMUM = enum.auto()
            AVERAGE = enum.auto()
            MAXIMUM = enum.auto()

    """
    (tmp_path / "model.py").write_text(textwrap.dedent(prefix) + textwrap.dedent(member), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "bilt", "check", "."],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "BILT004" in result.stderr
    assert "non-exhaustive Scenario match" in result.stderr
    assert "AVERAGE, MAXIMUM" in result.stderr
