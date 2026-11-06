import re
import typing
from pathlib import Path

import pytest

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

DOCS = Path(__file__).resolve().parents[2] / "docs"

OPENING_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(?:\s*)(python|py)\b(.*)$")
EXEC_OPTION = re.compile(r"""\bexec\s*=\s*["']?(on|off|true|false|1|0|yes|no)["']?""")


def python_fences(path: Path) -> Iterator[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    active_fence = None

    for line_number, line in enumerate(lines, start=1):
        if active_fence is not None:
            marker, length = active_fence
            if re.fullmatch(
                rf"\s{{0,3}}{re.escape(marker)}{{{length},}}\s*",
                line,
            ):
                active_fence = None
            continue

        match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if not match:
            continue

        fence = match.group(1)
        active_fence = (fence[0], len(fence))

        opening = OPENING_FENCE.match(line)
        if opening:
            yield line_number, opening.group(3)


@pytest.mark.parametrize(
    "path",
    sorted(DOCS.rglob("*.md")),
    ids=lambda path: str(path.relative_to(DOCS)),
)
def test_python_fences_have_execution_policy(path: Path) -> None:
    for line_number, options in python_fences(path):
        assert EXEC_OPTION.search(options), f'{path}:{line_number}: Python block must specify exec="on" or exec="off"'
