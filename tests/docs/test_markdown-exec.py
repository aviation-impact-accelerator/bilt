import logging
import re
from html import unescape
from pathlib import Path

import pytest
from markdown import Markdown
from markdown_exec import formatter, validator

DOCS = Path("docs")
VISIBLE_EXAMPLES = {
    "docs/guide/backends.md": ("class Motion(metaclass=Interface):", "with JAX_BACKEND, motion_model:"),
    "docs/guide/dimensions-domains-and-tensors.md": ("class FlightEnergy(metaclass=Interface):", "def select_route("),
    "docs/guide/models-to-computation.md": ("def gamma(", "class Pair(metaclass=Interface):"),
    "docs/getting-started/first-model.md": ("class Motion(metaclass=Interface):", "def motion_relation("),
    "docs/guide/transforms-and-implementations.md": (
        "class Drag(metaclass=Interface):",
        "def drag_coefficient(",
        "def drag_polar(",
    ),
    "docs/guide/systems-of-systems.md": ("class Resistor(metaclass=Interface):", "class Board(metaclass=Interface):"),
}


@pytest.mark.parametrize(
    "path",
    sorted(DOCS.rglob("*.md")),
    ids=str,
)
def test_executable_documentation(
    path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    renderer = Markdown(
        extensions=["pymdownx.superfences"],
        extension_configs={
            "pymdownx.superfences": {
                "custom_fences": [
                    {
                        "name": "python",
                        "class": "python",
                        "validator": validator,
                        "format": formatter,
                    },
                ],
            },
        },
    )

    with caplog.at_level(logging.WARNING):
        rendered = renderer.convert(path.read_text(encoding="utf-8"))

    failures = [record.getMessage() for record in caplog.records if "exited with errors" in record.getMessage()]

    if failures:
        pytest.fail(
            f"{path}: {len(failures)} executable code block(s) failed:\n\n"
            + "\n\n".join(f"--- Error {index} ---\n{message}" for index, message in enumerate(failures, start=1)),
            pytrace=False,
        )

    for snippet in VISIBLE_EXAMPLES.get(path.as_posix(), ()):
        code_blocks = re.findall(r"<pre\b[^>]*>(.*?)</pre>", rendered, flags=re.DOTALL)
        assert any(snippet in unescape(re.sub(r"<[^>]+>", "", block)) for block in code_blocks), (
            f"{path}: missing visible source code: {snippet}"
        )
