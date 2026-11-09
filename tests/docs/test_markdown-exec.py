import logging
from pathlib import Path

import pytest
from markdown import Markdown
from markdown_exec import formatter, validator

DOCS = Path("docs")


@pytest.mark.parametrize(
    "path",
    sorted(DOCS.rglob("*.md")),
    ids=lambda path: str(path),  # noqa: PLW0108
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
        renderer.convert(path.read_text(encoding="utf-8"))

    failures = [record.getMessage() for record in caplog.records if "exited with errors" in record.getMessage()]

    if failures:
        pytest.fail(
            f"{path}: {len(failures)} executable code block(s) failed:\n\n"
            + "\n\n".join(f"--- Error {index} ---\n{message}" for index, message in enumerate(failures, start=1)),
            pytrace=False,
        )
