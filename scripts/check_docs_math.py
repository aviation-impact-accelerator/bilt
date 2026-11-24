"""Check that Zensical preserves mathematical expressions for MathJax."""

import re
import sys
from html import unescape
from pathlib import Path

from markdown import Markdown

DOCS = Path("docs")
SITE = Path("site")
MINIMUM_EXPRESSIONS = 9
MATH = re.compile(r'<(?:div|span) class="arithmatex">(.*?)</(?:div|span)>', re.DOTALL)


def expressions(html: str) -> list[str]:
    """Return MathJax-ready expressions with normalised whitespace."""
    return [" ".join(unescape(match).split()) for match in MATH.findall(html)]


def main() -> int:
    """Compare the rendered site against the math in its Markdown sources."""
    total = 0
    for source in sorted(DOCS.rglob("*.md")):
        expected = expressions(
            Markdown(
                extensions=["pymdownx.superfences", "pymdownx.arithmatex"],
                extension_configs={"pymdownx.arithmatex": {"generic": True}},
            ).convert(source.read_text(encoding="utf-8"))
        )
        if not expected:
            continue

        relative = source.relative_to(DOCS)
        page = SITE / relative.with_suffix("") / "index.html"
        if source.name == "index.md":
            page = SITE / relative.parent / "index.html"
        actual = expressions(page.read_text(encoding="utf-8"))
        if actual != expected:
            sys.stderr.write(f"{source}: expected {expected!r}, rendered {actual!r}\n")
            return 1
        total += len(expected)

    if total < MINIMUM_EXPRESSIONS:
        sys.stderr.write(f"Expected at least nine math expressions, found {total}\n")
        return 1
    sys.stdout.write(f"Verified {total} math expressions in the built documentation\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
