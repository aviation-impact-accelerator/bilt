"""Check that Zensical preserves math and Mermaid diagrams for browser rendering."""

import re
import sys
from html import unescape
from pathlib import Path

from markdown import Markdown

DOCS = Path("docs")
SITE = Path("site")
MINIMUM_EXPRESSIONS = 9
MINIMUM_DIAGRAMS = 14
# Strip executable and ordinary fenced blocks before parsing prose. A standalone
# Markdown renderer does not recognise all of Zensical's executable fence options.
FENCED_CODE = re.compile(r"^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[ \t]*$", re.MULTILINE | re.DOTALL)
MERMAID_SOURCE = re.compile(r"^ {0,3}(?:`{3,}|~{3,})mermaid[ \t]*$")
MERMAID_HTML = re.compile(r'<pre\b[^>]*\bclass="mermaid"[^>]*>\s*<code>(.*?)</code>\s*</pre>', re.DOTALL)
MATH = re.compile(r'<(?:div|span) class="arithmatex">(.*?)</(?:div|span)>', re.DOTALL)


def expressions(html: str) -> list[str]:
    """Return MathJax-ready expressions with normalised whitespace."""
    return [" ".join(unescape(match).split()) for match in MATH.findall(html)]


def main() -> int:
    """Compare the rendered site against math and diagrams in its Markdown sources."""
    total = 0
    diagrams = 0
    for source in sorted(DOCS.rglob("*.md")):
        markdown = source.read_text(encoding="utf-8")
        expected_diagrams = [
            " ".join(match.group(0).split("\n")[1:-1]).split()
            for match in FENCED_CODE.finditer(markdown)
            if MERMAID_SOURCE.fullmatch(match.group(0).split("\n", 1)[0])
        ]
        prose = FENCED_CODE.sub("\n", markdown)
        expected = expressions(
            Markdown(
                extensions=["pymdownx.arithmatex"],
                extension_configs={"pymdownx.arithmatex": {"generic": True}},
            ).convert(prose)
        )
        if not expected and not expected_diagrams:
            continue

        relative = source.relative_to(DOCS)
        page = SITE / relative.with_suffix("") / "index.html"
        if source.name == "index.md":
            page = SITE / relative.parent / "index.html"
        html = page.read_text(encoding="utf-8")
        actual = expressions(html)
        actual_diagrams = [unescape(match).split() for match in MERMAID_HTML.findall(html)]
        if actual_diagrams != expected_diagrams:
            sys.stderr.write(
                f"{source}: expected {len(expected_diagrams)} Mermaid diagrams, rendered {len(actual_diagrams)}\n"
            )
            return 1
        diagrams += len(expected_diagrams)
        if actual != expected:
            sys.stderr.write(f"{source}: expected {expected!r}, rendered {actual!r}\n")
            return 1
        total += len(expected)

    if total < MINIMUM_EXPRESSIONS:
        sys.stderr.write(f"Expected at least nine math expressions, found {total}\n")
        return 1
    if diagrams < MINIMUM_DIAGRAMS:
        sys.stderr.write(f"Expected at least {MINIMUM_DIAGRAMS} Mermaid diagrams, found {diagrams}\n")
        return 1
    sys.stdout.write(f"Verified {total} math expressions and {diagrams} Mermaid diagrams in the built documentation\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
