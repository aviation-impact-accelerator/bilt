"""Generate Bilt's README figures from editable, procedural specifications.

No third-party dependencies. The committed SVGs are generated outputs, never
hand-edited artwork. Regenerate them after changing readme_figures.json:

    python scripts/generate_readme_svgs.py
    python scripts/generate_readme_svgs.py --check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "scripts" / "readme_figures.json"
OUTPUT_DIR = ROOT / ".github" / "assets"
SPRING_TURNS = 8
SPRING_AMPLITUDE = 12
NORMAL_POINTS = 90


def render_rectangle(args: list, palette: dict) -> str:
    """Render a filled or outline system boundary."""
    x, y, width, height, style = args
    if style == "boundary":
        return (
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'fill="none" stroke="{palette["rule"]}" stroke-width="1" rx="4"/>'
        )
    fill = palette["soft"] if style == "tint" else palette["paper"]
    stroke = palette["rule"] if style == "tint" else palette["ink"]
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'


def render_curve(kind: str, args: list, palette: dict, colors: dict) -> str:
    """Render procedural springs or a Gaussian curve."""
    if kind == "spring":
        left, right, y, _style = args
        points = [(left, y), (left + 17, y)]
        for index in range(SPRING_TURNS * 2 + 1):
            x = left + 17 + (right - left - 34) * index / (SPRING_TURNS * 2)
            height = y - SPRING_AMPLITUDE if index % 2 == 0 else y + SPRING_AMPLITUDE
            points.append((x, height))
        points.extend([(right - 17, y), (right, y)])
        data = " ".join(f"{'M' if index == 0 else 'L'}{x:.1f} {height:.1f}" for index, (x, height) in enumerate(points))
        return f'<path d="{data}" fill="none" stroke="{palette["ink"]}" stroke-width="1.8" stroke-linejoin="round"/>'
    if kind == "normal":
        left, right, baseline, amplitude, _style = args
        points = [
            (
                left + (right - left) * index / NORMAL_POINTS,
                baseline - amplitude * math.exp(-0.5 * ((index - NORMAL_POINTS / 2) / (NORMAL_POINTS / 6)) ** 2),
            )
            for index in range(NORMAL_POINTS + 1)
        ]
        data = " ".join(f"{'M' if index == 0 else 'L'}{x:.1f} {height:.1f}" for index, (x, height) in enumerate(points))
        return f'<path d="{data}" fill="none" stroke="{colors[_style]}" stroke-width="2.3"/>'
    msg = f"Unknown curve: {kind}"
    raise ValueError(msg)


def render_operation(operation: list, palette: dict, colors: dict) -> str:
    """Render a text, geometry, or mathematical-curve instruction."""
    kind, *args = operation
    if kind == "text":
        x, y, value, size, style, anchor = args
        if style in ("math", "result"):
            font = "STIX Two Math, Cambria Math, Georgia, serif"
        else:
            font = "Arial, Helvetica, sans-serif"
        spacing = ' letter-spacing="0.9"' if style == "small" else ""
        align = f' text-anchor="{anchor}"' if anchor != "start" else ""
        return (
            f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
            f'fill="{colors.get(style, palette["ink"])}"{spacing}{align}>'
            f"{escape(value)}</text>"
        )
    if kind == "line":
        x1, y1, x2, y2, style, weight = args
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colors[style]}" stroke-width="{weight}"/>'
    if kind == "path":
        data, style, weight = args
        return (
            f'<path d="{data}" fill="none" stroke="{colors[style]}" '
            f'stroke-width="{weight}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    if kind == "rect":
        return render_rectangle(args, palette)
    if kind == "circle":
        x, y, radius, style = args
        fill = palette["paper"] if style == "outline" else colors[style]
        border = f' stroke="{palette["muted"]}" stroke-width="1.3"' if style == "outline" else ""
        return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}"{border}/>'
    if kind in ("spring", "normal"):
        return render_curve(kind, args, palette, colors)
    msg = f"Unknown drawing operation: {kind}"
    raise ValueError(msg)


def svg_for(diagram: dict, palette: dict) -> str:
    """Render a diagram specification as a self-contained, accessible SVG."""
    colors = {
        key: palette.get(key, palette["ink"])
        for key in ("ink", "muted", "rule", "accent", "result", "soft", "outline", "small")
    }
    colors["result"] = palette["accent"]
    colors["small"] = palette["muted"]
    width, height = diagram["width"], diagram["height"]
    output = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">'
        ),
        f'<title id="title">{escape(diagram["title"])}</title>',
        f'<desc id="description">{escape(diagram["description"])}</desc>',
        f'<rect width="{width}" height="{height}" fill="{palette["paper"]}"/>',
    ]
    output.extend(render_operation(operation, palette, colors) for operation in diagram["operations"])
    output.extend(["</svg>", ""])
    return "\n".join(output)


def main() -> int:
    """Regenerate the SVGs or verify them against committed versions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the committed SVGs are stale")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    specs = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not args.check:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    stale = []
    for diagram in specs["diagrams"]:
        target = args.output_dir / diagram["name"]
        content = svg_for(diagram, specs["palette"])
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                stale.append(str(target))
        else:
            target.write_text(content, encoding="utf-8")
            sys.stdout.write(f"Generated {target}\n")

    if stale:
        sys.stderr.write("Stale diagrams: " + ", ".join(stale) + "\n")
        return 1
    if args.check:
        sys.stdout.write(f"Verified {len(specs['diagrams'])} generated SVGs\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
