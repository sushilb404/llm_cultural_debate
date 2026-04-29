from __future__ import annotations

import textwrap
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "proposal_materials" / "multicultural_llm_debate_outline.md"
DEFAULT_OUTPUT = REPO_ROOT / "proposal_materials" / "multicultural_llm_debate_outline.pdf"


def strip_markdown(line: str) -> tuple[str, str]:
    stripped = line.strip()
    if stripped.startswith("# "):
        return stripped[2:], "title"
    if stripped.startswith("## "):
        return stripped[3:], "section"
    if stripped.startswith("### "):
        return stripped[4:], "subsection"
    return line.replace("**", "").replace("`", ""), "body"


def render_page(pdf: PdfPages, lines: list[tuple[str, str]], page_number: int) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.94
    for text, style in lines:
        if style == "title":
            ax.text(0.08, y, text, ha="left", va="top", fontsize=18, weight="bold", wrap=True)
            y -= 0.055
        elif style == "section":
            ax.text(0.08, y, text, ha="left", va="top", fontsize=13, weight="bold", wrap=True)
            y -= 0.038
        elif style == "subsection":
            ax.text(0.08, y, text, ha="left", va="top", fontsize=11.5, weight="bold", wrap=True)
            y -= 0.032
        else:
            ax.text(0.08, y, text, ha="left", va="top", fontsize=9.5, wrap=True, family="DejaVu Sans")
            y -= 0.024

    ax.text(0.5, 0.035, str(page_number), ha="center", va="center", fontsize=8, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def paginate(source_text: str) -> list[list[tuple[str, str]]]:
    pages: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    used = 0.0

    for raw in source_text.splitlines():
        text, style = strip_markdown(raw)
        if not text.strip():
            wrapped = [""]
            cost_per_line = 0.018
        elif style == "title":
            wrapped = textwrap.wrap(text, width=54) or [text]
            cost_per_line = 0.05
        elif style == "section":
            wrapped = textwrap.wrap(text, width=68) or [text]
            cost_per_line = 0.036
        elif style == "subsection":
            wrapped = textwrap.wrap(text, width=76) or [text]
            cost_per_line = 0.032
        else:
            wrapped = textwrap.wrap(text, width=98, subsequent_indent="  ") or [text]
            cost_per_line = 0.024

        block_cost = len(wrapped) * cost_per_line + (0.012 if style in {"section", "subsection"} else 0.0)
        if current and used + block_cost > 0.86:
            pages.append(current)
            current = []
            used = 0.0

        for index, wrapped_line in enumerate(wrapped):
            current.append((wrapped_line, style if index == 0 else "body"))
        used += block_cost

    if current:
        pages.append(current)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a simple Markdown proposal file to PDF.")
    parser.add_argument("--input", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    if not source.is_absolute():
        source = REPO_ROOT / source
    if not output.is_absolute():
        output = REPO_ROOT / output

    source_text = source.read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    pages = paginate(source_text)
    with PdfPages(output) as pdf:
        for index, page in enumerate(pages, start=1):
            render_page(pdf, page, index)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
