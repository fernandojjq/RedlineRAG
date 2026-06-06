"""Render the RedlineRAG architecture diagram as a standalone SVG.

We use a pure-Python SVG builder so the diagram has zero extra
dependencies. SVG scales perfectly on every display, embeds inline in
markdown, and weighs a few KB on disk.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUTPUT_PATH = Path(__file__).resolve().parent / "diagram.svg"

# Layout constants. SVG origin is top-left.
WIDTH, HEIGHT = 1400, 720
STAGE_Y = 380
STAGE_HEIGHT = 180
STAGE_WIDTH = 260
GAP = 30
STAGES = [
    {"x": 30,                "label": "1. INGEST",  "sub": ".txt / .md / .pdf / .docx",       "color": "#1e3a5f"},
    {"x": 30 + 1*(STAGE_WIDTH+GAP), "label": "2. CHUNK",   "sub": "Recursive splitter\n600 chars / 80 overlap", "color": "#2d4a2d"},
    {"x": 30 + 2*(STAGE_WIDTH+GAP), "label": "3. INDEX",   "sub": "TF-IDF + L2 norm\njoblib / npz",            "color": "#5f3a1e"},
    {"x": 30 + 3*(STAGE_WIDTH+GAP), "label": "4. QUERY",   "sub": "Cosine top-k\nsimilarity floor 0.05",        "color": "#4a2d5f"},
    {"x": 30 + 4*(STAGE_WIDTH+GAP), "label": "5. AUDIT",   "sub": "10 risk families",                            "color": "#5f1e1e"},
]


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#222222", rx: float = 14) -> str:
    return (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" />'
    )


def _text(x: float, y: float, content: str, *, size: int = 12, weight: str = "normal", color: str = "#222") -> str:
    escaped = escape(content).replace("\n", "</tspan><tspan x='{}' dy='1.1em'>".format(x))
    return (
        f'  <text x="{x}" y="{y}" text-anchor="middle" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">'
        f"<tspan>{escaped}</tspan></text>"
    )


def _arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#333", width: float = 2.0, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    return (
        f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}"{dash} marker-end="url(#arrow)" />'
    )


def render_diagram() -> Path:
    """Build the architecture diagram and save it as a self-contained SVG."""
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}">'
    )

    # Background.
    parts.append(f'  <rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff" />')

    # Arrow marker definition.
    parts.append(
        '  <defs>'
        '    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        '           markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333" />'
        '    </marker>'
        '  </defs>'
    )

    # Title.
    parts.append(_text(WIDTH / 2, 50,
                       "RedlineRAG - Local RAG Pipeline for Terms of Service Risk Auditing",
                       size=22, weight="bold", color="#111"))
    parts.append(_text(WIDTH / 2, 80,
                       "100% offline  ·  no model download  ·  no GPU  ·  no API keys  ·  no system pollution",
                       size=13, color="#555"))

    # Stage boxes.
    arrow_y = STAGE_Y + STAGE_HEIGHT / 2
    for idx, stage in enumerate(STAGES):
        x = stage["x"]
        parts.append(_rect(x, STAGE_Y, STAGE_WIDTH, STAGE_HEIGHT, fill=stage["color"]))
        parts.append(_text(x + STAGE_WIDTH / 2, STAGE_Y + 38, stage["label"],
                           size=16, weight="bold", color="#fff"))
        # Subtitle may contain a newline - handled in _text.
        parts.append(_text(x + STAGE_WIDTH / 2, STAGE_Y + 90, stage["sub"],
                           size=12, color="#fff"))
        # Arrow to next stage.
        if idx < len(STAGES) - 1:
            parts.append(_arrow(
                x + STAGE_WIDTH + 2, arrow_y,
                x + STAGE_WIDTH + GAP - 2, arrow_y,
                color="#333", width=2.5,
            ))

    # Mock generator callout under the ingest stage.
    mock_x, mock_y, mock_w, mock_h = 30, 130, STAGE_WIDTH, 130
    parts.append(_rect(mock_x, mock_y, mock_w, mock_h, fill="#fef9c3", stroke="#aa8800"))
    parts.append(_text(mock_x + mock_w / 2, mock_y + 35, "Mock ToS Generator",
                       size=14, weight="bold", color="#3a2e00"))
    parts.append(_text(mock_x + mock_w / 2, mock_y + 75,
                       "3 sample agreements",
                       size=11, color="#3a2e00"))
    parts.append(_text(mock_x + mock_w / 2, mock_y + 95,
                       "with planted legal traps",
                       size=11, color="#3a2e00"))
    parts.append(_arrow(
        mock_x + mock_w / 2, mock_y + mock_h + 2,
        mock_x + mock_w / 2, STAGE_Y - 2,
        color="#aa8800", width=2.0, dashed=True,
    ))

    # User query callout above the QUERY stage.
    query_stage_x = STAGES[3]["x"]
    query_x = query_stage_x + STAGE_WIDTH / 2
    parts.append(_text(query_x, 130, "User question", size=13, weight="bold", color="#222"))
    parts.append(_text(query_x, 165, "binding arbitration", size=12, color="#222"))
    parts.append(_text(query_x, 195, "data selling", size=12, color="#222"))
    parts.append(_text(query_x, 225, "device fingerprinting", size=12, color="#222"))
    parts.append(_arrow(query_x, 245, query_x, STAGE_Y - 2, color="#333", width=2.0))

    # Output callout below the AUDIT stage.
    audit_stage_x = STAGES[4]["x"]
    audit_x = audit_stage_x + STAGE_WIDTH / 2
    parts.append(_rect(audit_x - 110, 595, 220, 100, fill="#fff5f5", stroke="#5f1e1e"))
    parts.append(_text(audit_x, 625, "Risk Report", size=14, weight="bold", color="#5f1e1e"))
    parts.append(_text(audit_x, 655, "CRITICAL  ·  HIGH", size=12, color="#5f1e1e", weight="bold"))
    parts.append(_text(audit_x, 678, "MEDIUM  ·  LOW", size=12, color="#5f1e1e", weight="bold"))
    parts.append(_arrow(audit_x, STAGE_Y + STAGE_HEIGHT + 2, audit_x, 595, color="#5f1e1e", width=2.0))

    # Storage badge below the INDEX stage.
    idx_stage_x = STAGES[2]["x"]
    storage_x = idx_stage_x + STAGE_WIDTH / 2
    parts.append(_rect(storage_x - 130, 620, 260, 80, fill="#eef2f7", stroke="#5f3a1e"))
    parts.append(_text(storage_x, 648, "data/vector_store/", size=12, weight="bold", color="#222"))
    parts.append(_text(storage_x, 672, "vocabulary.joblib · vectors.npz", size=10, color="#333"))
    parts.append(_text(storage_x, 688, "chunks.jsonl · manifest.json", size=10, color="#333"))
    # Two-way arrow between INDEX and storage.
    parts.append(_arrow(
        storage_x, STAGE_Y + STAGE_HEIGHT + 2,
        storage_x, 620,
        color="#5f3a1e", width=2.0,
    ))

    parts.append("</svg>")

    OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    return OUTPUT_PATH


if __name__ == "__main__":
    path = render_diagram()
    print(f"SVG diagram written to: {path}")
