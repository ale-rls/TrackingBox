"""Generate a printable checkerboard target for lens calibration.

Examples:
    python scripts/generate_checkerboard.py --output checkerboard_a4.pdf
    python scripts/generate_checkerboard.py --output checkerboard_a4.svg

The board dimensions are inner-corner counts, not printed square counts. A
9x6 inner-corner board prints as 10x7 alternating squares.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.sax.saxutils import escape

MM_PER_INCH = 25.4
PDF_POINTS_PER_INCH = 72.0

PAGE_SIZES_MM = {
    "a4": (210.0, 297.0),
}


def main() -> int:
    args = _parse_args()
    layout = make_layout(
        page=args.page,
        orientation=args.orientation,
        board_cols=args.board_cols,
        board_rows=args.board_rows,
        square_size_mm=args.square_size_mm,
        margin_mm=args.margin_mm,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".svg":
        write_svg(output, layout)
    elif output.suffix.lower() == ".pdf":
        write_pdf(output, layout)
    else:
        raise SystemExit("Output must end in .pdf or .svg")

    print(f"Wrote {layout['label']} checkerboard to {output}")
    print(
        "Print at 100% / actual size. "
        f"Use --board-cols {args.board_cols} --board-rows {args.board_rows} for calibration."
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/checkerboard_a4.pdf")
    parser.add_argument("--page", choices=sorted(PAGE_SIZES_MM), default="a4")
    parser.add_argument(
        "--orientation",
        choices=("landscape", "portrait"),
        default="landscape",
        help="A4 landscape is recommended for the default 9x6 board",
    )
    parser.add_argument("--board-cols", type=int, default=9, help="Inner checkerboard columns")
    parser.add_argument("--board-rows", type=int, default=6, help="Inner checkerboard rows")
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument("--margin-mm", type=float, default=10.0)
    return parser.parse_args()


def make_layout(
    *,
    page: str,
    orientation: str,
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
    margin_mm: float,
) -> dict[str, object]:
    if board_cols < 2 or board_rows < 2:
        raise ValueError("board rows/columns must be at least 2 inner corners")
    if square_size_mm <= 0:
        raise ValueError("square size must be positive")
    if margin_mm < 0:
        raise ValueError("margin must be zero or positive")

    page_w_mm, page_h_mm = PAGE_SIZES_MM[page]
    if orientation == "landscape":
        page_w_mm, page_h_mm = page_h_mm, page_w_mm

    squares_x = board_cols + 1
    squares_y = board_rows + 1
    board_w_mm = squares_x * square_size_mm
    board_h_mm = squares_y * square_size_mm
    usable_w_mm = page_w_mm - margin_mm * 2.0
    usable_h_mm = page_h_mm - margin_mm * 2.0
    if board_w_mm > usable_w_mm or board_h_mm > usable_h_mm:
        raise ValueError(
            "checkerboard does not fit page: "
            f"{board_w_mm:.1f}x{board_h_mm:.1f}mm board, "
            f"{usable_w_mm:.1f}x{usable_h_mm:.1f}mm usable page"
        )

    x0_mm = (page_w_mm - board_w_mm) / 2.0
    y0_mm = (page_h_mm - board_h_mm) / 2.0
    label = f"{board_cols}x{board_rows} inner corners, {square_size_mm:g}mm squares"
    return {
        "page_w_mm": page_w_mm,
        "page_h_mm": page_h_mm,
        "x0_mm": x0_mm,
        "y0_mm": y0_mm,
        "square_size_mm": square_size_mm,
        "squares_x": squares_x,
        "squares_y": squares_y,
        "board_cols": board_cols,
        "board_rows": board_rows,
        "label": label,
    }


def black_squares(layout: dict[str, object]) -> list[tuple[float, float, float]]:
    x0 = float(layout["x0_mm"])
    y0 = float(layout["y0_mm"])
    size = float(layout["square_size_mm"])
    squares_x = int(layout["squares_x"])
    squares_y = int(layout["squares_y"])

    squares = []
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                squares.append((x0 + col * size, y0 + row * size, size))
    return squares


def write_svg(path: Path, layout: dict[str, object]) -> None:
    page_w = float(layout["page_w_mm"])
    page_h = float(layout["page_h_mm"])
    label = escape(str(layout["label"]))
    rects = "\n".join(
        f'  <rect x="{x:.4f}" y="{y:.4f}" width="{size:.4f}" height="{size:.4f}" />'
        for x, y, size in black_squares(layout)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w}mm" '
        f'height="{page_h}mm" viewBox="0 0 {page_w} {page_h}">\n'
        f"""\
  <rect width="100%" height="100%" fill="white" />
  <g fill="black">
{rects}
  </g>
  <text x="{page_w / 2:.4f}" y="{page_h - 6:.4f}" text-anchor="middle"
        font-family="Arial, sans-serif" font-size="4">{label} - print at 100%</text>
</svg>
"""
    )
    path.write_text(svg, encoding="utf-8")


def write_pdf(path: Path, layout: dict[str, object]) -> None:
    page_w_pt = _mm_to_pt(float(layout["page_w_mm"]))
    page_h_pt = _mm_to_pt(float(layout["page_h_mm"]))
    commands = ["q", "0 0 0 rg"]
    for x_mm, y_mm, size_mm in black_squares(layout):
        x_pt = _mm_to_pt(x_mm)
        y_pt = page_h_pt - _mm_to_pt(y_mm + size_mm)
        size_pt = _mm_to_pt(size_mm)
        commands.append(f"{_num(x_pt)} {_num(y_pt)} {_num(size_pt)} {_num(size_pt)} re f")
    commands.append("Q")
    commands.extend(
        [
            "BT",
            "/F1 10 Tf",
            "0 0 0 rg",
            f"1 0 0 1 {_num(page_w_pt / 2.0)} {_num(_mm_to_pt(5.0))} Tm",
            f"({_pdf_escape(str(layout['label']) + ' - print at 100%')}) Tj",
            "ET",
        ]
    )
    stream = "\n".join(commands).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_num(page_w_pt)} {_num(page_h_pt)}] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ).encode("ascii"),
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    _write_pdf_objects(path, objects)


def _write_pdf_objects(path: Path, objects: list[bytes]) -> None:
    chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]
    current = len(chunks[0])
    for index, body in enumerate(objects, start=1):
        offsets.append(current)
        chunk = f"{index} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
        chunks.append(chunk)
        current += len(chunk)

    xref_offset = current
    xref_lines = [b"xref\n", f"0 {len(objects) + 1}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref_lines.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        b"trailer\n"
        + f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii")
        + b"startxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )
    path.write_bytes(b"".join(chunks + xref_lines + [trailer]))


def _mm_to_pt(value: float) -> float:
    return value / MM_PER_INCH * PDF_POINTS_PER_INCH


def _num(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


if __name__ == "__main__":
    raise SystemExit(main())
