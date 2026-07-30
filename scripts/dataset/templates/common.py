from pathlib import Path

import fitz

def render_pdf_to_png(
    pdf_path: Path,
    png_path: Path,
    scale: float = 2.0,
) -> None:
    document = fitz.open(pdf_path)

    try:
        page = document.load_page(0)
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            alpha=False,
        )
        pixmap.save(png_path)
    finally:
        document.close()