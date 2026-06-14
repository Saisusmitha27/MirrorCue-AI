from __future__ import annotations

from pathlib import Path

import pdfplumber
from PyPDF2 import PdfReader


def _clean_page_lines(lines: list[str]) -> list[str]:
    if not lines:
        return []

    trimmed = [line.strip() for line in lines if line.strip()]
    if len(trimmed) <= 4:
        return trimmed

    start = 1 if len(trimmed[0]) < 20 else 0
    end = len(trimmed) - 1 if len(trimmed[-1]) < 20 else len(trimmed)
    return trimmed[start:end]


def extract_text_from_pdf(file_path: str | Path) -> str:
    path = Path(file_path)
    text_parts: list[str] = []

    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            lines = _clean_page_lines(page_text.splitlines())
            text_parts.append("\n".join(lines).strip())

    extracted = "\n\n".join(part for part in text_parts if part).strip()
    if len(extracted) >= 100:
        return extracted

    fallback_parts: list[str] = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        fallback_text = page.extract_text() or ""
        lines = _clean_page_lines(fallback_text.splitlines())
        fallback_parts.append("\n".join(lines).strip())

    return "\n\n".join(part for part in fallback_parts if part).strip()
