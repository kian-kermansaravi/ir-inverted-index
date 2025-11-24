# src/pdf_reader.py
from typing import List
from pathlib import Path
import PyPDF2

def read_pdf_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    text_parts: List[str] = []
    with open(p, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                pass
    return "\n".join(text_parts)
