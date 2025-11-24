from pathlib import Path
from typing import List
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
                t = page.extract_text() or ""
                text_parts.append(t)
            except Exception:
                pass
    return "\n".join(text_parts)
