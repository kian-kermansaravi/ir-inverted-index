"""Text normalization and tokenization helpers."""
import re
from typing import List

_punct_re = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_space_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = _punct_re.sub(" ", cleaned)
    cleaned = _space_re.sub(" ", cleaned).strip()
    return cleaned


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [token for token in text.split(" ") if token]


def preprocess(text: str) -> List[str]:
    return tokenize(normalize_text(text))
