import re
from typing import List, Dict

STOPWORDS = {
    "the","and","is","in","to","of","a","an","for","on","that","this","it","with","as","are","was","by"
}

def normalize(text: str) -> str:
    # lowercase, remove excessive whitespace
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    text = normalize(text)
    # remove punctuation (keep alphanum and underscore)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    return tokens

def pre_token_score(doc_meta: Dict) -> float:
    """
    doc_meta: dictionary with optional keys like 'important' (bool), 'length' (int)
    returns a simple pre-token score (can be refined later)
    """
    score = 1.0
    if isinstance(doc_meta, dict):
        if doc_meta.get("important"):
            score += 1.0
        length = int(doc_meta.get("length", 0))
        score += min(length / 1000.0, 1.0)
    return score
