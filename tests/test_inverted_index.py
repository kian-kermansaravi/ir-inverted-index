import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from inverted_index import InvertedIndex  # noqa: E402
from preprocess import normalize_text, tokenize  # noqa: E402


def test_preprocess_pipeline():
    text = "Hello, World!!!  This  is\tTabbed"
    normalized = normalize_text(text)
    assert normalized == "hello world this is tabbed"
    assert tokenize(normalized) == ["hello", "world", "this", "is", "tabbed"]


def test_index_postings_and_df():
    index = InvertedIndex(min_degree=2)
    index.add_document("d1", "red blue red")
    index.add_document("d2", "blue green")

    red_postings = index.postings("red")
    blue_postings = index.postings("blue")
    missing_postings = index.postings("missing")

    assert red_postings == {"d1": 2}
    assert blue_postings == {"d1": 1, "d2": 1}
    assert missing_postings == {}

    assert "red" in index
    assert "green" in index
    assert "missing" not in index


def test_traversal_order_sorted_terms():
    index = InvertedIndex(min_degree=2)
    index.add_document("d1", "c b a")
    index.add_document("d2", "d e f")
    terms = [term for term, _ in index.iter_terms()]
    assert terms == sorted(terms)


def test_duplicate_terms_merge_across_documents():
    index = InvertedIndex(min_degree=2)
    index.add_document("d1", "dictionary")
    index.add_document("d2", "dictionary")

    postings = index.postings("dictionary")
    assert postings == {"d1": 1, "d2": 1}

    terms = [term for term, _ in index.iter_terms()]
    assert terms.count("dictionary") == 1


if __name__ == "__main__":
    pytest.main([__file__])
