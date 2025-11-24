"""Small usage example for the inverted index."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from inverted_index import InvertedIndex  # noqa: E402


def build_example_index() -> InvertedIndex:
    docs = {
        "doc1": "The quick brown fox jumps over the lazy dog.",
        "doc2": "Quick movements help the fox evade predators.",
        "doc3": "Lazy afternoons are perfect for reading about foxes.",
    }
    index = InvertedIndex(min_degree=3)
    for doc_id, text in docs.items():
        index.add_document(doc_id, text)
    return index


def main() -> None:
    index = build_example_index()
    print(index.describe())
    print("\nLookup results:")
    for term in ["quick", "fox", "lazy", "missing"]:
        postings = index.postings(term)
        hits = ", ".join(f"{doc_id}:tf={tf}" for doc_id, tf in sorted(postings.items())) or "none"
        print(f"- {term}: {hits}")


if __name__ == "__main__":
    main()
