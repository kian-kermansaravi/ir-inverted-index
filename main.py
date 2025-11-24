import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from inverted_index import InvertedIndex  # noqa: E402


DOCS = {
    "d1": "Information retrieval relies on inverted indexes to be fast.",
    "d2": "A B-tree dictionary keeps terms balanced for quick lookups.",
    "d3": "Tokenization and normalization come before indexing in the pipeline.",
    "d4": "Visualization of the B-tree helps debug the dictionary layout.",
}


def main() -> None:
    index = InvertedIndex(min_degree=3)

    for doc_id, text in DOCS.items():
        index.add_document(doc_id, text)

    print(index.describe())
    print("\nSample lookups:")
    for term in ["b", "tree", "dictionary", "pipeline", "missing"]:
        postings = index.postings(term)
        if postings:
            hits = ", ".join(f"{doc_id} (tf={tf})" for doc_id, tf in sorted(postings.items()))
            print(f"- '{term}': {hits}")
        else:
            print(f"- '{term}': no match")


if __name__ == "__main__":
    main()
