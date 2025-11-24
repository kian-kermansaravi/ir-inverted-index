"""Interactive search prompt backed by the B-tree inverted index."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from inverted_index import InvertedIndex  # noqa: E402
from preprocess import preprocess  # noqa: E402


DOCS = {
    "d1": "Information retrieval relies on inverted indexes to be fast.",
    "d2": "A B-tree dictionary keeps terms balanced for quick lookups.",
    "d3": "Tokenization and normalization come before indexing in the pipeline.",
    "d4": "Visualization of the B-tree helps debug the dictionary layout.",
    "d5": "Search engines map user terms to documents using inverted indexes.",
}


def build_index() -> InvertedIndex:
    index = InvertedIndex(min_degree=3)
    for doc_id, text in DOCS.items():
        index.add_document(doc_id, text)
    return index


def search_loop(index: InvertedIndex) -> None:
    print("Enter one or more terms. Type 'quit' or 'exit' to leave.\n")
    while True:
        raw = input("term> ").strip()
        if raw.lower() in {"quit", "exit"}:
            print("bye")
            break
        terms = preprocess(raw)
        if not terms:
            print("(no terms found after preprocessing)")
            continue

        for term in terms:
            postings = index.postings(term)
            if postings:
                docs = ", ".join(f"{doc_id} (tf={tf})" for doc_id, tf in sorted(postings.items()))
                print(f"{term}: {docs}")
            else:
                print(f"{term}: no documents found")


def main() -> None:
    index = build_index()
    search_loop(index)


if __name__ == "__main__":
    main()
