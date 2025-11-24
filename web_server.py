"""Minimal Flask server exposing inverted index search with a static frontend.""" 
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from inverted_index import InvertedIndex  # noqa: E402
from preprocess import preprocess  # noqa: E402


DOCS: Dict[str, str] = {
    "d1": "Information retrieval relies on inverted indexes to be fast.",
    "d2": "A B-tree dictionary keeps terms balanced for quick lookups.",
    "d3": "Tokenization and normalization come before indexing in the pipeline.",
    "d4": "Visualization of the B-tree helps debug the dictionary layout.",
}

index = InvertedIndex(min_degree=3)
for doc_id, text in DOCS.items():
    index.add_document(doc_id, text)

app = Flask(__name__, static_folder="web/static", static_url_path="")


@app.route("/")
def root() -> object:
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/search")
def api_search() -> object:
    query = request.args.get("q", "")
    terms = preprocess(query)
    scores: Dict[str, int] = defaultdict(int)
    matches: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for term in terms:
        postings = index.postings(term)
        for doc_id, tf in postings.items():
            scores[doc_id] += tf
            matches[doc_id].append({"term": term, "tf": tf})

    results = [
        {
            "doc_id": doc_id,
            "score": scores[doc_id],
            "text": DOCS[doc_id],
            "matches": matches[doc_id],
        }
        for doc_id in scores
    ]
    results.sort(key=lambda item: item["score"], reverse=True)
    return jsonify({"query": query, "terms": terms, "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
