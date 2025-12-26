from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from inverted_index import InvertedIndex  # noqa: E402
from preprocess import preprocess  # noqa: E402
from retrieval_systems import (  # noqa: E402
    get_retrieval_system, 
    AVAILABLE_SYSTEMS,
    BooleanRetrieval,
    TFIDFRetrieval,
    BM25Retrieval,
    ProbabilisticRetrieval,
)

# Documents storage
DOCS: Dict[str, str] = {}
DOCUMENTS_FOLDER = ROOT / "documents"
DOCUMENTS_FOLDER.mkdir(exist_ok=True)

# Retrieval systems cache
retrieval_systems: Dict[str, object] = {}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.txt', '.md', '.py', '.json', '.csv', '.html', '.xml', '.pdf'}

index = InvertedIndex(min_degree=3)


def extract_text_from_pdf(file_path: Path) -> str:
    if not PDF_SUPPORT:
        return ""
    try:
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting PDF text from {file_path}: {e}")
        return ""


def load_documents_from_folder():
    global DOCS, index
    DOCS.clear()
    index = InvertedIndex(min_degree=3)
    
    print(f"Loading documents from: {DOCUMENTS_FOLDER}")
    
    for file_path in DOCUMENTS_FOLDER.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
            try:
                if file_path.suffix.lower() == '.pdf':
                    text = extract_text_from_pdf(file_path)
                else:
                    text = file_path.read_text(encoding='utf-8')
                
                if text.strip():  # Only add if there's content
                    doc_id = file_path.name
                    DOCS[doc_id] = text
                    index.add_document(doc_id, text)
                    print(f"Loaded: {doc_id} ({len(text)} chars)")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    print(f"Total documents loaded: {len(DOCS)}")
    
    # Initialize retrieval systems
    _init_retrieval_systems()


def _init_retrieval_systems():
    global retrieval_systems
    retrieval_systems.clear()
    
    if DOCS:
        for system_type in AVAILABLE_SYSTEMS.keys():
            try:
                retrieval_systems[system_type] = get_retrieval_system(system_type, index, DOCS)
                print(f"Initialized {system_type} retrieval system")
            except Exception as e:
                print(f"Error initializing {system_type}: {e}")


# Initial load
load_documents_from_folder()

app = Flask(__name__, static_folder="web/static", static_url_path="")


@app.route("/")
def root() -> object:
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/search")
def api_search() -> object:
    query = request.args.get("q", "")
    system_type = request.args.get("system", "tfidf").lower()
    
    if not query.strip():
        return jsonify({"query": "", "terms": [], "results": [], "system": system_type})
    
    # Get the retrieval system
    if system_type not in retrieval_systems:
        # Fallback to simple term frequency
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
                "text": DOCS[doc_id][:500] + ("..." if len(DOCS[doc_id]) > 500 else ""),
                "matches": [m["term"] for m in matches[doc_id]],
            }
            for doc_id in scores
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return jsonify({
            "query": query, 
            "terms": terms, 
            "results": results, 
            "system": "simple",
            "system_name": "Simple Term Frequency"
        })
    
    # Use the selected retrieval system
    retrieval = retrieval_systems[system_type]
    search_results = retrieval.search(query)
    
    results = [
        {
            "doc_id": r.doc_id,
            "score": r.score,
            "text": r.text[:500] + ("..." if len(r.text) > 500 else ""),
            "matches": r.matched_terms,
        }
        for r in search_results
    ]
    
    return jsonify({
        "query": query,
        "terms": preprocess(query),
        "results": results,
        "system": system_type,
        "system_name": retrieval.name
    })


@app.route("/api/systems")
def api_systems():
    return jsonify({
        "systems": [
            {"id": k, "name": v, "available": k in retrieval_systems}
            for k, v in AVAILABLE_SYSTEMS.items()
        ],
        "default": "tfidf"
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    file_path = DOCUMENTS_FOLDER / filename
    file.save(str(file_path))
    
    # Re-index all documents
    load_documents_from_folder()
    
    return jsonify({"success": True, "filename": filename, "total_docs": len(DOCS)})


@app.route("/api/documents")
def api_documents():
    return jsonify({
        "documents": [
            {"id": doc_id, "preview": text[:200] + ("..." if len(text) > 200 else "")}
            for doc_id, text in DOCS.items()
        ],
        "total": len(DOCS)
    })


@app.route("/api/delete/<doc_id>", methods=["DELETE"])
def api_delete(doc_id: str):
    file_path = DOCUMENTS_FOLDER / doc_id
    if file_path.exists():
        file_path.unlink()
        load_documents_from_folder()
        return jsonify({"success": True, "deleted": doc_id})
    return jsonify({"error": "Document not found"}), 404


@app.route("/api/reload", methods=["POST"])
def api_reload():
    load_documents_from_folder()
    return jsonify({"success": True, "total_docs": len(DOCS)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
