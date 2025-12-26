# Inverted Index with B-Tree Dictionary

A comprehensive Information Retrieval (IR) system that implements multiple retrieval models with a B-tree based inverted index. This project demonstrates various IR concepts including Boolean retrieval, TF-IDF ranking, BM25, and probabilistic models.

## 🚀 Features

### Core Features
- Text normalization (lowercasing, punctuation strip, whitespace squeeze) before tokenization.
- Simple whitespace tokenizer with empty-token filtering.
- In-memory B-tree dictionary that stores each term with its postings list and term frequency per document.
- Tree visualization helper to print internal nodes and leaves in level order.
- PDF document support with text extraction.
- Web-based UI for searching and document management.

### Retrieval Systems (نمره اضافه ⭐)

| System | Description |
|--------|-------------|
| **Boolean** | Supports AND, OR, NOT operators. No ranking. |
| **TF-IDF** | Term Frequency × Inverse Document Frequency ranking |
| **BM25** | Best Matching 25 - State-of-the-art probabilistic ranking |
| **Probabilistic** | Binary Independence Model (BIM) |

### Boolean Query Examples
```
brain AND tumor          # Documents containing both terms
information OR retrieval # Documents containing either term
deep NOT learning        # Documents with "deep" but not "learning"
brain AND tumor NOT cancer # Complex boolean expression
```

## Project Structure
```
├── src/
│   ├── btree.py              # B-tree implementation for term dictionary
│   ├── preprocess.py         # Text normalization and tokenization
│   ├── inverted_index.py     # Core inverted index with B-tree dictionary
│   └── retrieval_systems.py  # Multiple IR models (Boolean, TF-IDF, BM25, Probabilistic)
├── web/
│   └── static/index.html     # Web UI for searching
├── documents/                 # Upload your documents here (PDF, TXT, etc.)
├── web_server.py             # Flask API server
├── main.py                   # Demo runner
├── search_cli.py             # Interactive CLI search
└── tests/                    # Pytest test suite
```

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage

### Web Interface (Recommended)
```bash
python web_server.py
# Open http://localhost:5000
```

Features:
- Select retrieval system (Boolean, TF-IDF, BM25, Probabilistic)
- Upload PDF/TXT documents
- Search with Boolean operators

### Command Line
```bash
python main.py      # Demo with sample documents
python search_cli.py  # Interactive search prompt
```

## Tests
```bash
pytest
```
