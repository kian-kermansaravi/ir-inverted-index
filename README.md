# Inverted Index with B-Tree Dictionary

A small IR project that builds a standard inverted index while keeping the term dictionary in a B-tree. The pipeline includes text cleanup (pre-token steps), tokenization, indexing, and the ability to print the B-tree layout.

## Features
- Text normalization (lowercasing, punctuation strip, whitespace squeeze) before tokenization.
- Simple whitespace tokenizer with empty-token filtering.
- In-memory B-tree dictionary that stores each term with its postings list and term frequency per document.
- Tree visualization helper to print internal nodes and leaves in level order.
- Query helper to fetch postings for a term.
- Demo script that indexes a small document set and shows the tree.
- Interactive search CLI that returns documents for entered terms.
- Duplicate terms across documents merge into a single dictionary entry with combined postings.

## Project Structure
- `src/btree.py` - Minimal B-tree implementation for term dictionary storage.
- `src/preprocess.py` - Text normalization and tokenization helpers.
- `src/inverted_index.py` - Inverted index that uses the B-tree dictionary.
- `main.py` - Demo runner: builds the index, prints the tree, and runs a few sample lookups.
- `example.py` - Additional example that builds an index from a small set of documents and shows lookups.
- `search_cli.py` - Simple interactive search prompt; enter terms to see matching documents.
- `web_server.py` + `web/static/index.html` - Minimal Flask API and static frontend for searching.
- `tests/` - Pytest suite covering preprocessing and inverted index behavior.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage
Run the main demo:
```bash
python main.py
```
Run the additional example:
```bash
python example.py
```
Run the interactive search prompt:
```bash
python search_cli.py
```
Run the web app:
```bash
python web_server.py
# then open http://localhost:5000
```

## Tests
```bash
pytest
```
