
from src.inverted_index import InvertedIndex
from src.pdf_reader import read_pdf_text
from pathlib import Path

def main():
    idx = InvertedIndex()
    base = Path.cwd() / "articles"
    sources = [str(base / f"Article{i}.pdf") for i in (1,2,3)]

    for i, src in enumerate(sources, start=1):
        txt = read_pdf_text(src)
        if not txt:
            print(f"warning: couldn't read {src}")
            continue
        doc_id = f"doc{i}"
        
        idx.index_document(doc_id, txt, {"important": False, "length": len(txt)})

    print("B-tree dictionary (traverse):")
    print(idx.visualize_dict())
    print("postings for 'retrieval':", idx.get_postings("retrieval"))
    print("score doc1,retrieval:", idx.compute_tf_idf("retrieval", "doc1"))

if __name__ == "__main__":
    main()
