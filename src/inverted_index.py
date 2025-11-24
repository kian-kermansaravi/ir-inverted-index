# src/inverted_index.py
from typing import Dict, List
from collections import defaultdict
from .btree import BTree
from .tokenizer import tokenize, pre_token_score
import math

class InvertedIndex:
    def __init__(self, btree_degree: int = 3):
        self.dict_tree = BTree(t=btree_degree)
        self.postings: Dict[str, List[dict]] = defaultdict(list)
        self.doc_count = 0
        self.doc_lengths: Dict[str, int] = {}

    def index_document(self, doc_id: str, text: str, metadata: Dict = None):
        metadata = metadata or {}
        tokens = tokenize(text)
        pre_score = pre_token_score({**metadata, "length": len(text)})
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for term, freq in tf.items():
            self.postings[term].append({"doc": doc_id, "tf": freq, "pre_score": pre_score})
            entry = self.dict_tree.search(self.dict_tree.root, term)
            if entry is None:
                self.dict_tree.insert(term, {"df": 1})
            else:
                if isinstance(entry, dict):
                    entry["df"] = entry.get("df", 0) + 1
        self.doc_count += 1
        self.doc_lengths[doc_id] = len(tokens)

    def get_postings(self, term: str):
        return self.postings.get(term, [])

    def compute_tf_idf(self, term: str, doc_id: str):
        postings = self.get_postings(term)
        N = max(1, self.doc_count)
        df = len(postings)
        for p in postings:
            if p["doc"] == doc_id:
                tf = p["tf"]
                idf = math.log((N+1)/(df+1)) + 1
                score = tf * idf * p.get("pre_score", 1.0)
                return score
        return 0.0

    def visualize_dict(self):
        return self.dict_tree.traverse()
