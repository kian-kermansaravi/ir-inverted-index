"""Inverted index that keeps its dictionary in a B-tree."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from btree import BTree, BTreeNode
from preprocess import preprocess


@dataclass
class TermStats:
    df: int = 0
    postings: Dict[str, int] = field(default_factory=dict)


class InvertedIndex:
    def __init__(self, min_degree: int = 3) -> None:
        self.dictionary = BTree(min_degree=min_degree)

    def add_document(self, doc_id: str, text: str) -> None:
        tokens = preprocess(text)
        if not tokens:
            return
        frequencies = Counter(tokens)
        for term, count in frequencies.items():
            new_stats = TermStats(df=1, postings={doc_id: count})
            self.dictionary.insert(term, new_stats, merge_fn=self._merge_term_stats)

    @staticmethod
    def _merge_term_stats(existing: TermStats, incoming: TermStats) -> TermStats:
        incoming_doc_id, incoming_tf = next(iter(incoming.postings.items()))
        if incoming_doc_id in existing.postings:
            existing.postings[incoming_doc_id] += incoming_tf
        else:
            existing.postings[incoming_doc_id] = incoming_tf
            existing.df += 1
        return existing

    def postings(self, term: str) -> Dict[str, int]:
        stats = self.dictionary.search(term)
        return dict(stats.postings) if stats else {}

    def iter_terms(self) -> Iterable[Tuple[str, TermStats]]:
        yield from self._traverse(self.dictionary.root)

    def _traverse(self, node: BTreeNode) -> Iterable[Tuple[str, TermStats]]:
        if node.leaf:
            for key, value in zip(node.keys, node.values):
                yield key, value
            return

        for i, key in enumerate(node.keys):
            yield from self._traverse(node.children[i])
            yield key, node.values[i]
        yield from self._traverse(node.children[-1])

    def describe(self) -> str:
        lines: List[str] = ["B-tree dictionary (level order):", self.dictionary.pretty_print(), ""]
        lines.append("Dictionary entries (in-order traversal):")
        for term, stats in self.iter_terms():
            postings_str = ", ".join(f"{doc_id}:{tf}" for doc_id, tf in sorted(stats.postings.items()))
            lines.append(f"- {term} -> df={stats.df} [{postings_str}]")
        return "\n".join(lines)

    def __contains__(self, term: str) -> bool:
        return self.dictionary.search(term) is not None
