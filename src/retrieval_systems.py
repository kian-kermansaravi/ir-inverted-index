from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set
from collections import defaultdict

from inverted_index import InvertedIndex
from preprocess import preprocess


@dataclass
class SearchResult:
    doc_id: str
    score: float
    text: str = ""
    matched_terms: List[str] = None
    
    def __post_init__(self):
        if self.matched_terms is None:
            self.matched_terms = []


class RetrievalSystem(ABC):
    def __init__(self, index: InvertedIndex, docs: Dict[str, str]):
        self.index = index
        self.docs = docs
        self._doc_lengths: Dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._total_docs: int = len(docs)
        self._compute_stats()
    
    def _compute_stats(self):
        total_length = 0
        for doc_id, text in self.docs.items():
            tokens = preprocess(text)
            self._doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
        
        if self._total_docs > 0:
            self._avg_doc_length = total_length / self._total_docs
    
    @abstractmethod
    def search(self, query: str) -> List[SearchResult]:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class BooleanRetrieval(RetrievalSystem):
    
    @property
    def name(self) -> str:
        return "Boolean Retrieval"
    
    def search(self, query: str) -> List[SearchResult]:
        parsed = self._parse_query(query)
        if not parsed:
            return []
        
        result_docs = self._evaluate(parsed)
        
        results = []
        for doc_id in result_docs:
            if doc_id in self.docs:
                query_terms = [t for t in parsed if t.upper() not in ('AND', 'OR', 'NOT')]
                matched = [t for t in query_terms if doc_id in self.index.postings(t)]
                results.append(SearchResult(
                    doc_id=doc_id,
                    score=1.0,
                    text=self.docs[doc_id],
                    matched_terms=matched
                ))
        return results
    
    def _parse_query(self, query: str) -> List[str]:
        query = re.sub(r'\bAND\b', 'AND', query, flags=re.IGNORECASE)
        query = re.sub(r'\bOR\b', 'OR', query, flags=re.IGNORECASE)
        query = re.sub(r'\bNOT\b', 'NOT', query, flags=re.IGNORECASE)
        
        tokens = re.split(r'\s+(AND|OR|NOT)\s+', query)
        
        result = []
        for token in tokens:
            token = token.strip()
            if token.upper() in ('AND', 'OR', 'NOT'):
                result.append(token.upper())
            elif token:
                processed = preprocess(token)
                result.extend(processed)
        return result
    
    def _evaluate(self, tokens: List[str]) -> Set[str]:
        if not tokens:
            return set()
        
        all_docs = set(self.docs.keys())
        i = 0
        result = set()
        current_op = 'OR'
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == 'AND':
                current_op = 'AND'
                i += 1
            elif token == 'OR':
                current_op = 'OR'
                i += 1
            elif token == 'NOT':
                i += 1
                if i < len(tokens):
                    next_term = tokens[i]
                    if next_term.upper() not in ('AND', 'OR', 'NOT'):
                        term_docs = set(self.index.postings(next_term).keys())
                        not_docs = all_docs - term_docs
                        if current_op == 'AND':
                            result = result & not_docs if result else not_docs
                        else:
                            result = result | not_docs
                    i += 1
            else:
                term_docs = set(self.index.postings(token).keys())
                if not result:
                    result = term_docs
                elif current_op == 'AND':
                    result = result & term_docs
                else:
                    result = result | term_docs
                current_op = 'AND'
                i += 1
        
        return result


class TFIDFRetrieval(RetrievalSystem):
    
    @property
    def name(self) -> str:
        return "TF-IDF Ranked Retrieval"
    
    def search(self, query: str) -> List[SearchResult]:
        query_terms = preprocess(query)
        if not query_terms:
            return []
        
        scores: Dict[str, float] = defaultdict(float)
        matched_terms: Dict[str, List[str]] = defaultdict(list)
        
        for term in query_terms:
            postings = self.index.postings(term)
            if not postings:
                continue
            
            df = len(postings)
            idf = math.log((self._total_docs + 1) / (df + 1)) + 1
            
            for doc_id, tf in postings.items():
                tf_weight = 1 + math.log(tf) if tf > 0 else 0
                scores[doc_id] += tf_weight * idf
                if term not in matched_terms[doc_id]:
                    matched_terms[doc_id].append(term)
        
        results = []
        for doc_id, score in scores.items():
            if doc_id in self.docs:
                results.append(SearchResult(
                    doc_id=doc_id,
                    score=round(score, 4),
                    text=self.docs[doc_id],
                    matched_terms=matched_terms[doc_id]
                ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results


class BM25Retrieval(RetrievalSystem):
    
    def __init__(self, index: InvertedIndex, docs: Dict[str, str], 
                 k1: float = 1.5, b: float = 0.75):
        super().__init__(index, docs)
        self.k1 = k1
        self.b = b
    
    @property
    def name(self) -> str:
        return "BM25 Probabilistic Retrieval"
    
    def _idf(self, df: int) -> float:
        return math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query: str) -> List[SearchResult]:
        query_terms = preprocess(query)
        if not query_terms:
            return []
        
        scores: Dict[str, float] = defaultdict(float)
        matched_terms: Dict[str, List[str]] = defaultdict(list)
        
        for term in query_terms:
            postings = self.index.postings(term)
            if not postings:
                continue
            
            df = len(postings)
            idf = self._idf(df)
            
            for doc_id, tf in postings.items():
                doc_len = self._doc_lengths.get(doc_id, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_length)
                scores[doc_id] += idf * (numerator / denominator) if denominator > 0 else 0
                if term not in matched_terms[doc_id]:
                    matched_terms[doc_id].append(term)
        
        results = []
        for doc_id, score in scores.items():
            if doc_id in self.docs:
                results.append(SearchResult(
                    doc_id=doc_id,
                    score=round(score, 4),
                    text=self.docs[doc_id],
                    matched_terms=matched_terms[doc_id]
                ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results


class ProbabilisticRetrieval(RetrievalSystem):
    
    @property
    def name(self) -> str:
        return "Probabilistic (BIM) Retrieval"
    
    def search(self, query: str) -> List[SearchResult]:
        query_terms = preprocess(query)
        if not query_terms:
            return []
        
        scores: Dict[str, float] = defaultdict(float)
        matched_terms: Dict[str, List[str]] = defaultdict(list)
        
        for term in query_terms:
            postings = self.index.postings(term)
            if not postings:
                continue
            
            df = len(postings)
            rsv_weight = math.log((self._total_docs - df + 0.5) / (df + 0.5))
            
            for doc_id in postings:
                scores[doc_id] += rsv_weight
                if term not in matched_terms[doc_id]:
                    matched_terms[doc_id].append(term)
        
        results = []
        for doc_id, score in scores.items():
            if doc_id in self.docs:
                results.append(SearchResult(
                    doc_id=doc_id,
                    score=round(score, 4),
                    text=self.docs[doc_id],
                    matched_terms=matched_terms[doc_id]
                ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results


def get_retrieval_system(system_type: str, index: InvertedIndex, 
                         docs: Dict[str, str]) -> RetrievalSystem:
    systems = {
        'boolean': BooleanRetrieval,
        'tfidf': TFIDFRetrieval,
        'bm25': BM25Retrieval,
        'probabilistic': ProbabilisticRetrieval,
    }
    
    system_class = systems.get(system_type.lower())
    if not system_class:
        raise ValueError(f"Unknown: {system_type}")
    
    return system_class(index, docs)


AVAILABLE_SYSTEMS = {
    'boolean': 'Boolean Retrieval (AND, OR, NOT)',
    'tfidf': 'TF-IDF Ranked Retrieval',
    'bm25': 'BM25 Probabilistic Retrieval',
    'probabilistic': 'Binary Independence Model',
}
