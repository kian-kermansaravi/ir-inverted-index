"""
Advanced Information Retrieval Systems
- Boolean Retrieval (AND, OR, NOT)
- TF-IDF Ranked Retrieval
- BM25 Ranked Retrieval
- Probabilistic Retrieval
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from inverted_index import InvertedIndex
from preprocess import preprocess


@dataclass
class SearchResult:
    """A single search result with score and metadata."""
    doc_id: str
    score: float
    text: str = ""
    matched_terms: List[str] = None
    
    def __post_init__(self):
        if self.matched_terms is None:
            self.matched_terms = []


class RetrievalSystem(ABC):
    """Abstract base class for retrieval systems."""
    
    def __init__(self, index: InvertedIndex, docs: Dict[str, str]):
        self.index = index
        self.docs = docs
        self._doc_lengths: Dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._total_docs: int = len(docs)
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute document statistics for scoring."""
        total_length = 0
        for doc_id, text in self.docs.items():
            tokens = preprocess(text)
            self._doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
        
        if self._total_docs > 0:
            self._avg_doc_length = total_length / self._total_docs
    
    @abstractmethod
    def search(self, query: str) -> List[SearchResult]:
        """Search and return ranked results."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this retrieval system."""
        pass


class BooleanRetrieval(RetrievalSystem):
    """
    Boolean Retrieval System
    Supports: AND, OR, NOT operators
    Example queries:
      - "information retrieval" (implicit AND)
      - "information AND retrieval"
      - "information OR retrieval"
      - "information NOT noise"
      - "information AND retrieval NOT noise"
    """
    
    @property
    def name(self) -> str:
        return "Boolean Retrieval"
    
    def search(self, query: str) -> List[SearchResult]:
        """Parse and execute boolean query."""
        # Parse query into tokens and operators
        parsed = self._parse_query(query)
        
        if not parsed:
            return []
        
        # Evaluate the boolean expression
        result_docs = self._evaluate(parsed)
        
        # Convert to SearchResult objects
        results = []
        for doc_id in result_docs:
            if doc_id in self.docs:
                # Find which terms matched
                query_terms = [t for t in parsed if t.upper() not in ('AND', 'OR', 'NOT')]
                matched = [t for t in query_terms if doc_id in self.index.postings(t)]
                
                results.append(SearchResult(
                    doc_id=doc_id,
                    score=1.0,  # Boolean doesn't rank
                    text=self.docs[doc_id],
                    matched_terms=matched
                ))
        
        return results
    
    def _parse_query(self, query: str) -> List[str]:
        """Parse query into tokens and operators."""
        # Normalize operators
        query = re.sub(r'\bAND\b', 'AND', query, flags=re.IGNORECASE)
        query = re.sub(r'\bOR\b', 'OR', query, flags=re.IGNORECASE)
        query = re.sub(r'\bNOT\b', 'NOT', query, flags=re.IGNORECASE)
        
        # Split by operators while keeping them
        tokens = re.split(r'\s+(AND|OR|NOT)\s+', query)
        
        # Process each token
        result = []
        for token in tokens:
            token = token.strip()
            if token.upper() in ('AND', 'OR', 'NOT'):
                result.append(token.upper())
            elif token:
                # Preprocess non-operator tokens
                processed = preprocess(token)
                result.extend(processed)
        
        return result
    
    def _evaluate(self, tokens: List[str]) -> Set[str]:
        """Evaluate boolean expression."""
        if not tokens:
            return set()
        
        # Get all document IDs
        all_docs = set(self.docs.keys())
        
        # Start with first term
        i = 0
        result = set()
        current_op = 'OR'  # Default: first term is added
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == 'AND':
                current_op = 'AND'
                i += 1
            elif token == 'OR':
                current_op = 'OR'
                i += 1
            elif token == 'NOT':
                # NOT applies to next term
                i += 1
                if i < len(tokens):
                    next_term = tokens[i]
                    if next_term.upper() not in ('AND', 'OR', 'NOT'):
                        term_docs = set(self.index.postings(next_term).keys())
                        not_docs = all_docs - term_docs
                        
                        if current_op == 'AND':
                            result = result & not_docs if result else not_docs
                        else:  # OR
                            result = result | not_docs
                    i += 1
            else:
                # Regular term
                term_docs = set(self.index.postings(token).keys())
                
                if not result:
                    result = term_docs
                elif current_op == 'AND':
                    result = result & term_docs
                else:  # OR
                    result = result | term_docs
                
                current_op = 'AND'  # Default to AND between consecutive terms
                i += 1
        
        return result


class TFIDFRetrieval(RetrievalSystem):
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) Retrieval
    
    Score = Σ tf(t,d) × idf(t)
    where:
      tf(t,d) = 1 + log(freq(t,d)) if freq > 0, else 0
      idf(t) = log(N / df(t))
    """
    
    @property
    def name(self) -> str:
        return "TF-IDF Ranked Retrieval"
    
    def search(self, query: str) -> List[SearchResult]:
        """Search using TF-IDF scoring."""
        query_terms = preprocess(query)
        
        if not query_terms:
            return []
        
        # Calculate scores for each document
        scores: Dict[str, float] = defaultdict(float)
        matched_terms: Dict[str, List[str]] = defaultdict(list)
        
        for term in query_terms:
            postings = self.index.postings(term)
            
            if not postings:
                continue
            
            # Calculate IDF (with smoothing to avoid 0)
            df = len(postings)
            # Add 1 to numerator and denominator for smoothing
            idf = math.log((self._total_docs + 1) / (df + 1)) + 1
            
            for doc_id, tf in postings.items():
                # Calculate TF (log normalization)
                tf_weight = 1 + math.log(tf) if tf > 0 else 0
                
                # TF-IDF score
                scores[doc_id] += tf_weight * idf
                
                if term not in matched_terms[doc_id]:
                    matched_terms[doc_id].append(term)
        
        # Create and sort results
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
    """
    BM25 (Best Matching 25) Retrieval
    
    State-of-the-art probabilistic ranking function.
    
    Score = Σ IDF(t) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |d|/avgdl))
    
    where:
      k1 = 1.5 (term frequency saturation parameter)
      b = 0.75 (length normalization parameter)
    """
    
    def __init__(self, index: InvertedIndex, docs: Dict[str, str], 
                 k1: float = 1.5, b: float = 0.75):
        super().__init__(index, docs)
        self.k1 = k1
        self.b = b
    
    @property
    def name(self) -> str:
        return "BM25 Probabilistic Retrieval"
    
    def _idf(self, df: int) -> float:
        """Calculate IDF using BM25 formula."""
        # BM25 IDF: log((N - df + 0.5) / (df + 0.5))
        return math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query: str) -> List[SearchResult]:
        """Search using BM25 scoring."""
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
                
                # BM25 term score
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_length)
                
                scores[doc_id] += idf * (numerator / denominator) if denominator > 0 else 0
                
                if term not in matched_terms[doc_id]:
                    matched_terms[doc_id].append(term)
        
        # Create and sort results
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
    """
    Binary Independence Model (BIM) - Probabilistic Retrieval
    
    Based on probability ranking principle.
    Assumes terms are independent given relevance.
    
    RSV(d) = Σ log(p(t|R) × (1-p(t|NR))) / (p(t|NR) × (1-p(t|R)))
    
    Simplified without relevance feedback:
    RSV(d) ≈ Σ log((N - df + 0.5) / (df + 0.5))
    """
    
    @property
    def name(self) -> str:
        return "Probabilistic (BIM) Retrieval"
    
    def search(self, query: str) -> List[SearchResult]:
        """Search using probabilistic model."""
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
            
            # RSV weight (log odds ratio)
            # Without relevance feedback, use collection statistics
            rsv_weight = math.log((self._total_docs - df + 0.5) / (df + 0.5))
            
            for doc_id in postings:
                scores[doc_id] += rsv_weight
                
                if term not in matched_terms[doc_id]:
                    matched_terms[doc_id].append(term)
        
        # Create and sort results
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


# Factory function to get retrieval system
def get_retrieval_system(system_type: str, index: InvertedIndex, 
                         docs: Dict[str, str]) -> RetrievalSystem:
    """
    Factory function to create a retrieval system.
    
    Args:
        system_type: One of 'boolean', 'tfidf', 'bm25', 'probabilistic'
        index: The inverted index
        docs: Dictionary of document ID -> text
    
    Returns:
        The appropriate retrieval system
    """
    systems = {
        'boolean': BooleanRetrieval,
        'tfidf': TFIDFRetrieval,
        'bm25': BM25Retrieval,
        'probabilistic': ProbabilisticRetrieval,
    }
    
    system_class = systems.get(system_type.lower())
    if not system_class:
        raise ValueError(f"Unknown system type: {system_type}. "
                        f"Available: {', '.join(systems.keys())}")
    
    return system_class(index, docs)


# List all available systems
AVAILABLE_SYSTEMS = {
    'boolean': 'Boolean Retrieval (AND, OR, NOT operators)',
    'tfidf': 'TF-IDF Ranked Retrieval (Term Frequency - Inverse Document Frequency)',
    'bm25': 'BM25 Probabilistic Retrieval (Best Matching 25 - State of the art)',
    'probabilistic': 'Binary Independence Model (Classic probabilistic IR)',
}
