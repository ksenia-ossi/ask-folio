"""Dense, BM25, and reciprocal-rank hybrid retrieval."""

import re
import unicodedata

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Tokenize Unicode words after stripping accents."""
    normalized = unicodedata.normalize("NFKD", text)
    plain = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"\w+", plain.lower())


class Retriever:
    """Search chunks with dense, keyword, or hybrid ranking."""

    def __init__(self, chunks, matrix, embedder, mode: str = "hybrid"):
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"unknown retriever mode: {mode}")
        self.chunks = chunks
        self.matrix = np.asarray(matrix, dtype=np.float32) if matrix is not None else None
        self.embedder = embedder
        self.mode = mode
        self.bm25 = (
            BM25Okapi([tokenize(chunk.page_content) for chunk in chunks]) if mode != "dense" and chunks else None
        )

    def _dense(self, query: str, k: int, allowed: list[int]) -> list[Document]:
        if not allowed:
            return []
        query_vector = self.embedder.embed_query(query)
        scores = self.matrix[allowed] @ query_vector
        best = np.argsort(-scores, kind="stable")[:k]
        return [self.chunks[allowed[index]] for index in best]

    def _bm25(self, query: str, k: int, allowed: list[int]) -> list[Document]:
        if not allowed or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        best = sorted(allowed, key=lambda index: (-scores[index], index))[:k]
        return [self.chunks[index] for index in best]

    def search(self, query: str, k: int = 5, sources: list[str] | None = None) -> list[Document]:
        """Return top matches, optionally restricted to named sources."""
        allowed = [
            index
            for index, chunk in enumerate(self.chunks)
            if sources is None or chunk.metadata.get("source") in sources
        ]
        if self.mode == "dense":
            return self._dense(query, k, allowed)
        if self.mode == "bm25":
            return self._bm25(query, k, allowed)
        dense = self._dense(query, k * 2, allowed)
        bm25 = self._bm25(query, k * 2, allowed)
        return reciprocal_rank_fusion([dense, bm25], k)


def reciprocal_rank_fusion(rankings: list[list[Document]], k: int, c: int = 60) -> list[Document]:
    """Merge ranked lists with reciprocal-rank scores."""
    scores, docs = {}, {}
    for ranking in rankings:
        for rank, document in enumerate(ranking):
            chunk_id = document.metadata["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (c + rank + 1)
            docs[chunk_id] = document
    return [docs[chunk_id] for chunk_id in sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]]
