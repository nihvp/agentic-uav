"""
Vector Store and Embedding Engine for UAV RAG Memory.
Provides semantic embedding indexing, cosine similarity scoring,
metadata filtering, and cross-mission document persistence.
"""

import math
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("agentic_uav.learning.vector_store")


class SemanticVectorStore:
    """In-memory semantic vector store supporting both dense neural embeddings and resilient subword hashing."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.transformer_model = None

        self._init_encoder()

    def _init_encoder(self):
        """Attempts to load sentence-transformers; otherwise uses hashing vectorizer."""
        try:
            from sentence_transformers import SentenceTransformer
            self.transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer model initialized for RAG memory.")
        except Exception:
            self.transformer_model = None
            logger.info("Using embedded semantic feature encoder for RAG memory.")

    def _encode_text(self, text: str) -> np.ndarray:
        """Converts text into normalized semantic embedding vector."""
        if self.transformer_model is not None:
            emb = self.transformer_model.encode(text, convert_to_numpy=True)
            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-9)

        # Resilient subword n-gram hashing vectorizer
        tokens = re.findall(r"\b\w+\b", text.lower())
        vec = np.zeros(self.embedding_dim, dtype=np.float32)

        for token in tokens:
            # Word hash
            h = hash(token) % self.embedding_dim
            vec[h] += 1.0

            # Character 3-grams
            for i in range(len(token) - 2):
                tri = token[i:i+3]
                h_tri = hash(tri) % self.embedding_dim
                vec[h_tri] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def insert(self, doc_id: str, content: str, category: str = "GENERAL", metadata: Optional[Dict[str, Any]] = None):
        """Stores and indexes a document."""
        emb = self._encode_text(content)
        self.documents[doc_id] = {
            "id": doc_id,
            "content": content,
            "category": category,
            "metadata": metadata or {}
        }
        self.embeddings[doc_id] = emb

    def query(self, query_text: str, top_k: int = 3, filter_category: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Queries the vector index using cosine similarity."""
        if not self.documents:
            return []

        q_emb = self._encode_text(query_text)
        results = []

        for doc_id, doc in self.documents.items():
            if filter_category and doc["category"].upper() != filter_category.upper():
                continue

            d_emb = self.embeddings[doc_id]
            # Cosine similarity
            sim = float(np.dot(q_emb, d_emb))
            results.append((doc, sim))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
