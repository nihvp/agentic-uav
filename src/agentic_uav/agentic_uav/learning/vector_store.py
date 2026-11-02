"""Persistent vector store for procedures and mission reports.

Embeddings come from sentence-transformers when available, otherwise from a
deterministic hashed bag-of-n-grams (stable across processes, unlike hash()).
Documents persist as JSON lines; embeddings are recomputed on load.
"""

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class HashingEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _bucket(self, token: str) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "little") % self.dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            vec[self._bucket(tok)] += 1.0
            for i in range(len(tok) - 2):
                vec[self._bucket("#" + tok[i:i + 3])] += 0.5
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)

    def encode(self, text: str) -> np.ndarray:
        v = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return v.astype(np.float32)


class VectorStore:
    def __init__(self, path: Optional[str], embedder=None):
        self.path = path
        self.embedder = embedder or HashingEmbedder()
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.vecs: Dict[str, np.ndarray] = {}
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        self._index(d)

    def _index(self, doc: Dict[str, Any]):
        self.docs[doc["id"]] = doc
        self.vecs[doc["id"]] = self.embedder.encode(f"{doc.get('title', '')} {doc['content']}")

    def __len__(self) -> int:
        return len(self.docs)

    def insert(self, doc_id: str, content: str, category: str = "GENERAL", **extra: Any):
        doc = {"id": doc_id, "content": content, "category": category, **extra}
        self._index(doc)
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(doc) + "\n")

    def query(self, text: str, top_k: int = 3, category: str = "") -> List[Tuple[Dict[str, Any], float]]:
        q = self.embedder.encode(text)
        scored = [(self.docs[i], float(np.dot(q, v))) for i, v in self.vecs.items()
                  if not category or self.docs[i].get("category") == category]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max(1, top_k)]
