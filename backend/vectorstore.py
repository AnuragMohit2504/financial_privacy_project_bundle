# backend/vectorstore.py
import os
import faiss
import pickle
import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer

# ----------------------------
# VectorStore for RAG
# ----------------------------
class VectorStore:
    def __init__(self, index_path: str = "vectorstore/index.faiss", meta_path: str = "vectorstore/meta.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # small + fast embedding model
        self.index = None
        self.meta = []
        self.last_retrieved_ids: List[int] = []

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load()
        else:
            print("⚠️ No existing FAISS index found. Run ingestion.py first to build it.")

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.meta = pickle.load(f)
        print(f"✅ Vector store loaded: {len(self.meta)} documents")

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.meta, f)

    def build(self, texts: List[str], metadatas: List[dict] = None):
        """Build FAISS index from list of texts."""
        print("🔨 Building FAISS index...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.meta = metadatas if metadatas else [{} for _ in texts]
        self._save()
        print(f"✅ Index built with {len(texts)} entries")

    def add(self, texts: List[str], metadatas: List[dict] = None):
        """Add new texts to existing index."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.index.add(embeddings)
        if metadatas:
            self.meta.extend(metadatas)
        else:
            self.meta.extend([{} for _ in texts])
        self._save()

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k documents relevant to query."""
        if self.index is None:
            print("⚠️ Vector store is empty. Run ingestion first.")
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)

        self.last_retrieved_ids = indices[0].tolist()
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.meta):
                text = self.meta[idx].get("text", "")
                results.append(text)
        return results
