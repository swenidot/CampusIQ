"""
CampusIQ Embedding Engine
Wraps sentence-transformers (BAAI/bge-small-en-v1.5) with local caching,
batch inference, and asymmetric query instruction support.
"""

import logging
import os
from typing import List, Optional

# Ensure cache is on D: drive to protect C: drive space
os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "D:/hf_cache")

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("campusiq.embedder")

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingEngine:
    _instance: Optional["EmbeddingEngine"] = None

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=device)
        # Support modern and legacy sentence-transformers attribute
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully. Dimension: {self.embedding_dim}")

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL_NAME) -> "EmbeddingEngine":
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def embed_passages(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embeds a list of passage texts for storage.
        Embeddings are normalized (L2) for cosine similarity.
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str, add_instruction: bool = True) -> List[float]:
        """
        Embeds a search query.
        Applies BGE query instruction prefix for asymmetric retrieval.
        """
        formatted = f"{QUERY_INSTRUCTION}{query}" if add_instruction and "bge" in self.model_name.lower() else query
        embedding = self.model.encode(
            formatted,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()
