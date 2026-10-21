"""
CampusIQ Persistent Vector Store
Manages persistent ChromaDB vector storage with cosine similarity indexing,
metadata filtering, batch upserts, and structured search retrieval.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from retrieval.embedder import EmbeddingEngine

logger = logging.getLogger("campusiq.vector_store")

DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db"
)
DEFAULT_COLLECTION_NAME = "campusiq_docs"


class SearchResult:
    def __init__(
        self,
        chunk_id: str,
        content: str,
        score: float,
        distance: float,
        doc_id: str,
        title: str,
        page: int,
        metadata: Dict[str, Any],
        applies_to: str = "",
        document_date: str = "",
        file_date: str = "",
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.score = round(score, 4)
        self.distance = round(distance, 4)
        self.doc_id = doc_id
        self.title = title
        self.page = page
        self.metadata = metadata
        self.applies_to = applies_to or metadata.get("applies_to", "")
        self.document_date = document_date or file_date or metadata.get("file_date", metadata.get("document_date", ""))
        self.file_date = self.document_date

    def to_citation(self) -> str:
        """
        Formats a human-readable, grounded citation including page and version metadata.
        """
        parts = [f"[{self.title or self.doc_id}, p.{self.page}]"]
        date_str = self.file_date or self.document_date
        if date_str:
            parts.append(f"Date: {date_str}")
        if self.applies_to:
            parts.append(f"Applies to: {self.applies_to}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "distance": self.distance,
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page,
            "applies_to": self.applies_to,
            "document_date": self.document_date,
            "citation": self.to_citation(),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"<SearchResult [{self.doc_id}, p.{self.page}] applies_to='{self.applies_to}' score={self.score}>"


class VectorStore:
    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedder: Optional[EmbeddingEngine] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = embedder or EmbeddingEngine.get_instance()

        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Initialized ChromaDB at '{self.persist_dir}', collection '{self.collection_name}' (existing items: {self.collection.count()})"
        )

    def count(self) -> int:
        return self.collection.count()

    def index_chunks(self, chunks_file: str, batch_size: int = 64) -> int:
        """
        Reads chunks from JSONL, computes embeddings in batches, and upserts into ChromaDB.
        Returns the total number of indexed chunks.
        """
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Chunks file not found at: {chunks_file}")

        chunks: List[Dict] = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        total_chunks = len(chunks)
        logger.info(f"Indexing {total_chunks} chunks into collection '{self.collection_name}'")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c["chunk_id"] for c in batch]
            documents = [c["content"] for c in batch]
            metadatas = [
                {
                    "doc_id": str(c.get("doc_id", "")),
                    "title": str(c.get("title", "")),
                    "page": int(c.get("page", 0)),
                    "chunk_index": int(c.get("chunk_index", 0)),
                    "token_count": int(c.get("token_count", 0)),
                    "applies_to": str(c.get("applies_to", "")),
                    "document_date": str(c.get("document_date", "")),
                }
                for c in batch
            ]

            embeddings = self.embedder.embed_passages(documents, batch_size=batch_size)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"Indexed batch {i + len(batch)}/{total_chunks} chunks.")

        final_count = self.collection.count()
        logger.info(f"Indexing complete. Collection total chunks: {final_count}")
        return final_count

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Searches the vector store using dense cosine similarity.
        query: Search string.
        k: Number of nearest neighbors to retrieve.
        filters: Optional metadata filter dict (e.g., {"doc_id": "rules_ug_2019"}).
        Returns a sorted list of SearchResult objects with score in [0.0, 1.0].
        """
        if not query.strip() or self.collection.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_kwargs["where"] = filters

        results = self.collection.query(**query_kwargs)

        search_results: List[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance d is in [0, 2].
            # Cosine similarity s = 1.0 - d
            sim_score = max(0.0, min(1.0, 1.0 - dist))

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=doc,
                    score=sim_score,
                    distance=dist,
                    doc_id=meta.get("doc_id", ""),
                    title=meta.get("title", ""),
                    page=int(meta.get("page", 0)),
                    metadata=meta,
                    applies_to=meta.get("applies_to", ""),
                    document_date=meta.get("document_date", ""),
                )
            )

        return search_results

    def get_chunk(self, chunk_id: str) -> Optional[SearchResult]:
        """
        Retrieves a single chunk by chunk_id.
        """
        res = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )
        if not res or not res["ids"]:
            return None

        meta = res["metadatas"][0] if res["metadatas"] else {}
        doc = res["documents"][0] if res["documents"] else ""

        return SearchResult(
            chunk_id=chunk_id,
            content=doc,
            score=1.0,
            distance=0.0,
            doc_id=meta.get("doc_id", ""),
            title=meta.get("title", ""),
            page=int(meta.get("page", 0)),
            metadata=meta,
            applies_to=meta.get("applies_to", ""),
            document_date=meta.get("document_date", ""),
        )
