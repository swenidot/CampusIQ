"""
CampusIQ Vector Indexing Script
Indexes chunks from data/processed/chunks.jsonl into persistent ChromaDB.
"""

import argparse
import logging
import os
import sys

from retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("campusiq.indexer")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHUNKS = os.path.join(BASE_DIR, "data", "processed", "chunks.jsonl")
DEFAULT_CHROMA = os.path.join(BASE_DIR, "chroma_db")


def main():
    parser = argparse.ArgumentParser(description="Index chunks into ChromaDB")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS, help="Path to chunks.jsonl")
    parser.add_argument("--persist-dir", default=DEFAULT_CHROMA, help="ChromaDB persistence directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    store = VectorStore(persist_dir=args.persist_dir)
    total_indexed = store.index_chunks(chunks_file=args.chunks, batch_size=args.batch_size)
    print(f"\n[OK] Successfully indexed {total_indexed} chunks into ChromaDB at '{args.persist_dir}'.")


if __name__ == "__main__":
    main()
