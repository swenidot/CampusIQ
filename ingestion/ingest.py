"""
CampusIQ End-to-End Ingestion Pipeline
Parses raw PDFs, cleans text, chunks documents, saves chunks to JSONL,
and outputs a summary of chunks per document.
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List

from ingestion.parser import parse_pdf
from ingestion.chunker import DocumentChunker, TextChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("campusiq.ingestion")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCES_CSV = os.path.join(BASE_DIR, "data", "sources.csv")
DEFAULT_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DEFAULT_OUT_FILE = os.path.join(DEFAULT_PROCESSED_DIR, "chunks.jsonl")


def run_ingestion(
    sources_csv: str = DEFAULT_SOURCES_CSV,
    raw_dir: str = DEFAULT_RAW_DIR,
    out_file: str = DEFAULT_OUT_FILE,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    Executes the ingestion pipeline across all documents listed in sources_csv.
    Saves chunks to out_file in JSONL format and returns summary metrics per document.
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if not os.path.exists(sources_csv):
        raise FileNotFoundError(f"sources.csv not found at: {sources_csv}")

    with open(sources_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sources = list(reader)

    logger.info(f"Loaded {len(sources)} document definitions from {sources_csv}")

    chunker = DocumentChunker(target_chunk_tokens=chunk_size, overlap_tokens=chunk_overlap)
    all_chunks: List[TextChunk] = []
    summary_records: List[Dict] = []

    for row in sources:
        doc_id = row["doc_id"]
        file_name = row["file_name"]
        title = row.get("title", file_name)
        pdf_path = os.path.join(raw_dir, file_name)
        if not os.path.exists(pdf_path):
            logger.warning(f"[{doc_id}] PDF file not found at: {pdf_path}. Skipping.")
            continue

        applies_to = row.get("applies_to", "")
        document_date = row.get("document_date", "")

        pages = parse_pdf(
            file_path=pdf_path,
            doc_id=doc_id,
            title=title,
            applies_to=applies_to,
            document_date=document_date,
        )
        scanned_count = sum(1 for p in pages if p.is_scanned)
        doc_chunks = chunker.chunk_document(pages)
        all_chunks.extend(doc_chunks)

        total_tokens = sum(c.token_count for c in doc_chunks)
        avg_tokens = (total_tokens / len(doc_chunks)) if doc_chunks else 0

        summary_records.append({
            "doc_id": doc_id,
            "title": title,
            "file_name": file_name,
            "total_pages": len(pages),
            "scanned_or_empty_pages": scanned_count,
            "total_chunks": len(doc_chunks),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": round(avg_tokens, 1),
        })

    # Write all chunks to JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    # Save summary metadata
    summary_path = os.path.join(os.path.dirname(out_file), "ingestion_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
                "total_documents_processed": len(summary_records),
                "total_chunks_produced": len(all_chunks),
                "documents": summary_records,
            },
            f,
            indent=2,
        )

    # Print summary table to console
    print("\n" + "=" * 90)
    print(f"{'CampusIQ Ingestion Summary':^90}")
    print("=" * 90)
    print(f"{'Doc ID':<22} | {'Pages':>5} | {'Chunks':>6} | {'Avg Tokens':>10} | {'Title'}")
    print("-" * 90)
    for rec in summary_records:
        title_snippet = rec["title"][:42] + "..." if len(rec["title"]) > 42 else rec["title"]
        print(f"{rec['doc_id']:<22} | {rec['total_pages']:>5} | {rec['total_chunks']:>6} | {rec['avg_tokens_per_chunk']:>10.1f} | {title_snippet}")
    print("-" * 90)
    total_pages = sum(r["total_pages"] for r in summary_records)
    print(f"{'TOTAL':<22} | {total_pages:>5} | {len(all_chunks):>6} | {'-':>10} | Saved to: {out_file}")
    print("=" * 90 + "\n")

    return summary_records


def main():
    parser = argparse.ArgumentParser(description="CampusIQ Document Ingestion Pipeline")
    parser.add_argument("--sources", default=DEFAULT_SOURCES_CSV, help="Path to sources.csv")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Path to raw PDFs directory")
    parser.add_argument("--out-file", default=DEFAULT_OUT_FILE, help="Path to output chunks.jsonl")
    parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk token size (default: 500)")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in tokens (default: 50)")

    args = parser.parse_args()
    run_ingestion(
        sources_csv=args.sources,
        raw_dir=args.raw_dir,
        out_file=args.out_file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
