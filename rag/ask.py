"""
CampusIQ Interactive QA CLI
Usage:
    python -m rag.ask "What is the minimum attendance requirement?"
    python -m rag.ask "What is the minimum attendance requirement?" --threshold 0.65 --top-k 5
"""

import argparse
import json
import sys
from rag.qa import RAGPipeline, DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_TOP_K


def main():
    parser = argparse.ArgumentParser(
        description="CampusIQ Academic RAG Assistant - Grounded Question Answering"
    )
    parser.add_argument("question", type=str, help="Question to ask CampusIQ")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Minimum similarity score threshold for retrieval (default: {DEFAULT_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass local generation cache",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw structured JSON response",
    )

    args = parser.parse_args()

    pipeline = RAGPipeline(
        similarity_threshold=args.threshold,
        top_k=args.top_k,
    )
    if args.no_cache:
        pipeline.generator.use_cache = False

    result = pipeline.answer(args.question)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("\n" + "=" * 70)
    print(f"Question : {result['question']}")
    print("=" * 70)
    print(f"\nAnswer:\n{result['answer']}\n")

    if result["abstained"]:
        print(f"[STATUS: ABSTAINED ({result['abstention_reason']})]")
    else:
        print("Citations:")
        for idx, cite in enumerate(result["citations"], 1):
            print(f"  [{idx}] {cite}")

    print("\nMetadata:")
    print(f"  Top Retrieval Score : {result['top_score']:.4f} (Threshold: {args.threshold})")
    print(f"  Tokens              : {result['token_usage']['total_tokens']} (prompt: {result['token_usage']['input_tokens']}, completion: {result['token_usage']['output_tokens']})")
    print(f"  Total Latency       : {result['latency_ms']:.1f} ms (retrieval: {result['retrieval_latency_ms']:.1f} ms, generation: {result['generation_latency_ms']:.1f} ms)")
    if result.get("cached"):
        print("  Cache               : HIT (served from local cache)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
