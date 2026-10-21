"""
CampusIQ Retrieval Evaluation Harness
Benchmarks Hit Rate @ k (k=1, 3, 5, 10) and MRR against the eval dataset.
Saves reproducible baseline numbers to eval/results/retrieval_baseline.json.
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List
from retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("campusiq.eval_retrieval")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QUESTIONS = os.path.join(BASE_DIR, "eval", "questions.jsonl")
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "eval", "results")
DEFAULT_OUT_FILE = os.path.join(DEFAULT_RESULTS_DIR, "retrieval_baseline.json")


def evaluate_retrieval(
    questions_file: str = DEFAULT_QUESTIONS,
    persist_dir: str = os.path.join(BASE_DIR, "chroma_db"),
    out_file: str = DEFAULT_OUT_FILE,
    top_k: int = 10,
) -> Dict:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    store = VectorStore(persist_dir=persist_dir)
    if store.count() == 0:
        raise ValueError("ChromaDB is empty. Please run index_corpus.py first.")

    questions: List[Dict] = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    # Evaluate answerable questions with ground truth
    answerable_questions = [q for q in questions if q.get("answerable") is True]
    logger.info(f"Loaded {len(questions)} total questions ({len(answerable_questions)} answerable ground-truth).")

    k_values = [1, 3, 5, 10]
    doc_hits = {k: 0 for k in k_values}
    any_page_hits = {k: 0 for k in k_values}
    first_page_hits = {k: 0 for k in k_values}
    # Within 1 page window (page +- 1) hits
    page_window_hits = {k: 0 for k in k_values}

    doc_reciprocal_ranks: List[float] = []
    any_page_reciprocal_ranks: List[float] = []
    first_page_reciprocal_ranks: List[float] = []
    latencies: List[float] = []

    per_question_results: List[Dict] = []

    for q in answerable_questions:
        q_id = q["id"]
        q_text = q["question"]
        expected_doc = q["source_doc"]
        if "expected_pages" in q and q["expected_pages"]:
            expected_pages = [int(p) for p in q["expected_pages"]]
        elif "source_page" in q:
            expected_pages = [int(q["source_page"])]
        else:
            expected_pages = []

        primary_page = expected_pages[0] if expected_pages else None

        start_time = time.perf_counter()
        retrieved = store.search(q_text, k=top_k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(elapsed_ms)

        first_doc_rank = None
        first_any_page_rank = None
        first_listed_page_rank = None

        for rank, res in enumerate(retrieved, start=1):
            if res.doc_id == expected_doc and first_doc_rank is None:
                first_doc_rank = rank

            # Match exact doc and any of expected_pages
            if res.doc_id == expected_doc and res.page in expected_pages and first_any_page_rank is None:
                first_any_page_rank = rank

            # Match exact doc and primary (first-listed) expected page
            if res.doc_id == expected_doc and primary_page is not None and res.page == primary_page and first_listed_page_rank is None:
                first_listed_page_rank = rank

        # Update hit counts
        for k in k_values:
            top_k_results = retrieved[:k]
            if any(r.doc_id == expected_doc for r in top_k_results):
                doc_hits[k] += 1
            if any(r.doc_id == expected_doc and r.page in expected_pages for r in top_k_results):
                any_page_hits[k] += 1
            if any(r.doc_id == expected_doc and primary_page is not None and r.page == primary_page for r in top_k_results):
                first_page_hits[k] += 1
            if any(r.doc_id == expected_doc and any(abs(r.page - ep) <= 1 for ep in expected_pages) for r in top_k_results):
                page_window_hits[k] += 1

        doc_reciprocal_ranks.append(1.0 / first_doc_rank if first_doc_rank else 0.0)
        any_page_reciprocal_ranks.append(1.0 / first_any_page_rank if first_any_page_rank else 0.0)
        first_page_reciprocal_ranks.append(1.0 / first_listed_page_rank if first_listed_page_rank else 0.0)

        per_question_results.append({
            "id": q_id,
            "question": q_text,
            "expected_doc": expected_doc,
            "expected_pages": expected_pages,
            "primary_page": primary_page,
            "first_doc_rank": first_doc_rank,
            "first_any_page_rank": first_any_page_rank,
            "first_listed_page_rank": first_listed_page_rank,
            "top1_retrieved": f"[{retrieved[0].doc_id}, p.{retrieved[0].page}] (score={retrieved[0].score})" if retrieved else "None",
            "latency_ms": round(elapsed_ms, 2),
        })

    n = len(answerable_questions)
    metrics = {
        "total_answerable_questions": n,
        "average_retrieval_latency_ms": round(sum(latencies) / n, 2),
        "doc_mrr": round(sum(doc_reciprocal_ranks) / n, 4),
        "any_page_mrr": round(sum(any_page_reciprocal_ranks) / n, 4),
        "first_listed_page_mrr": round(sum(first_page_reciprocal_ranks) / n, 4),
        "page_mrr": round(sum(any_page_reciprocal_ranks) / n, 4),
        "doc_hit_rate": {f"hit@{k}": round(doc_hits[k] / n, 4) for k in k_values},
        "any_page_hit_rate": {f"hit@{k}": round(any_page_hits[k] / n, 4) for k in k_values},
        "first_listed_page_hit_rate": {f"hit@{k}": round(first_page_hits[k] / n, 4) for k in k_values},
        "page_hit_rate_exact": {f"hit@{k}": round(any_page_hits[k] / n, 4) for k in k_values},
        "page_hit_rate_window1": {f"hit@{k}": round(page_window_hits[k] / n, 4) for k in k_values},
    }

    report = {
        "benchmark": "CampusIQ Retrieval Evaluation v2 (Corrected Labels)",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "questions": per_question_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'CampusIQ Retrieval Evaluation (Corrected Labels)':^75}")
    print("=" * 75)
    print(f"Total Evaluated Questions : {n}")
    print(f"Avg Retrieval Latency     : {metrics['average_retrieval_latency_ms']:.2f} ms")
    print(f"Doc MRR / Any-Page MRR / First-Page MRR : {metrics['doc_mrr']:.4f} / {metrics['any_page_mrr']:.4f} / {metrics['first_listed_page_mrr']:.4f}")
    print("-" * 75)
    print(f"{'Metric':<28} | {'k=1':>8} | {'k=3':>8} | {'k=5':>8} | {'k=10':>8}")
    print("-" * 75)
    print(f"{'Document Hit Rate':<28} | {metrics['doc_hit_rate']['hit@1']*100:>7.1f}% | {metrics['doc_hit_rate']['hit@3']*100:>7.1f}% | {metrics['doc_hit_rate']['hit@5']*100:>7.1f}% | {metrics['doc_hit_rate']['hit@10']*100:>7.1f}%")
    print(f"{'Any-Page Hit Rate':<28} | {metrics['any_page_hit_rate']['hit@1']*100:>7.1f}% | {metrics['any_page_hit_rate']['hit@3']*100:>7.1f}% | {metrics['any_page_hit_rate']['hit@5']*100:>7.1f}% | {metrics['any_page_hit_rate']['hit@10']*100:>7.1f}%")
    print(f"{'First-Listed-Page Hit Rate':<28} | {metrics['first_listed_page_hit_rate']['hit@1']*100:>7.1f}% | {metrics['first_listed_page_hit_rate']['hit@3']*100:>7.1f}% | {metrics['first_listed_page_hit_rate']['hit@5']*100:>7.1f}% | {metrics['first_listed_page_hit_rate']['hit@10']*100:>7.1f}%")
    print(f"{'Page (±1) Hit Rate':<28} | {metrics['page_hit_rate_window1']['hit@1']*100:>7.1f}% | {metrics['page_hit_rate_window1']['hit@3']*100:>7.1f}% | {metrics['page_hit_rate_window1']['hit@5']*100:>7.1f}% | {metrics['page_hit_rate_window1']['hit@10']*100:>7.1f}%")
    print("-" * 75)
    print(f"Full benchmark results saved to: {out_file}")
    print("=" * 75 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChromaDB Retrieval against Eval Set")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="Path to questions.jsonl")
    parser.add_argument("--persist-dir", default=os.path.join(BASE_DIR, "chroma_db"), help="ChromaDB path")
    parser.add_argument("--out-file", default=DEFAULT_OUT_FILE, help="Path to output baseline json")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results to evaluate (default: 10)")
    args = parser.parse_args()

    evaluate_retrieval(
        questions_file=args.questions,
        persist_dir=args.persist_dir,
        out_file=args.out_file,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
