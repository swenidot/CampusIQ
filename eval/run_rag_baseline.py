"""
CampusIQ Baseline RAG Evaluation Runner
Runs all 34 eval questions through the RAG pipeline.
Saves raw outputs to eval/results/rag_baseline_raw.jsonl.
Ensures single consistent model (gemini-3.5-flash-lite),
stores exact model in every row, parsed citations, and full retrieved chunks.
"""

import json
import logging
import os
import time
from typing import Dict, List
from rag.qa import RAGPipeline, DEFAULT_SIMILARITY_THRESHOLD
from rag.generator import GeminiGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("campusiq.rag.runner")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_FILE = os.path.join(BASE_DIR, "eval", "questions.jsonl")
OUT_FILE = os.path.join(BASE_DIR, "eval", "results", "rag_baseline_raw.jsonl")
TARGET_MODEL = "gemini-3.5-flash-lite"


def run_benchmark():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(questions)} evaluation questions from {QUESTIONS_FILE}")

    # Explicitly use single target model
    generator = GeminiGenerator(model=TARGET_MODEL)
    pipeline = RAGPipeline(generator=generator, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, top_k=5)

    completed: Dict[str, Dict] = {}
    if os.path.exists(OUT_FILE):
        try:
            with open(OUT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Only reuse if it was completed with TARGET_MODEL or threshold abstained (model None)
                        # and has the new structured fields
                        row_model = item.get("model")
                        has_retrieved = "retrieved" in item
                        is_valid_row = (row_model == TARGET_MODEL or (row_model is None and item.get("abstained"))) and has_retrieved
                        if is_valid_row:
                            completed[item["id"]] = item
            logger.info(f"Loaded {len(completed)} validated records with model={TARGET_MODEL} from {OUT_FILE}")
        except Exception as e:
            logger.warning(f"Could not read existing {OUT_FILE}: {e}")

    results: List[Dict] = []
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    print("\n" + "=" * 75)
    print(f"{'Running CampusIQ Phase 3/4 RAG Evaluation (n=34)':^75}")
    print(f"{'Target Model: ' + TARGET_MODEL:^75}")
    print("=" * 75)

    for idx, q in enumerate(questions, 1):
        qid = q["id"]
        q_text = q["question"]
        is_answerable = q.get("answerable", False)

        if qid in completed:
            logger.info(f"[{idx}/{len(questions)}] Skipping {qid} (already completed with {TARGET_MODEL})")
            record = completed[qid]
            results.append(record)
            tokens = record.get("tokens") or record.get("token_usage", {})
            total_tokens += tokens.get("total_tokens", 0)
            total_prompt_tokens += tokens.get("input_tokens", 0)
            total_completion_tokens += tokens.get("output_tokens", 0)
            continue

        logger.info(f"[{idx}/{len(questions)}] Processing {qid} (answerable={is_answerable})...")
        res = pipeline.answer(q_text)

        tokens = res.get("tokens") or res.get("token_usage", {})
        total_tokens += tokens.get("total_tokens", 0)
        total_prompt_tokens += tokens.get("input_tokens", 0)
        total_completion_tokens += tokens.get("output_tokens", 0)

        record = {
            "id": qid,
            "question": q_text,
            "ground_truth_answerable": is_answerable,
            "expected_answer": q.get("expected_answer", ""),
            "expected_doc": q.get("source_doc"),
            "expected_pages": q.get("expected_pages", []),
            "generated_answer": res["answer"],
            "abstained": res["abstained"],
            "abstention_stage": res.get("abstention_stage"),
            "abstention_reason": res.get("abstention_reason"),
            "model": res.get("model"),
            "citations": res.get("citations", []),
            "retrieved": res.get("retrieved", []),
            "top_score": res.get("top_score"),
            "tokens": tokens,
            "token_usage": tokens,
            "latency_ms": res.get("latency_ms"),
            "cached": res.get("cached", False),
        }
        results.append(record)

        # Write incrementally to out_file so progress is never lost
        with open(OUT_FILE, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Pacing for free tier rate limits (15 RPM) if an actual LLM call was made and not cached
        if not res.get("cached") and not res.get("abstained") and res.get("generation_latency_ms", 0) > 0:
            time.sleep(5.0)

    # Overwrite complete file cleanly
    with open(OUT_FILE, "w", encoding="utf-8") as f_out:
        for r in results:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Saved all {len(results)} outputs to {OUT_FILE}")

    # Print summary
    abstained_count = sum(1 for r in results if r["abstained"])
    answered_count = len(results) - abstained_count

    print("\n" + "=" * 75)
    print(f"{'Phase 3/4 RAG Execution Summary':^75}")
    print("=" * 75)
    print(f"Total Questions Evaluated : {len(results)}")
    print(f"Generated Answers        : {answered_count}")
    print(f"Abstained Responses       : {abstained_count}")
    print(f"Total Tokens Used         : {total_tokens} (prompt: {total_prompt_tokens}, completion: {total_completion_tokens})")
    print(f"Raw Results Saved To      : {OUT_FILE}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_benchmark()
