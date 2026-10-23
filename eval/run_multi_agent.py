"""
CampusIQ Multi-Agent Benchmark Evaluation Runner (Phase 6)
Runs all 34 eval questions through the MultiAgentPipeline.
Saves raw outputs to eval/results/multi_agent_raw.jsonl.
Ensures single consistent model (gemini-3.5-flash-lite),
stores exact model in every row, parsed citations, full retrieved chunks,
and reports number of LLM calls per question.
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agents.coordinator import MultiAgentPipeline
from rag.generator import GeminiGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("campusiq.multi_agent.runner")


QUESTIONS_FILE = os.path.join(BASE_DIR, "eval", "questions.jsonl")
DEFAULT_TAG = "v1"
TARGET_MODEL = "gemini-3.5-flash-lite"


def run_benchmark(tag: str = DEFAULT_TAG):
    out_file = os.path.join(BASE_DIR, "eval", "results", f"multi_agent_{tag}_raw.jsonl")
    cache_file = os.path.join(BASE_DIR, "eval", "results", f"multi_agent_{tag}_cache.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(questions)} evaluation questions from {QUESTIONS_FILE} (tag={tag})")

    # Explicitly use single target model across all agents
    generator = GeminiGenerator(model=TARGET_MODEL, use_cache=False)
    pipeline = MultiAgentPipeline(
        generator=generator,
        model=TARGET_MODEL,
        use_cache=True,
        cache_file=cache_file,
        pacing_seconds=4.5,
    )

    completed: Dict[str, Dict] = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        row_model = item.get("model")
                        is_valid_row = (row_model == TARGET_MODEL or (row_model is None and item.get("abstained"))) and "llm_calls" in item
                        if is_valid_row:
                            completed[item["id"]] = item
            logger.info(f"Loaded {len(completed)} validated multi-agent records from {out_file}")
        except Exception as e:
            logger.warning(f"Could not read existing {out_file}: {e}")

    results: List[Dict] = []
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_llm_calls = 0
    total_tool_calls = 0

    print("\n" + "=" * 78)
    print(f"{'Running CampusIQ Phase 6 Multi-Agent Evaluation (n=34)':^78}")
    print(f"{'Target Model: ' + TARGET_MODEL:^78}")
    print("=" * 78)

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
            total_llm_calls += record.get("llm_calls", 1)
            total_tool_calls += record.get("tool_calls", 1)
            continue

        logger.info(f"[{idx}/{len(questions)}] Multi-Agent Processing {qid} (answerable={is_answerable})...")
        res = pipeline.answer(q_text, pacing=True)
        print(f"[{idx}/{len(questions)}] {qid}: ans_len={len(res['answer'])} abstained={res['abstained']} llm_calls={res.get('llm_calls')} tool_calls={res.get('tool_calls')}", flush=True)

        tokens = res.get("tokens") or res.get("token_usage", {})
        total_tokens += tokens.get("total_tokens", 0)
        total_prompt_tokens += tokens.get("input_tokens", 0)
        total_completion_tokens += tokens.get("output_tokens", 0)
        total_llm_calls += res.get("llm_calls", 1)
        total_tool_calls += res.get("tool_calls", 1)

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
            "model": TARGET_MODEL,
            "citations": res.get("citations", []),
            "retrieved": res.get("retrieved", []),
            "top_score": res.get("top_score"),
            "tokens": tokens,
            "token_usage": tokens,
            "latency_ms": res.get("latency_ms"),
            "llm_calls": res.get("llm_calls", 1),
            "tool_calls": res.get("tool_calls", 1),
            "cached": res.get("cached", False),
        }
        results.append(record)

        # Write incrementally to out_file so progress is never lost
        with open(out_file, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Overwrite complete file cleanly
    with open(out_file, "w", encoding="utf-8") as f_out:
        for r in results:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Saved all {len(results)} outputs to {out_file}")

    # Summary metrics
    abstained_count = sum(1 for r in results if r["abstained"])
    answered_count = len(results) - abstained_count
    mean_llm_calls = total_llm_calls / len(results) if results else 0
    mean_tool_calls = total_tool_calls / len(results) if results else 0

    print("\n" + "=" * 78)
    print(f"{'Phase 6 Multi-Agent Execution Summary':^78}")
    print("=" * 78)
    print(f"Total Questions Evaluated : {len(results)}")
    print(f"Generated Answers        : {answered_count}")
    print(f"Abstained Responses       : {abstained_count}")
    print(f"Mean LLM Calls / Question : {mean_llm_calls:.2f}")
    print(f"Mean Tool Calls / Question: {mean_tool_calls:.2f}")
    print(f"Total Tokens Used         : {total_tokens} (prompt: {total_prompt_tokens}, completion: {total_completion_tokens})")
    print(f"Raw Results Saved To      : {out_file}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CampusIQ Multi-Agent Benchmark Evaluation Runner")
    parser.add_argument("--tag", type=str, default="v1", help="Tag for output files (default: v1)")
    args = parser.parse_args()
    run_benchmark(tag=args.tag)
