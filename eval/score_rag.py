"""
CampusIQ Phase 4 RAG Evaluation & Scoring Harness
Evaluates raw RAG baseline responses against eval/questions.jsonl:
1. Abstention accuracy (correct abstention on unanswerables, wrong abstention on answerables).
2. Citation accuracy (share of answered questions with >= 1 cited [doc, page] in expected_pages; share of invalid citations).
3. LLM-as-judge correctness (CORRECT / PARTIAL / INCORRECT) with strict rubric, one-line reasons, and caching.
4. Latency p50/p95 and tokens per request.
5. Saves eval/results/rag_baseline_scored.json and produces summary table and needs_human_check list.
"""

import hashlib
import json
import logging
import math
import os
import random
import time
from typing import Any, Dict, List, Optional
from rag.generator import GeminiGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("campusiq.eval.score")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RESULTS_FILE = os.path.join(BASE_DIR, "eval", "results", "rag_baseline_raw.jsonl")
QUESTIONS_FILE = os.path.join(BASE_DIR, "eval", "questions.jsonl")
SCORED_RESULTS_FILE = os.path.join(BASE_DIR, "eval", "results", "rag_baseline_scored.json")
JUDGE_CACHE_FILE = os.path.join(BASE_DIR, "eval", "results", "judge_cache.json")
TARGET_MODEL = "gemini-3.5-flash-lite"

JUDGE_SYSTEM_PROMPT = """You are an impartial academic evaluator assessing an AI assistant's answers against official institutional ground truth for IIT Jodhpur.
Evaluate whether the generated answer accurately reflects the ground truth expected answer.
The judge sees ONLY the question, the expected_answer, and the generated answer.

Strict Rubric:
- CORRECT: The generated answer captures the essential facts, figures, conditions, and meaning of the expected answer without material omissions or factual errors. (If the question is unanswerable and the model abstained with 'I could not find this in the documents.', mark CORRECT).
- PARTIAL: The generated answer is partially correct or contains some relevant facts from the expected answer, but omits a key figure/condition or includes minor inaccuracies.
- INCORRECT: The generated answer contradicts the expected answer, includes major factual hallucinations, or fails to answer the question (e.g. abstained when an answer was expected, or answered when it should have abstained).

Return STRICTLY a JSON object with:
{
  "verdict": "CORRECT" | "PARTIAL" | "INCORRECT",
  "reason": "One concise sentence explaining the verdict under 25 words."
}"""


def percentile(data: List[float], p: float) -> float:
    """Calculates p-th percentile of a list of floats (0 to 100)."""
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = (len(sorted_d) - 1) * (p / 100.0)
    floor_idx = math.floor(idx)
    ceil_idx = math.ceil(idx)
    if floor_idx == ceil_idx:
        return sorted_d[int(idx)]
    d0 = sorted_d[floor_idx] * (ceil_idx - idx)
    d1 = sorted_d[ceil_idx] * (idx - floor_idx)
    return d0 + d1


def load_judge_cache() -> Dict[str, Dict]:
    if os.path.exists(JUDGE_CACHE_FILE):
        try:
            with open(JUDGE_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read judge cache: {e}")
    return {}


def save_judge_cache(cache: Dict[str, Dict]) -> None:
    os.makedirs(os.path.dirname(JUDGE_CACHE_FILE), exist_ok=True)
    try:
        with open(JUDGE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save judge cache: {e}")


def judge_answer(
    generator: GeminiGenerator,
    cache: Dict[str, Dict],
    question: str,
    expected_answer: str,
    generated_answer: str,
) -> Dict[str, str]:
    payload = f"{question}:::{expected_answer}:::{generated_answer}"
    cache_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    if cache_key in cache:
        return cache[cache_key]

    prompt = f"""Question: {question}

Expected Ground Truth Answer:
{expected_answer}

Generated Answer to Evaluate:
{generated_answer}"""

    # Model call with JSON response
    try:
        res = generator.generate(
            prompt=prompt,
            system_instruction=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
        )
        text = res["text"].strip()
        # Clean json code block if present
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        data = json.loads(text)
        verdict = data.get("verdict", "INCORRECT").upper()
        if verdict not in ["CORRECT", "PARTIAL", "INCORRECT"]:
            verdict = "INCORRECT"
        reason = data.get("reason", "").strip()
    except Exception as e:
        logger.warning(f"Judge parsing failed for question '{question[:30]}': {e}. Falling back.")
        verdict = "INCORRECT"
        reason = f"Evaluation error: {str(e)[:50]}"

    out = {"verdict": verdict, "reason": reason}
    cache[cache_key] = out
    save_judge_cache(cache)
    time.sleep(4.5)  # gentle pacing for 15 RPM free tier
    return out


def evaluate():
    logger.info("Starting Phase 4 RAG Evaluation & Scoring...")

    # Load questions
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        q_map = {q["id"]: q for q in (json.loads(line) for line in f if line.strip())}

    # Load raw RAG results
    with open(RAW_RESULTS_FILE, "r", encoding="utf-8") as f:
        raw_items = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(raw_items)} raw results and {len(q_map)} ground truth questions.")

    judge_cache = load_judge_cache()
    judge_generator = GeminiGenerator(model=TARGET_MODEL, use_cache=False)

    scored_items: List[Dict[str, Any]] = []

    # Counters
    unanswerable_total = 0
    unanswerable_correct_abstain = 0
    unanswerable_wrong_answer = 0

    answerable_total = 0
    answerable_answered = 0
    answerable_wrong_abstain = 0
    wrong_abstain_ids = []

    answerable_citation_hits = 0
    total_cited_markers = 0
    total_invalid_citations = 0

    latencies: List[float] = []
    input_tokens_list: List[int] = []
    output_tokens_list: List[int] = []
    total_tokens_list: List[int] = []

    for item in raw_items:
        qid = item["id"]
        gt = q_map.get(qid, {})
        is_answerable = gt.get("answerable", item.get("ground_truth_answerable", False))
        expected_pages = [int(p) for p in gt.get("expected_pages", item.get("expected_pages", []))]
        expected_doc = gt.get("source_doc", item.get("expected_doc", ""))
        expected_answer = gt.get("expected_answer", item.get("expected_answer", ""))
        generated_answer = item.get("generated_answer", "")
        abstained = item.get("abstained", False)

        lat = item.get("latency_ms", 0.0)
        toks = item.get("tokens") or item.get("token_usage", {})
        latencies.append(lat)
        input_tokens_list.append(toks.get("input_tokens", 0))
        output_tokens_list.append(toks.get("output_tokens", 0))
        total_tokens_list.append(toks.get("total_tokens", 0))

        # 1. Abstention check
        if not is_answerable:
            unanswerable_total += 1
            if abstained:
                unanswerable_correct_abstain += 1
            else:
                unanswerable_wrong_answer += 1
        else:
            answerable_total += 1
            if abstained:
                answerable_wrong_abstain += 1
                wrong_abstain_ids.append(qid)
            else:
                answerable_answered += 1

        # 2. Citation evaluation
        citations = item.get("citations", [])
        has_expected_page_hit = False
        invalid_in_item = 0

        for c in citations:
            total_cited_markers += 1
            if c.get("invalid_citation"):
                total_invalid_citations += 1
                invalid_in_item += 1
            else:
                # Check if this valid citation matches expected doc and page
                cit_page = c.get("page")
                cit_chunk = c.get("chunk_id", "")
                doc_matched = True
                if expected_doc:
                    exp_base = expected_doc.replace(".pdf", "").lower()
                    doc_matched = exp_base in cit_chunk.lower() or exp_base in c.get("doc_title", "").lower()

                if cit_page in expected_pages and doc_matched:
                    has_expected_page_hit = True

        if is_answerable and not abstained:
            if has_expected_page_hit:
                answerable_citation_hits += 1

        # 3. LLM-as-judge correctness
        judge_res = judge_answer(
            generator=judge_generator,
            cache=judge_cache,
            question=item["question"],
            expected_answer=expected_answer,
            generated_answer=generated_answer,
        )

        scored_item = {
            "id": qid,
            "question": item["question"],
            "answerable": is_answerable,
            "expected_pages": expected_pages,
            "expected_doc": expected_doc,
            "abstained": abstained,
            "abstention_stage": item.get("abstention_stage"),
            "abstention_reason": item.get("abstention_reason"),
            "model": item.get("model"),
            "top_score": item.get("top_score"),
            "generated_answer": generated_answer,
            "citations": citations,
            "citation_hit": has_expected_page_hit if (is_answerable and not abstained) else None,
            "invalid_citations_count": invalid_in_item,
            "judge_verdict": judge_res["verdict"],
            "judge_reason": judge_res["reason"],
            "tokens": toks,
            "latency_ms": lat,
        }
        scored_items.append(scored_item)

    # Summaries
    verdict_counts = {
        "CORRECT": sum(1 for s in scored_items if s["judge_verdict"] == "CORRECT"),
        "PARTIAL": sum(1 for s in scored_items if s["judge_verdict"] == "PARTIAL"),
        "INCORRECT": sum(1 for s in scored_items if s["judge_verdict"] == "INCORRECT"),
    }

    unans_abstain_pct = (unanswerable_correct_abstain / unanswerable_total * 100) if unanswerable_total else 0.0
    ans_wrong_abstain_pct = (answerable_wrong_abstain / answerable_total * 100) if answerable_total else 0.0

    cit_acc_pct = (answerable_citation_hits / answerable_answered * 100) if answerable_answered else 0.0
    inv_cit_pct = (total_invalid_citations / total_cited_markers * 100) if total_cited_markers else 0.0

    lat_p50 = percentile(latencies, 50)
    lat_p95 = percentile(latencies, 95)

    mean_in_tok = sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0.0
    mean_out_tok = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0.0
    mean_tot_tok = sum(total_tokens_list) / len(total_tokens_list) if total_tokens_list else 0.0

    # needs_human_check: Every PARTIAL or INCORRECT verdict plus 5 random CORRECT ones
    non_correct_ids = [s["id"] for s in scored_items if s["judge_verdict"] in ["PARTIAL", "INCORRECT"]]
    correct_ids = [s["id"] for s in scored_items if s["judge_verdict"] == "CORRECT"]

    rng = random.Random(42)
    sampled_correct = rng.sample(correct_ids, min(5, len(correct_ids)))
    needs_human_check = non_correct_ids + sampled_correct

    summary_data = {
        "evaluation_model": TARGET_MODEL,
        "judge_model": TARGET_MODEL,
        "total_questions": len(scored_items),
        "unanswerable_questions": {
            "total": unanswerable_total,
            "correct_abstentions": unanswerable_correct_abstain,
            "correct_abstention_pct": round(unans_abstain_pct, 2),
            "wrong_answers": unanswerable_wrong_answer,
        },
        "answerable_questions": {
            "total": answerable_total,
            "answered": answerable_answered,
            "wrong_abstentions": answerable_wrong_abstain,
            "wrong_abstention_pct": round(ans_wrong_abstain_pct, 2),
            "wrong_abstention_ids": wrong_abstain_ids,
        },
        "citation_metrics": {
            "answered_questions_evaluated": answerable_answered,
            "citation_accuracy_hits": answerable_citation_hits,
            "citation_accuracy_pct": round(cit_acc_pct, 2),
            "total_cited_markers": total_cited_markers,
            "invalid_citations_count": total_invalid_citations,
            "invalid_citations_share_pct": round(inv_cit_pct, 2),
        },
        "correctness_judge_verdicts": {
            "CORRECT": verdict_counts["CORRECT"],
            "CORRECT_pct": round(verdict_counts["CORRECT"] / len(scored_items) * 100, 2),
            "PARTIAL": verdict_counts["PARTIAL"],
            "PARTIAL_pct": round(verdict_counts["PARTIAL"] / len(scored_items) * 100, 2),
            "INCORRECT": verdict_counts["INCORRECT"],
            "INCORRECT_pct": round(verdict_counts["INCORRECT"] / len(scored_items) * 100, 2),
        },
        "performance_metrics": {
            "latency_p50_ms": round(lat_p50, 2),
            "latency_p95_ms": round(lat_p95, 2),
            "mean_input_tokens": round(mean_in_tok, 1),
            "mean_output_tokens": round(mean_out_tok, 1),
            "mean_total_tokens": round(mean_tot_tok, 1),
        },
        "needs_human_check": needs_human_check,
    }

    output_payload = {
        "summary": summary_data,
        "needs_human_check": needs_human_check,
        "results": scored_items,
    }

    with open(SCORED_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved scored evaluation to {SCORED_RESULTS_FILE}")

    # Print summary table to stdout
    print("\n" + "=" * 78)
    print(f"{'CampusIQ Phase 4 RAG Baseline Scored Evaluation':^78}")
    print("=" * 78)
    print(f"{'Metric':<46} | {'Value':<28}")
    print("-" * 78)
    print(f"{'Evaluation / Judge Model':<46} | {TARGET_MODEL:<28}")
    print(f"{'Total Questions':<46} | {len(scored_items):<28}")
    print(f"{'Unanswerable Correct Abstention Rate':<46} | {unanswerable_correct_abstain}/{unanswerable_total} ({unans_abstain_pct:.1f}%)")
    print(f"{'Answerable Wrong Abstention Rate':<46} | {answerable_wrong_abstain}/{answerable_total} ({ans_wrong_abstain_pct:.1f}%)")
    print(f"{'  Wrong Abstention IDs':<46} | {str(wrong_abstain_ids):<28}")
    print(f"{'Citation Accuracy (>=1 cited doc,page in GT)':<46} | {answerable_citation_hits}/{answerable_answered} ({cit_acc_pct:.1f}%)")
    print(f"{'Invalid Citation Share (unmatched markers)':<46} | {total_invalid_citations}/{total_cited_markers} ({inv_cit_pct:.1f}%)")
    print(f"{'LLM Judge - CORRECT':<46} | {verdict_counts['CORRECT']}/{len(scored_items)} ({verdict_counts['CORRECT']/len(scored_items)*100:.1f}%)")
    print(f"{'LLM Judge - PARTIAL':<46} | {verdict_counts['PARTIAL']}/{len(scored_items)} ({verdict_counts['PARTIAL']/len(scored_items)*100:.1f}%)")
    print(f"{'LLM Judge - INCORRECT':<46} | {verdict_counts['INCORRECT']}/{len(scored_items)} ({verdict_counts['INCORRECT']/len(scored_items)*100:.1f}%)")
    print(f"{'Latency p50 / p95 (ms)':<46} | {lat_p50:.1f} ms / {lat_p95:.1f} ms")
    print(f"{'Tokens per Request (Mean In / Out / Total)':<46} | {mean_in_tok:.1f} / {mean_out_tok:.1f} / {mean_tot_tok:.1f}")
    print(f"{'Needs Human Check Count':<46} | {len(needs_human_check)} questions")
    print("=" * 78 + "\n")

    return output_payload


if __name__ == "__main__":
    evaluate()
