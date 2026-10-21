"""
Unit tests for Phase 0: Project Setup & Corpus Schema
Validates:
1. .env.example exists and contains necessary keys.
2. data/sources.csv is valid CSV, non-empty, and contains expected columns.
3. eval/questions.jsonl is valid JSON Lines with valid schema and minimum question counts.
4. Unanswerable questions are properly flagged.
"""

import csv
import json
import os
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_env_example_exists_and_has_keys():
    env_example_path = os.path.join(BASE_DIR, ".env.example")
    assert os.path.exists(env_example_path), ".env.example must exist"
    with open(env_example_path, "r", encoding="utf-8") as f:
        content = f.read()

    required_keys = [
        "GEMINI_API_KEY",
        "EMBEDDING_MODEL_NAME",
        "CHROMA_PERSIST_DIR",
        "MCP_SERVER_TOKEN",
        "API_PORT",
    ]
    for key in required_keys:
        assert key in content, f"Missing key {key} in .env.example"


def test_sources_csv_schema():
    sources_path = os.path.join(BASE_DIR, "data", "sources.csv")
    assert os.path.exists(sources_path), "data/sources.csv must exist"

    with open(sources_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) >= 10, f"Expected at least 10 sources, found {len(rows)}"

    expected_cols = {"doc_id", "file_name", "title", "source_url", "date_accessed"}
    assert expected_cols.issubset(set(reader.fieldnames or [])), "sources.csv missing required columns"

    doc_ids = set()
    for row in rows:
        assert row["doc_id"].strip(), "doc_id cannot be empty"
        assert row["doc_id"] not in doc_ids, f"Duplicate doc_id found: {row['doc_id']}"
        doc_ids.add(row["doc_id"])
        assert row["file_name"].endswith(".pdf"), f"File name must end with .pdf: {row['file_name']}"
        assert row["source_url"].startswith("http"), f"Invalid source_url: {row['source_url']}"


def test_eval_questions_schema():
    questions_path = os.path.join(BASE_DIR, "eval", "questions.jsonl")
    assert os.path.exists(questions_path), "eval/questions.jsonl must exist"

    sources_path = os.path.join(BASE_DIR, "data", "sources.csv")
    with open(sources_path, "r", encoding="utf-8") as f:
        valid_doc_ids = {row["doc_id"] for row in csv.DictReader(f)}

    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            questions.append(item)

            assert "id" in item, f"Missing 'id' on line {line_num}"
            assert "question" in item, f"Missing 'question' on line {line_num}"
            assert "expected_answer" in item, f"Missing 'expected_answer' on line {line_num}"
            assert "answerable" in item, f"Missing 'answerable' on line {line_num}"
            assert isinstance(item["answerable"], bool), f"'answerable' must be boolean on line {line_num}"

            if item["answerable"]:
                assert item["source_doc"] in valid_doc_ids, (
                    f"source_doc '{item['source_doc']}' on line {line_num} not found in data/sources.csv"
                )
                assert item.get("expected_pages") is not None or item.get("source_page") is not None, (
                    f"Answerable question {item['id']} requires expected_pages or source_page"
                )
            else:
                assert item.get("source_doc") is None or item.get("source_doc") == "", (
                    f"Unanswerable question {item['id']} must have null or empty source_doc"
                )

    assert len(questions) >= 30, f"Expected at least 30 eval questions, got {len(questions)}"
    unanswerable = [q for q in questions if not q["answerable"]]
    assert len(unanswerable) >= 8, f"Expected at least 8 unanswerable questions, got {len(unanswerable)}"
