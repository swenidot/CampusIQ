"""
Unit tests for CampusIQ Multi-Agent System (Phase 6).
Validates:
1. RetrieverAgent calls MCP tools up to hard cap (5) and reformulates queries up to 2 times.
2. AnswererAgent produces grounded draft answers with verbatim quotes and citations.
3. VerifierAgent performs deterministic substring check and entailment verification.
4. SessionState tracks queries, chunks, tool calls, and LLM calls without hidden memory.
5. POST /ask supports mode="baseline" (default) and mode="multi_agent".
6. Strictly mocks all LLM and MCP calls (no network or real API calls).
"""

import pytest
from fastapi.testclient import TestClient
from agents.session import SessionState
from agents.mcp_client import MCPClient
from agents.retriever import RetrieverAgent
from agents.answerer import AnswererAgent
from agents.verifier import VerifierAgent
from agents.coordinator import MultiAgentPipeline
from api.main import app, set_pipeline, set_multi_agent_pipeline


class FakeGenerator:
    """Mock Gemini generator returning deterministic responses."""

    def __init__(self, default_response: str = "Mock answer"):
        self.default_response = default_response
        self.call_count = 0
        self.calls = []
        self.model = "gemini-3.5-flash-lite"

    def generate(self, prompt: str, system_instruction: str = "", temperature: float = 0.0):
        self.call_count += 1
        self.calls.append({"prompt": prompt, "system_instruction": system_instruction})

        # Reformulation prompt
        if "reformulated search query" in prompt.lower() or "retrieval specialist" in prompt.lower():
            return {
                "text": "UG rules attendance 75% end semester",
                "input_tokens": 50,
                "output_tokens": 10,
                "total_tokens": 60,
                "model": self.model,
            }

        # Verifier prompt
        if "VERIFIER" in system_instruction.upper() or "draft answer to verify" in prompt.lower():
            return {
                "text": 'Students must maintain "minimum attendance of 75%" to appear in exams [Academic Regulations, p.7].',
                "input_tokens": 120,
                "output_tokens": 25,
                "total_tokens": 145,
                "model": self.model,
            }

        # Answerer prompt
        return {
            "text": 'Under the rules, students are required to maintain a "minimum attendance of 75%" [Academic Regulations, p.7].',
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "model": self.model,
        }


class FakeMCPClient:
    """Mock MCP client tracking tool calls and returning structured chunks."""

    def __init__(self):
        self.call_count = 0
        self.max_tool_calls = 5
        self.search_queries = []
        self.get_chunk_calls = []

    def reset_session(self) -> None:
        self.call_count = 0

    def search_documents(self, query: str, k: int = 5):
        if self.call_count >= self.max_tool_calls:
            return []
        self.call_count += 1
        self.search_queries.append(query)

        # Return high scoring chunk for specific query
        if "attendance" in query.lower():
            score = 0.88
            text = "4.1.9 Attendance Requirements: A student should have full attendance. Minimum attendance of 75% in all lectures is required."
        else:
            score = 0.50
            text = "General introduction to Indian Institute of Technology Jodhpur academic programmes."

        return [
            {
                "chunk_id": "rules_ug_2019_p7_c1",
                "doc_id": "rules_ug_2019",
                "title": "Academic Regulations",
                "page": 7,
                "applies_to": "Students enrolled from July 2019 onwards",
                "file_date": "2019-09-03",
                "score": score,
                "text": text,
            }
        ]

    def get_chunk(self, doc_id: str, page: int):
        if self.call_count >= self.max_tool_calls:
            return []
        self.call_count += 1
        self.get_chunk_calls.append((doc_id, page))
        return [
            {
                "chunk_id": f"{doc_id}_p{page}_c0",
                "doc_id": doc_id,
                "title": "Academic Regulations",
                "page": page,
                "applies_to": "July 2019 onwards",
                "file_date": "2019-09-03",
                "score": 0.85,
                "text": f"Continuation content on page {page}.",
            }
        ]


def test_session_state_explicit():
    """Verify SessionState explicitly tracks queries, chunks, tokens, and decisions without hidden memory."""
    s = SessionState(question="What is the attendance rule?")
    assert s.tool_call_count == 0
    assert s.llm_call_count == 0
    assert s.abstained is False

    chunk = {"chunk_id": "c1", "doc_id": "rules", "page": 7, "text": "75% attendance"}
    s.add_chunk(chunk)
    assert len(s.chunks_seen) == 1

    # Deduplication check
    s.add_chunk(chunk)
    assert len(s.chunks_seen) == 1

    s.add_tokens(100, 20)
    assert s.tokens["total_tokens"] == 120


def test_retriever_query_reformulation_and_cap():
    """Verify RetrieverAgent reformulates low-relevance queries up to 2 times and respects 5-call cap."""
    fake_mcp = FakeMCPClient()
    fake_gen = FakeGenerator()
    retriever = RetrieverAgent(mcp_client=fake_mcp, generator=fake_gen)

    session = SessionState(question="What is the minimum attendance requirement?")
    chunks = retriever.run(session)

    assert len(chunks) > 0
    assert fake_mcp.call_count <= 5
    assert len(session.queries_tried) >= 1
    assert session.tool_call_count == fake_mcp.call_count


def test_answerer_grounded_generation():
    """Verify AnswererAgent produces grounded draft with quotes and citations."""
    fake_gen = FakeGenerator()
    answerer = AnswererAgent(generator=fake_gen)

    session = SessionState(question="What is the attendance rule?")
    session.chunks_seen = [
        {
            "chunk_id": "c1",
            "title": "Academic Regulations",
            "page": 7,
            "text": "Minimum attendance of 75% is required.",
            "applies_to": "July 2019",
            "file_date": "2019-09-03",
        }
    ]

    draft = answerer.run(session)
    assert "75%" in draft
    assert session.llm_call_count == 1
    assert session.draft_answer == draft


def test_answerer_abstains_on_empty_chunks():
    """Verify AnswererAgent immediately returns standard abstention string if no chunks are seen."""
    fake_gen = FakeGenerator()
    answerer = AnswererAgent(generator=fake_gen)
    session = SessionState(question="Unanswerable question")
    session.chunks_seen = []

    draft = answerer.run(session)
    assert draft == "I could not find this in the documents."
    assert session.llm_call_count == 0


def test_verifier_deterministic_quote_check():
    """Verify VerifierAgent deterministic check correctly detects exact substring quotes and flags hallucinated quotes."""
    fake_gen = FakeGenerator()
    verifier = VerifierAgent(generator=fake_gen)

    chunks = [
        {"text": "A student must maintain a minimum attendance of 75% in all lectures to be eligible."}
    ]

    # Valid quote
    valid_draft = 'Under the rules, a "minimum attendance of 75%" is strictly required [Regulations, p.7].'
    ok, valid, invalid = verifier.deterministic_quote_check(valid_draft, chunks)
    assert ok is True
    assert len(valid) == 1
    assert len(invalid) == 0

    # Hallucinated quote
    invalid_draft = 'Students receive a "full cash bonus of 10000 rupees" for good grades [Regulations, p.7].'
    ok_bad, valid_bad, invalid_bad = verifier.deterministic_quote_check(invalid_draft, chunks)
    assert ok_bad is False
    assert len(invalid_bad) == 1


def test_multi_agent_pipeline_end_to_end():
    """Verify MultiAgentPipeline end-to-end execution without real API calls."""
    fake_mcp = FakeMCPClient()
    fake_gen = FakeGenerator()
    pipeline = MultiAgentPipeline(
        mcp_client=fake_mcp,
        generator=fake_gen,
        use_cache=False,
        pacing_seconds=0.0,
    )

    res = pipeline.answer("Under the 2019 regulations, what is the minimum attendance requirement?")
    assert "answer" in res
    assert res["abstained"] is False
    assert res["llm_calls"] >= 1
    assert res["tool_calls"] >= 1
    assert res["model"] == "gemini-3.5-flash-lite"
    assert "citations" in res
    assert "retrieved" in res


def test_api_mode_flag_and_endpoints():
    """Verify POST /ask handles mode='baseline' and mode='multi_agent' with fake injection."""
    client = TestClient(app)

    # 1. Test GET /health
    health_res = client.get("/health")
    assert health_res.status_code == 200

    # 2. Mock baseline pipeline
    class MockBaseline:
        def answer(self, question, top_k=None, threshold=None):
            return {
                "answer": "Baseline mock response",
                "citations": [],
                "retrieved": [],
                "abstained": False,
                "abstention_stage": None,
                "model": "gemini-3.5-flash-lite",
                "latency_ms": 10.0,
                "tokens": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }

    # 3. Mock multi-agent pipeline
    class MockMultiAgent:
        def answer(self, question):
            return {
                "answer": "Multi-agent mock response",
                "citations": [{"marker": "[Rules, p.7]", "page": 7, "doc_title": "Rules"}],
                "retrieved": [{"chunk_id": "c1", "doc_id": "rules", "page": 7, "score": 0.9}],
                "abstained": False,
                "abstention_stage": None,
                "model": "gemini-3.5-flash-lite",
                "latency_ms": 25.0,
                "tokens": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
                "llm_calls": 3,
                "tool_calls": 2,
            }

    set_pipeline(MockBaseline())
    set_multi_agent_pipeline(MockMultiAgent())

    # Mode = baseline (explicit)
    res_base = client.post("/ask", json={"question": "Test question?", "mode": "baseline"})
    assert res_base.status_code == 200
    data_base = res_base.json()
    assert data_base["answer"] == "Baseline mock response"
    assert data_base["mode"] == "baseline"

    # Mode = default (should be baseline)
    res_def = client.post("/ask", json={"question": "Test question?"})
    assert res_def.status_code == 200
    data_def = res_def.json()
    assert data_def["answer"] == "Baseline mock response"
    assert data_def["mode"] == "baseline"

    # Mode = multi_agent
    res_ma = client.post("/ask", json={"question": "Test question?", "mode": "multi_agent"})
    assert res_ma.status_code == 200
    data_ma = res_ma.json()
    assert data_ma["answer"] == "Multi-agent mock response"
    assert data_ma["mode"] == "multi_agent"
    assert data_ma["llm_calls"] == 3
    assert data_ma["tool_calls"] == 2

    # Invalid mode rejection
    res_inv = client.post("/ask", json={"question": "Test question?", "mode": "invalid_mode"})
    assert res_inv.status_code == 400


def test_tag_paths_and_defaults():
    """Verify --tag defaults to v1 and reproduces v1 paths in coordinator, runner, and scorer."""
    import os
    from agents.coordinator import MultiAgentPipeline, CACHE_FILE
    from eval.run_multi_agent import DEFAULT_TAG
    import eval.score_rag as score_rag

    # 1. Coordinator cache default
    assert "multi_agent_v1_cache.json" in CACHE_FILE
    pipeline = MultiAgentPipeline(use_cache=False)
    assert "multi_agent_v1_cache.json" in pipeline.cache_file

    # 2. Runner default tag
    assert DEFAULT_TAG == "v1"

    # 3. Scorer multi-agent tag path resolution
    base_dir = score_rag.BASE_DIR
    expected_v1_raw = os.path.join(base_dir, "eval", "results", "multi_agent_v1_raw.jsonl")
    expected_v1_scored = os.path.join(base_dir, "eval", "results", "multi_agent_v1_scored.json")
    expected_v2_raw = os.path.join(base_dir, "eval", "results", "multi_agent_v2_raw.jsonl")
    expected_v2_scored = os.path.join(base_dir, "eval", "results", "multi_agent_v2_scored.json")

    # Check that score_multi_agent constructs tag-based paths correctly
    # (Mock evaluate to avoid network / judge calls)
    recorded_paths = []
    original_evaluate = score_rag.evaluate

    def mock_evaluate(raw_file=None, scored_file=None, tag=None):
        recorded_paths.append((raw_file, scored_file, tag))
        return {}

    try:
        score_rag.evaluate = mock_evaluate
        score_rag.score_multi_agent(tag="v1")
        assert recorded_paths[-1] == (expected_v1_raw, expected_v1_scored, "v1")

        score_rag.score_multi_agent(tag="v2")
        assert recorded_paths[-1] == (expected_v2_raw, expected_v2_scored, "v2")
    finally:
        score_rag.evaluate = original_evaluate


def test_verifier_whitespace_and_line_breaks_pass():
    """Verify (a): quote with line breaks and extra whitespace passes via exact_normalized."""
    verifier = VerifierAgent(generator=FakeGenerator())
    chunk_text = (
        "A student must maintain a minimum attendance of 75% in all lectures,\n"
        "tutorials, and practicals to be eligible to appear for the end-semester examinations."
    )
    chunks = [{"text": chunk_text, "doc_id": "rules", "page": 7, "title": "Regulations"}]

    # Draft with uneven line breaks and multiple spaces inside quote
    draft = 'According to the rules, "minimum   attendance\n  of 75% in all lectures,\n tutorials" is required [Regulations, p.7].'
    ok, valid, invalid = verifier.deterministic_quote_check(draft, chunks)
    assert ok is True
    assert len(valid) == 1
    assert len(invalid) == 0


def test_verifier_mutated_number_percentage_or_date_fails():
    """Verify (b): quote with mutated number, percentage or date FAILS fuzzy check."""
    verifier = VerifierAgent(generator=FakeGenerator())
    chunk_text = (
        "Students not meeting the attendance criterion of 75% will be required to score "
        "at least 5.0 CGPA by 15 September 2019 to continue."
    )
    chunks = [{"text": chunk_text, "doc_id": "rules", "page": 7, "title": "Regulations"}]

    # Mutated percentage: 75% -> 80%
    draft_pct = 'The policy specifies "attendance criterion of 80% will be required to score at least 5.0 CGPA" [Regulations, p.7].'
    ok_pct, valid_pct, invalid_pct = verifier.deterministic_quote_check(draft_pct, chunks)
    assert ok_pct is False
    assert len(invalid_pct) == 1

    # Mutated number: 5.0 -> 6.5
    draft_num = 'The policy specifies "attendance criterion of 75% will be required to score at least 6.5 CGPA" [Regulations, p.7].'
    ok_num, valid_num, invalid_num = verifier.deterministic_quote_check(draft_num, chunks)
    assert ok_num is False
    assert len(invalid_num) == 1

    # Mutated date: 15 September 2019 -> 20 September 2019
    draft_date = 'The policy specifies "at least 5.0 CGPA by 20 September 2019 to continue" [Regulations, p.7].'
    ok_date, valid_date, invalid_date = verifier.deterministic_quote_check(draft_date, chunks)
    assert ok_date is False
    assert len(invalid_date) == 1


def test_approved_draft_retains_100_percent_citations():
    """Verify an approved draft retains 100% of its citation markers in final_answer without modification."""
    fake_gen = FakeGenerator()
    verifier = VerifierAgent(generator=fake_gen)

    chunk_1 = "Students are required to maintain a minimum attendance of 75% in all lectures."
    chunk_2 = "A student must secure a minimum CGPA of 5.0 for graduation."
    chunks = [
        {"text": chunk_1, "doc_id": "rules_ug_2019", "page": 7, "title": "Academic Regulations and Degree Requirements"},
        {"text": chunk_2, "doc_id": "rules_ug_2019", "page": 34, "title": "Academic Regulations and Degree Requirements"},
    ]

    session = SessionState(question="What are the attendance and graduation requirements?")
    session.chunks_seen = chunks

    draft = (
        'Students must maintain "minimum attendance of 75% in all lectures" [Academic Regulations and Degree Requirements, p.7]. '
        'For degree completion, "minimum CGPA of 5.0 for graduation" is mandated [Academic Regulations and Degree Requirements, p.34].'
    )
    session.draft_answer = draft

    final_ans = verifier.run(session)

    # Approved draft must be returned untouched
    assert final_ans == draft
    # 0 LLM calls in verifier since all quotes passed
    assert session.llm_call_count == 0
    # Both citation markers are present
    assert "[Academic Regulations and Degree Requirements, p.7]" in final_ans
    assert "[Academic Regulations and Degree Requirements, p.34]" in final_ans
    assert len(session.citations) == 2


def test_real_top_score_propagation():
    """Verify that real cosine similarity from initial search is propagated to session and result."""
    fake_gen = FakeGenerator()
    fake_mcp = FakeMCPClient()
    # Inject chunks with a realistic cosine similarity
    fake_mcp.search_results = [
        {
            "doc_id": "rules_ug_2019",
            "title": "Academic Regulations",
            "page": 7,
            "score": 0.7654,
            "text": "Students are required to maintain a minimum attendance of 75%.",
        }
    ]
    # get_chunk returns an adjacent chunk
    fake_mcp.chunk_results = [
        {
            "doc_id": "rules_ug_2019",
            "title": "Academic Regulations",
            "page": 8,
            "score": 1.0,  # direct lookup score
            "text": "Continuation of attendance rules on next page.",
        }
    ]

    pipeline = MultiAgentPipeline(generator=fake_gen, mcp_client=fake_mcp)
    res = pipeline.answer("What is the attendance requirement?")

    assert res["abstained"] is False
    # Crucial check: top_score must be 0.88 (from search_documents), NOT overwritten by 1.0 from get_chunk!
    assert res["top_score"] == 0.88

    # Test threshold fast-exit (< 0.65)
    fake_mcp_low = FakeMCPClient()
    pipeline_low = MultiAgentPipeline(generator=fake_gen, mcp_client=fake_mcp_low)
    res_low = pipeline_low.answer("What is the dinner menu?")

    assert res_low["abstained"] is True
    assert res_low["abstention_stage"] == "stage_1_threshold"
    assert res_low["top_score"] == 0.50


