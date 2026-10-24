# CampusIQ Engineering DevLog

## Evaluation Audit & Diagnosis: Page Hit Rate Analysis

**Date:** 2026-10-19  
**Evaluation Set:** `eval/questions.jsonl` (24 answerable ground-truth questions)  
**Retrieval Index:** ChromaDB (`BAAI/bge-small-en-v1.5`, cosine similarity, 321 chunks)

---

### Executive Summary

In Phase 2 retrieval evaluation, CampusIQ demonstrated strong document-level retrieval (**Document Hit@1 = 79.2%**, **Hit@10 = 100.0%**, **Document MRR = 0.8636**), but exhibited an apparent low exact page hit rate (**Exact Page Hit@1 = 4.2%**, **Page MRR = 0.1458**).

A systematic, page-by-page audit across all 24 evaluation questions revealed that **the primary driver of the low page hit rate is ground-truth label errors in `eval/questions.jsonl`**, rather than semantic retrieval failures:

- **16 of 24 questions (66.7%) suffer from `LABEL_ERROR`**: The ground truth `source_page` points to a blank page, table of contents, or an unrelated section header. In many of these cases (such as SGPA computation `q06`, attendance penalties `q02`, and academic probation `q07`), the dense retriever returned the **exact true page containing the answer at Rank 1 or Rank 2**, but was scored as a miss by the evaluation harness.
- **2 of 24 questions (8.3%) are `MULTI_PAGE_ANSWER`**: The requested topic (e.g., degree requirements, curriculum course lists) spans several semesters across multiple consecutive pages.
- **1 of 24 questions (4.2%) is a `REAL_RETRIEVAL_MISS`**: The embedder ranked generic curriculum pages higher than the specific elective page.
- **5 of 24 questions (20.8%) achieved top-rank matches**: Retrieved at Rank 1 (`q12`) or Rank 2–3 (`q09`, `q10`, `q18`, `q19`, `q20`).

---

### Deep Dive Case Studies

#### Case 1: Question `q06` (SGPA Mathematical Computation)
- **Question:** *"How is the Semester Grade Point Average (SGPA) computed mathematically?"*
- **Labeled Source:** `[rules_ug_2019, p.15]`
- **Audit of Page 15:** Page 15 discusses course credit waivers for exchange students (*"On return, the student may apply to DUGC for waiver of credits..."*). The term "SGPA" does not appear on page 15.
- **Audit of Actual Rule:** The explicit mathematical formula for SGPA is in **Section 5.3.1 on Page 10**.
- **Retriever Performance:** The dense embedder ranked **`[rules_ug_2019, p.10]` at Rank 1** with a similarity score of **0.8312**.
- **Verdict:** `LABEL_ERROR`. The retriever performed correctly; the ground truth label in the eval set was wrong.

#### Case 2: Question `q02` (Attendance Drop Penalty)
- **Question:** *"What consequence does a student face if attendance drops below the 75% threshold without approved leave?"*
- **Labeled Source:** `[rules_ug_2019, p.13]`
- **Audit of Page 13:** Page 13 contains only 110 characters: a section division banner reading *"A. REGULATIONS FOR UNDERGRADUATE PROGRAMMES 13"*.
- **Audit of Actual Rule:** Section 4.1 (*Attendance Requirements*) is on **Page 7**: *"Students not meeting the attendance criterion of 75% will be required to score at least C grade to pass a course. These students would be awarded F grade if their marks are lower than the cut-off for C grade in a course."*
- **Retriever Performance:** The retriever ranked **`[rules_ug_2019, p.7]` at Rank 1** (score: 0.7137).
- **Verdict:** `LABEL_ERROR`. The evaluation harness marked Rank 1 retrieval as a page miss because page 13 was labeled.

#### Case 3: Question `q01` (Minimum Attendance Requirement)
- **Question:** *"What is the minimum attendance requirement for students under IIT Jodhpur Academic Regulations?"*
- **Labeled Source:** `[rules_ug_2019, p.12]`
- **Audit of Page 12:** Page 12 is a completely blank page containing 80 whitespace characters and footer number "12".
- **Audit of Actual Rule:** The attendance rule is on **Page 7**.
- **Verdict:** `LABEL_ERROR`.

#### Case 4: Question `q07` (Academic Probation Criteria)
- **Question:** *"What academic criteria trigger academic probation or warning for students?"*
- **Labeled Source:** `[rules_ug_2019, p.17]`
- **Audit of Page 17:** Page 17 defines Section 11.2 (semester credit registration limits). Table 11.3 (*"Criteria for continuation at the end of the second registered semester"*) is on **Page 18**.
- **Retriever Performance:** Retriever returned **`[rules_ug_2019, p.18]` at Rank 1** (score: 0.6986).
- **Verdict:** `LABEL_ERROR`.

#### Case 5: Question `q15` (Postgraduate Assistantship Continuation)
- **Question:** *"What is the minimum CGPA required to maintain financial assistance or teaching assistantship in postgraduate programs?"*
- **Labeled Source:** `[rules_pg_2022, p.8]`
- **Audit of Page 8:** Page 8 discusses independent study courses and project registration.
- **Audit of Actual Rule:** Section 12.8.4 (*"Minimum performance for continuation Assistantship/Fellowship"*) is on **Page 27**.
- **Retriever Performance:** Retriever returned **`[rules_pg_2022, p.27]` at Rank 2** (score: 0.7618).
- **Verdict:** `LABEL_ERROR`.

---

### Comprehensive 24-Question Diagnostic Table

| QID | Question Summary | Labeled Doc & Page | Doc Rank | Page Rank | True Page in Document | Diagnostic Classification | Root Cause & Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **q01** | Minimum attendance requirement | `rules_ug_2019`, p.12 | 1 | None | 7 | `LABEL_ERROR` | Page 12 is blank (80 chars). Rule is on Page 7. |
| **q02** | Attendance < 75% consequence | `rules_ug_2019`, p.13 | 1 | None | 7 | `LABEL_ERROR` | Page 13 is section title. Rule is on Page 7 (retrieved Rank 1). |
| **q03** | Minimum CGPA for graduation | `rules_ug_2019`, p.16 | 4 | None | 24, 34 | `LABEL_ERROR` | Page 16 is course load. Graduation CGPA is on p.24 & p.34. |
| **q04** | Maximum duration for UG degree | `rules_ug_2019`, p.7 | 1 | None | 17, 25 | `LABEL_ERROR` | Page 7 is attendance/grades. Max duration is on p.17 & p.25. |
| **q05** | Passing letter grades & points | `rules_ug_2019`, p.14 | 1 | None | 8 | `LABEL_ERROR` | Page 14 is internship. Table 5.1 grade points is on Page 8. |
| **q06** | SGPA computation formula | `rules_ug_2019`, p.15 | 1 | None | 10 | `LABEL_ERROR` | Page 15 is exchange waiver. SGPA formula is on Page 10 (retrieved Rank 1). |
| **q07** | Academic probation criteria | `rules_ug_2019`, p.17 | 1 | None | 18 | `LABEL_ERROR` | Page 17 is credit limits. Table 11.3 probation is on Page 18 (retrieved Rank 1). |
| **q08** | Medical leave procedure | `rules_ug_2019`, p.13 | 7 | None | 9, 11 | `LABEL_ERROR` | Page 13 is title page. Medical leave is on Page 9 & 11. |
| **q09** | AI & DS foundational math courses | `curriculum_aids`, p.2 | 1 | 2 | 2 | `RANK_TOP5` | Retrieved at Rank 2. |
| **q10** | AI & DS core ML courses | `curriculum_aids`, p.3 | 1 | 2 | 3 | `RANK_TOP5` | Retrieved at Rank 2. |
| **q11** | AI & DS elective course categories | `curriculum_aids`, p.4 | 1 | None | 4 | `REAL_RETRIEVAL_MISS` | Target page contains electives, but semester core chunks scored higher. |
| **q12** | Systems Science CS curriculum | `curriculum_systems_science`, p.2 | 1 | 1 | 2 | `PERFECT_MATCH` | Retrieved at Rank 1. |
| **q13** | BS Physics curriculum structure | `curriculum_physics`, p.5 | 1 | None | 4, 5, 6 | `MULTI_PAGE_ANSWER` | Physics curriculum courses span multiple semester tables (pp. 4–7). |
| **q14** | M.Tech CPS core courses | `curriculum_mtech_cps`, p.2 | 1 | None | 1 | `LABEL_ERROR` | Labeled p.2 (electives); core courses table is on Page 1. |
| **q15** | PG assistantship CGPA cutoff | `rules_pg_2022`, p.8 | 2 | None | 27 | `LABEL_ERROR` | Labeled p.8 (independent study); Section 12.8.4 assistantship is on p.27 (retrieved Rank 2). |
| **q16** | M.Tech thesis submission procedure | `rules_mtech`, p.6 | 1 | None | 8, 9, 10 | `LABEL_ERROR` | Labeled p.6 (calendar); thesis guidelines are in Section 5 (pp. 8–10). |
| **q17** | M.Sc. degree credit requirement | `rules_msc`, p.4 | 1 | None | 16 | `LABEL_ERROR` | Labeled p.4 (Table of Contents); degree credits are in Section 8 on p.16. |
| **q18** | Autumn 2026 class commencement | `academic_calendar_ay2026_27`, p.1 | 2 | 2 | 1 | `RANK_TOP5` | Retrieved at Rank 2. |
| **q19** | Autumn 2026 exam schedule | `academic_calendar_ay2026_27`, p.2 | 1 | 2 | 2 | `RANK_TOP5` | Retrieved at Rank 2. |
| **q20** | Summer Term 2026 duration | `academic_calendar_summer2026`, p.1 | 3 | 3 | 1 | `RANK_TOP5` | Retrieved at Rank 3. |
| **q21** | Statutory authorities of IITJ | `statutes_code_of_conduct`, p.3 | 1 | None | 1 | `LABEL_ERROR` | Gazette English section starts on p.1 (Gazette p.240). |
| **q22** | Wardens hostel disciplinary powers | `statutes_code_of_conduct`, p.7 | 1 | 6 | 12 | `LABEL_ERROR` | Statute 13 (Wardens) is on Page 12 (retrieved Rank 1 & 2). Labeled p.7 was offset. |
| **q23** | Student indiscipline penalties | `statutes_code_of_conduct`, p.10 | 1 | None | 20 | `LABEL_ERROR` | Penalties schedule is on Page 20 (retrieved Rank 1). |
| **q24** | Senate jurisdiction & constitution | `statutes_code_of_conduct`, p.4 | 1 | None | 2 | `LABEL_ERROR` | Senate authorities statute is on Page 2 (Gazette p.241). |

---

### Breakdown Summary

| Category | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **`LABEL_ERROR`** | **16** | **66.7%** | Ground truth page number in `eval/questions.jsonl` does not match the actual page in the official PDF where the rule/data is stated. Retriever often placed the true page in top ranks. |
| **`RANK_TOP5` / `PERFECT_MATCH`** | **5** | **20.8%** | Retriever correctly returned the labeled page in top-3 ranks (1 exact hit at Rank 1, 4 hits at Rank 2–3). |
| **`MULTI_PAGE_ANSWER`** | **2** | **8.3%** | Information naturally spans several pages (e.g. multi-semester curriculum tables). |
| **`REAL_RETRIEVAL_MISS`** | **1** | **4.2%** | True semantic miss where another section was ranked higher (`q11`). |
| **`BAD_EXTRACTION`** | **0** | **0.0%** | `pypdf` extracted clean text for all text-based documents; extraction errors did not cause retrieval misses. |

---

### Conclusion & Proposed Next Step

The retrieval baseline of CampusIQ is fundamentally sound:
- **Document Hit@10 is 100%**, and **Document Hit@1 is 79.2%**.
- In questions where page retrieval appeared to fail, the retriever was actually retrieving the **correct, true page** from the official IIT Jodhpur documents (e.g., retrieving Page 10 for SGPA, Page 7 for attendance penalties, Page 18 for probation, and Page 20 for penalties).
- Once approved, the ground-truth page numbers in `eval/questions.jsonl` should be updated to their verified true page numbers, aligning the benchmark with reality without touching the underlying queries or answers.

---

## Phase 2 Retrieval Evaluation v2 (Corrected Ground-Truth Labels)

**Date:** 2026-10-21  
**Evaluation Set:** `eval/questions.jsonl` (23 answerable ground-truth questions with verified contiguous substring evidence)  
**Output Artifact:** `eval/results/retrieval_v2_corrected_labels.json` (baseline preserved as `eval/results/retrieval_v1_baseline.json`)  
**Retriever:** Unchanged (`BAAI/bge-small-en-v1.5`, ChromaDB collection `campusiq_docs`, 321 chunks)

### Key Result: Metric Shift Arises From Label Corrections, Not Retrieval Improvement

> [!IMPORTANT]
> **The change in benchmark numbers between v1 and v2 comes entirely from ground-truth label corrections, NOT from any retrieval algorithm improvement or index modification.**
> The underlying dense retriever, chunking strategy, embeddings, and ChromaDB vector store were completely unchanged. In v1, the evaluation harness marked correct retrievals as misses because the ground-truth labels pointed to blank pages, section division banners, or offset indices. Correcting the labels to reflect where the rules and evidence actually reside in the official PDFs aligns the benchmark metrics with the retriever's true performance.

### Comparison: v1 Baseline vs. v2 Corrected Labels

| Metric | v1 Baseline (Flawed Labels, n=24) | v2 Corrected (Any-Page, n=23) | v2 Corrected (First-Listed-Page, n=23) |
| :--- | :---: | :---: | :---: |
| **Doc Hit@1** | 79.2% | **78.3%** | **78.3%** |
| **Doc Hit@3** | 91.7% | **91.3%** | **91.3%** |
| **Doc Hit@5** | 95.8% | **91.3%** | **91.3%** |
| **Doc Hit@10** | 100.0% | **100.0%** | **100.0%** |
| **Page Hit@1** | 4.2% | **47.8%** (+43.6%) | **39.1%** (+34.9%) |
| **Page Hit@3** | 25.0% | **73.9%** (+48.9%) | **65.2%** (+40.2%) |
| **Page Hit@5** | 25.0% | **82.6%** (+57.6%) | **73.9%** (+48.9%) |
| **Page Hit@10** | 29.2% | **91.3%** (+62.1%) | **87.0%** (+57.8%) |
| **Page (±1) Hit@5** | 62.5% | **82.6%** | **82.6%** |
| **Doc MRR** | 0.8636 | **0.8522** | **0.8522** |
| **Page MRR** | 0.1458 | **0.6203** (+0.4745) | **0.5447** (+0.3989) |
| **Avg Latency** | 43.6 ms | **176.5 ms** | **176.5 ms** |

---

## Phase 2 Retrieval Evaluation v3 (Frozen Evaluation Labels)

**Date:** 2026-10-21  
**Evaluation Set:** `eval/questions.jsonl` (34 total questions: 23 answerable, 11 unanswerable; frozen labels with `source_page` removed, `q01`/`q02` historical scope clarified, `q24` verified against Statute 5)  
**Output Artifact:** `eval/results/retrieval_v3_frozen.json` (all prior runs preserved: `retrieval_baseline.json`, `retrieval_v1_baseline.json`, `retrieval_v2_corrected_labels.json`)  
**Retriever:** Unchanged (`BAAI/bge-small-en-v1.5`, ChromaDB collection `campusiq_docs`, 321 chunks)

### Frozen Benchmark Performance
- **Document Hit Rate:** Hit@1 = **78.3%**, Hit@3 = **91.3%**, Hit@5 = **91.3%**, Hit@10 = **100.0%**
- **Any-Page Hit Rate:** Hit@1 = **47.8%**, Hit@3 = **73.9%**, Hit@5 = **82.6%**, Hit@10 = **91.3%**
- **First-Listed-Page Hit Rate:** Hit@1 = **39.1%**, Hit@3 = **65.2%**, Hit@5 = **73.9%**, Hit@10 = **87.0%**
- **Page (±1) Hit Rate:** Hit@1 = **52.2%**, Hit@3 = **73.9%**, Hit@5 = **82.6%**, Hit@10 = **91.3%**
- **Doc MRR / Any-Page MRR / First-Page MRR:** **0.8522 / 0.6203 / 0.5447**
- **Average Retrieval Latency:** **163.6 ms**

> [!NOTE]
> All differences between the v1 baseline (Page Hit@5 = 25.0%) and v3 frozen benchmark (Any-Page Hit@5 = 82.6%) are entirely due to ground-truth label corrections pointing to the true sections and pages where answers reside. Zero retrieval or indexing algorithmic changes were made.

---

## Phase 4: Baseline RAG Evaluation & LLM-as-Judge Benchmark

**Date:** 2026-10-22  
**Generator Model:** `gemini-3.5-flash-lite`  
**Judge Model:** `gemini-3.5-flash-lite`  
**Evaluation Set:** `eval/questions.jsonl` (34 total questions: 23 answerable, 11 unanswerable)  
**Output Artifacts:** `eval/results/rag_baseline_raw.jsonl`, `eval/results/rag_baseline_scored.json`

> [!WARNING]
> **Optimistic Bias Advisory:** The LLM-as-judge belongs to the identical model family (`gemini-3.5-flash-lite`) as the generation pipeline. Consequently, agreement scores and CORRECT verdicts may be subject to same-family stylistic and reasoning alignment bias, making automated evaluation scores optimistic until confirmed by manual human review.

### Summary Results Table

| Metric | Measured Value | Benchmark Interpretation |
| :--- | :---: | :--- |
| **Unanswerable Correct Abstention Rate** | **11 / 11 (100.0%)** | Zero hallucinations; 4 caught at Stage 1 (threshold < 0.65), 7 at Stage 2 (model abstention). |
| **Answerable Wrong Abstention Rate** | **8 / 23 (34.8%)** | Model conservatively abstained on answerable questions (`q01`, `q08`, `q12`, `q17`, `q19`, `q20`, `q22`, `q24`). |
| **Citation Accuracy (>=1 cited doc, page in GT)** | **13 / 15 (86.7%)** | Of the 15 generated answers, 13 cited verified ground-truth pages directly in text. |
| **Invalid Citation Share** | **1 / 22 (4.5%)** | 21 of 22 cited bracket markers accurately mapped to retrieved chunks with verified dates & scopes. |
| **LLM Judge: CORRECT** | **21 / 34 (61.8%)** | 10 answerable answers + 11 correct unanswerable abstentions. |
| **LLM Judge: PARTIAL** | **3 / 34 (8.8%)** | `q09`, `q15`, `q16` captured core aspects but omitted minor details. |
| **LLM Judge: INCORRECT** | **10 / 34 (29.4%)** | 8 wrong abstentions + 2 answered with factual gaps (`q02`, `q21`). |
| **Latency p50 / p95** | **72.5 ms / 1906.7 ms** | P50 benefits from Stage 1 thresholding and disk caching; p95 reflects cold dense search + LLM call. |
| **Tokens per Request (Mean In / Out / Total)** | **2854.9 / 71.0 / 2925.9** | Bounded context prompt with top-5 retrieved chunks. |

### Diagnostic Breakdown: Wrong Abstentions on Answerable Questions (8 / 23)
The 8 answerable questions where the model abstained (`q01`, `q08`, `q12`, `q17`, `q19`, `q20`, `q22`, `q24`) stem from the strict grounding constraint in the system prompt. When top-5 chunks retrieve adjacent sections or partial schedules rather than the complete clause, the model prefers safe abstention over guessing.
- **Classification of 8 Wrong Abstentions:** 3 `RETRIEVAL_MISS` (`q01`, `q08`, `q17`: expected page outside top-5) and 5 `MODEL_OVER_REFUSAL` (`q12`, `q19`, `q20`, `q22`, `q24`: expected page retrieved at rank 1–3, but model refused).
- **Abstention Stage Analysis:** 100% (8/8) occurred at Stage 2 (model generation constraint); 0 were blocked by the Stage 1 similarity threshold.
- **k=8 Depth Scaling Experiment:** Rerunning the 5 `MODEL_OVER_REFUSAL` queries with $k=8$ resulted in 0/5 answered, proving that deeper chunk retrieval alone does not resolve prompt-enforced refusal thresholds.
- **Answerable-Only Ground Truth Performance (n=23):** Correct: 10/23 (43.5%), Partial: 3/23 (13.0%), Incorrect: 2/23 (8.7%), Wrong-Abstained: 8/23 (34.8%).
- **Engineering Target for Phase 5+:** Implement hybrid dense+BM25 search to address retrieval misses and refine prompt instructions for administrative schedules and multi-clause rules.

---

## Phase 5: Authenticated Read-Only MCP Server

**Date:** 2026-10-22  
**Component:** `api/mcp_server.py`  
**Protocol:** Model Context Protocol (MCP) over Starlette / HTTP SSE transport  
**Authentication:** Bearer token authorization via `MCP_AUTH_TOKEN` (environment variable, never logged or committed)

### Architecture & Tools
An isolated, read-only MCP server exposing two core primitives:
1. `search_documents(query: str, n_results: int = 5)`: Dense vector search over the indexed 321 chunk collection in ChromaDB, returning chunk IDs, text, relevance scores, and metadata (`doc_title`, `doc_id`, `page`, `applies_to`, `file_date`).
2. `get_chunk(chunk_id: str)`: Deterministic direct chunk lookup by ID for targeted context retrieval (enables split-page and continuation reading).

### Security & Protocol Guarantees
- Strict authentication header verification (`Authorization: Bearer <token>`). Missing or invalid tokens immediately reject with HTTP 401.
- Read-only operations only: server contains zero mutation or delete endpoints.
- Integration tests in `tests/test_mcp.py` verified tool listing, parameter schemas, authenticated tool execution, and unauthenticated rejections with fake test client harness.

---

## Phase 6: Multi-Agent System with Google ADK (Multi-agent v1)

**Date:** 2026-10-22  
**Framework:** Google Agent Development Kit (`google-adk`)  
**Model Family:** `gemini-3.5-flash-lite` (consistent across Retriever, Answerer, Verifier, Baseline, and Judge)  
**Evaluation Set:** `eval/questions.jsonl` (34 total questions: 23 answerable, 11 unanswerable)  
**Output Artifacts:** `eval/results/multi_agent_v1_raw.jsonl`, `eval/results/multi_agent_v1_scored.json`

### 1. Step-0 Diagnostic Findings (Pre-Build Analysis)
Before designing the agentic pipeline, a read-only audit analyzed the 5 baseline over-refusals (`q12`, `q19`, `q20`, `q22`, `q24`) and 3 retrieval misses (`q01`, `q08`, `q17`):
- **Contained in one chunk (5/8):** `q01` (`rules_ug_2019_p7_c1`), `q08` (`rules_ug_2019_p21_c0`), `q17` (`rules_msc_p11_c0`), `q19` (`academic_calendar_ay2026_27_p2_c0`), `q20` (`academic_calendar_summer2026_p1_c0`).
- **Split across chunks (3/8):** `q12` (Systems Science core courses split across p.2 and p.3), `q22` (Hostel Warden duties split across p.12 and p.13), `q24` (Senate constitution split across p.2 and p.3).
- **Missing from chunk text (0/8):** 0 questions. All expected information exists within the indexed corpus.
- **Top-5 Metadata Re-Check (`q01`, `q08`):** Verified using `doc_id` metadata. In both cases, the target page from the correct document was absent from the top 5 chunks (`q01` true page rank was 11; `q08` retrieved only rules_pg_2022 page 11 rather than rules_ug_2019 pages 9/11/21). Confirmed genuine retrieval misses.

### 2. Multi-Agent Architecture (`agents/`)
- **Explicit Session State (`agents/session.py`):** `SessionState` records `question`, `queries_tried`, `chunks_seen`, `tool_call_count`, `llm_call_count`, `draft_answer`, `final_answer`, `verifier_result`, `abstained`, and `tokens` with no hidden memory.
- **Authenticated MCP Client (`agents/mcp_client.py`):** Communicates with the MCP server using `MCP_AUTH_TOKEN`. Enforces a strict hard cap of 5 tool calls per question. Supports session resetting and shared vector store execution.
- **Retriever Agent (`agents/retriever.py`):** ADK-based agent that inspects chunk relevance scores, reformulates queries if relevance is insufficient (up to 2 retries), and proactively calls `get_chunk` for adjacent split pages when continuation patterns are detected.
- **Answerer Step (`agents/answerer.py`):** Produces draft answers using strictly retrieved text. Mandates that every factual claim includes a verbatim supporting quote under 15 words and bracketed citation `[Title, p.X]`. If evidence is missing, outputs exactly `"I could not find this in the documents."`
- **Verifier Agent (`agents/verifier.py`):** Two-tier verification pipeline:
  - *(a) Deterministic exact substring check:* Verifies each quoted snippet exists verbatim within the cited chunk text.
  - *(b) LLM entailment verification:* Validates claims logically follow from the quotes, preserves `applies_to` and `file_date`, and highlights cohort rule discrepancies. Revises unsupported claims once or safely abstains.
- **API Integration (`api/main.py`):** `POST /ask` exposes a `mode` parameter (`"baseline"` or `"multi_agent"`, default `"baseline"`). Comprehensive test suite (`tests/test_agents.py`, `tests/test_api.py`) runs hermetically with fake LLM and mock MCP fixtures.

### 3. Benchmark Results: Baseline vs. Multi-Agent v1 ADK

| Metric | RAG Baseline (`rag_baseline_scored.json`) | Multi-Agent v1 (`multi_agent_v1_scored.json`) |
| :--- | :---: | :---: |
| **Model** | `gemini-3.5-flash-lite` | `gemini-3.5-flash-lite` |
| **Unanswerable Questions (out of 11)** | | |
| • Correct Abstentions | **11 / 11 (100.0%)** | **11 / 11 (100.0%)** |
| • Wrong Answers | 0 / 11 (0.0%) | 0 / 11 (0.0%) |
| **Answerable Questions (out of 23)** | | |
| • Answered | 15 / 23 (65.22%) | 7 / 23 (30.43%) |
| • Correct | 10 / 23 (43.48%) | 4 / 23 (17.39%) |
| • Partial | 3 / 23 (13.04%) | 0 / 23 (0.0%) |
| • Incorrect | 10 / 23 (43.48%) | 19 / 23 (82.61%) |
| • Wrong Abstentions | **8 / 23 (34.78%)** | **16 / 23 (69.57%)** |
| **Citation Metrics** | | |
| • Citation Accuracy (hit ground-truth page) | 13 / 15 (86.67%) | 0 / 7 (0.0%) |
| • Total Cited Markers | 22 | 2 |
| • Invalid Citations (unretrieved markers) | 1 / 22 (4.55%) | **0 / 2 (0.0%)** |
| **Performance & Compute** | | |
| • LLM Calls per Question | 1.00 | 1.50 |
| • Mean Tool Calls per Question | 0 (direct retrieval) | 1.41 |
| • Latency p50 / p95 | **72.5 ms / 1,906.7 ms** | 16,198.3 ms / 293,038.6 ms |
| • Mean Tokens (Input / Output / Total) | 2,854.9 / 71.0 / 2,925.9 | 1,279.3 / 42.4 / 1,321.7 |

### 4. Empirical Observations & Failure Modes
1. **Unanswerable Robustness:** Abstention on out-of-scope / unanswerable questions did **not** degrade; it remained perfect at **100.0% (11/11)** with zero hallucinations.
2. **Conservative Over-Verification:** Wrong abstentions on answerable questions increased from 8 to 16. The Verifier agent enforces exact verbatim substring matches for evidence quotes. In cases where chunks contain OCR line-breaks, LaTeX notation (e.g. `q06` SGPA formula), or tabular layouts (e.g. `q05` grading scale), the LLM normalizes whitespace, causing the deterministic substring match to fail and triggering conservative abstention.
3. **Zero Invalid Citations:** The multi-agent pipeline eliminated hallucinated or misattributed citation markers (0.0% invalid citations vs. 4.55% in baseline).
4. **Latency Overhead:** Average p50 latency increased from 72.5 ms (baseline caching + single prompt) to 16.2 s due to multi-agent turn pacing (4.5s free-tier rate limits) and sequential retriever/verifier reasoning steps.

## Phase 6.2: Multi-Agent System Optimization (v2)

### 1. Architectural Adjustments and Fixes Implemented
1. **MCP Client Unpacking Bug Resolution:**
   - In v1, `agents/mcp_client.py` only read `content[0]`, discarding chunks 1 through 4 returned by `search_documents`. This starved the Answerer of ~80% of retrieved context (mean input tokens was 1,279 vs 2,855 in baseline). Fixed to iterate through all elements of `content`, restoring full multi-chunk retrieval context (mean input tokens rose to 6,006.2).
2. **Normalized & Guarded Fuzzy Quote Verification (`agents/verifier.py`):**
   - Normalized text (lowercased, line-break hyphenation repaired, newlines/tabs collapsed, LaTeX commands and punctuation stripped).
   - Added two-tier verification: `exact_normalized` substring check, followed by a guarded `fuzzy` check requiring 100% of numbers, percentages, and dates to match and >= 90% non-stopword tokens to match.
   - Preserved `verification_paths` in SessionState: 82 `exact_normalized`, 3 `fuzzy`, 11 `failed`.
3. **Non-Destructive Verifier Pass:**
   - Valid drafts are returned untouched without an LLM revision call, eliminating marker stripping and reducing mean LLM calls to 1.59.
4. **Baseline-First Retrieval with Top-2 Adjacent-Chunk Expansion & Fast-Exit:**
   - First call uses direct search with `k=5`.
   - If top similarity < 0.65, fast-exits immediately with 0 LLM calls.
   - Deterministically expands top-2 ranked chunks via `get_chunk` for next page continuations (capped at 8 chunks, <= 5 tool calls).
   - Only triggers targeted query reformulation (k=8) when similarity >= 0.65 but draft abstains or fails verification.

### 2. Tuning Set Honesty Note
> [!WARNING]
> Multi-Agent v2 was tuned specifically after inspecting the failure traces on this same 34-question evaluation set. Consequently, these evaluation metrics are inherently optimistic due to tuning on the test set and must be validated on held-out questions before drawing generalized conclusions.

### 3. Benchmark Comparison: Baseline vs. Multi-Agent v1 vs. Multi-Agent v2

| Metric | RAG Baseline (`rag_baseline_scored.json`) | Multi-Agent v1 (`multi_agent_v1_scored.json`) | Multi-Agent v2 (`multi_agent_v2_scored.json`) |
| :--- | :---: | :---: | :---: |
| **Model** | `gemini-3.5-flash-lite` | `gemini-3.5-flash-lite` | `gemini-3.5-flash-lite` |
| **Unanswerable Questions (n=11)** | | | |
| • Correct Abstentions | **11 / 11 (100.0%)** | **11 / 11 (100.0%)** | **11 / 11 (100.0%)** |
| • Wrong Answers | 0 / 11 (0.0%) | 0 / 11 (0.0%) | 0 / 11 (0.0%) |
| **Answerable Questions (n=23)** | | | |
| • Answered | 15 / 23 (65.22%) | 7 / 23 (30.43%) | **19 / 23 (82.61%)** |
| • Correct | 10 / 23 (43.48%) | 4 / 23 (17.39%) | **12 / 23 (52.17%)** |
| • Partial | 3 / 23 (13.04%) | 0 / 23 (0.0%) | 4 / 23 (17.39%) |
| • Incorrect | 10 / 23 (43.48%) | 19 / 23 (82.61%) | **7 / 23 (30.43%)** |
| • Wrong Abstentions | 8 / 23 (34.78%) | 16 / 23 (69.57%) | **4 / 23 (17.39%)** |
| **Citation Metrics** | | | |
| • Citation Accuracy (hit ground-truth page) | **13 / 15 (86.67%)** | 0 / 7 (0.0%) | 16 / 19 (84.21%) |
| • Total Cited Markers | 22 | 2 | **32** |
| • Invalid Citations (unretrieved markers) | 1 / 22 (4.55%) | **0 / 2 (0.0%)** | **0 / 32 (0.0%)** |
| **Efficiency & Compute** | | | |
| • Mean LLM Calls / Question | **0.88** | 1.50 | 1.59 |
| • Mean Tool Calls / Question | **1.0** (direct) | 1.41 | 3.12 |
| • Latency Wall Clock p50 / p95 (ms) | **72.5 / 1,906.7** | 16,198.3 / 293,038.6 | 2,893.7 / 13,008.8 |
| • Latency Compute p50 / p95 (ms) | **72.5 / 1,906.7** | 16,198.3 / 293,038.6 | 2,893.0 / 13,006.2 |
| • Mean Tokens (In / Out / Total) | 2,854.9 / 71.0 / 2,925.9 | 1,279.3 / 42.4 / 1,321.7 | 6,006.2 / 114.9 / 6,121.1 |
| **Verification Path Counts** | — | — | exact_normalized: 82, fuzzy: 3, failed: 11 |

### 4. Metrics Where Multi-Agent v2 Trails Baseline
- **Citation Accuracy Rate:** 84.21% (16/19) vs. 86.67% (13/15) in baseline. While v2 generated far more citations overall (32 markers vs 22 in baseline) and answered 4 more questions, 3 answered questions (`q08`, `q16`, `q17`) cited adjacent regulation sections rather than the exact ground truth page.
- **Latency & Resource Cost:** Due to multi-step agent reasoning, adjacent chunk expansion, and rate-limit backoffs, compute latency is higher (p50: 2,893.0 ms vs 72.5 ms in baseline) and token usage is higher (6,006 in-tokens vs 2,855).
- **LLM Calls:** 1.59 vs 0.88 per question (baseline benefited from immediate threshold fast-exit on unanswerables).

### 5. Updated Stage-by-Stage Failure Trace for Remaining 4 Wrong Abstentions
The 4 remaining wrong abstentions in v2 (`q01`, `q19`, `q20`, `q24`) break down as:
1. `q01` (Minimum attendance requirement under 2019 rules):
   - **Stage: `retriever` (Dense Retrieval Miss).** Dense search with question embeddings fetched `rules_ug_2019` p.1 and p.13, missing p.7. Reformulation did not surface p.7.
2. `q19` (Mid-Sem and End-Sem exams for Autumn 2026):
   - **Stage: `answerer` (Model Over-Refusal).** Target pages [2, 6, 8] from `academic_calendar_ay2026_27` were retrieved, but calendar tables with complex multi-column dates led the Answerer to refuse.
3. `q20` (Duration and purpose of Summer Term):
   - **Stage: `answerer` (Model Over-Refusal).** Target page 1 was retrieved in context, but the Answerer refused due to lack of a single verbatim sentence defining "purpose".
4. `q24` (Senate constitution and jurisdiction):
   - **Stage: `answerer` (Model Over-Refusal).** Target pages [2, 3] from `statutes_code_of_conduct` were retrieved in top-5 chunks, but the Answerer refused to synthesize across multi-clause sections.

## Phase 7: Observability & Integrity Audit

### 1. Integrity Audit & Reproducibility (Part A)
- **Elimination of Post-Hoc Patching:**
  In Phase 6.2, an initial post-hoc utility (`scratch/fix_citations.py`) was used to map `chunk_id` for citations where the MCP tool output had omitted the field. During this audit, the resolution logic was moved directly into `MultiAgentPipeline._resolve_citation_chunk_ids` in `agents/coordinator.py` and `agents/session.py`.
  - The patched raw output was preserved in `eval/results/multi_agent_v2_raw_patched_backup.jsonl` (git-ignored).
  - The raw file `eval/results/multi_agent_v2_raw.jsonl` was deleted and regenerated cleanly from the cache with native production resolution.
  - Re-scoring via `eval/score_rag.py --tag v2` reproduced identical metrics: **16/19 (84.21%) citation accuracy**, **0/32 invalid citations**, **23/34 correct**, confirming 100% native reproducibility.
- **Manual Reading & Cohort-Mixing Check (All 19 Answers):**
  - **Off-Topic Answers:** Exactly **0** off-topic answers exist across all 19 answered questions. All generated responses directly address the questions as defined in `eval/questions.jsonl` and `eval/results/multi_agent_v2_raw.jsonl`.
  - **Cohort Mixing:** Several queries where cohort was implicit or unstated (`q05`, `q06`, `q08`, `q15`, `q16`, `q17`) retrieved mixed regulatory documents (e.g. 2022 rules or M.Sc. rules alongside 2019 UG regulations). This highlights document scoping as an area for query router enhancement.
- **Answered-Only Latency Comparison:**
  - RAG Baseline ($n=15$): p50 = 78.0 ms, p95 = 1,906.0 ms
  - Multi-Agent v2 ($n=19$): p50 = 2,371.8 ms (wall-clock) / 2,308.0 ms (compute), p95 = 6,834.8 ms (wall-clock) / 6,780.6 ms (compute).

### 2. OpenTelemetry Structured Tracing Implementation
- **Tracer Module (`observability/tracing.py`):** Configured with `opentelemetry-api` and `opentelemetry-sdk`. Supports hierarchical span export to `eval/results/traces/traces.jsonl` (git-ignored) and console export.
- **Spans & Attributes:**
  - `request` (root): records `question_id`, `mode`, `model`, `num_chunks`, `input_tokens`, `output_tokens`, `cache_hit`, `abstained`, `abstention_stage`, `verification_path`, `tool_calls`.
  - `retrieval`, `adjacent_expansion`, `generation`, `verification`, and `retry`.
- **Strict Privacy Guard:** Automatically sanitizes attributes to reject prompt text, chunk text, API keys, and bearer tokens. Validated with unit tests in `tests/test_tracing.py`.
### 3. Follow-Up Integrity & Telemetry Audit
- **Top-1 Score Propagation:** Direct lookups (`get_chunk`, `get_page_chunks`) returned an artificial score of 1.0, which previously polluted `SessionState.top_score` in `coordinator.py`. Fixed by recording `top_score` explicitly in `SessionState` during `RetrieverAgent.run()` from `search_documents`. All 34 questions maintain identical Stage-1 routing (30 queries >= 0.65, 4 queries < 0.65; 0 result changes).
- **Baseline Latency Real-World Verification:** In `rag_baseline_raw.jsonl`, 30 of 34 total queries (and 12 of 15 answered queries) were recorded with `cached: True` and sub-200ms latency, accounting for the reported 72.5 ms baseline latency. An uncached live comparison ($n=8$) revealed real baseline generation latency averages ~1,064 ms.
- **Trace Test Isolation:** Added session fixture in `tests/conftest.py` and updated `tests/test_tracing.py` so unit tests write only to pytest's `tmp_path`, keeping `eval/results/traces/traces.jsonl` clean for production and benchmark runs.
- **Benchmark Benchmark Label & Scoping Findings:**
  - `q02`: Ground-truth label (*"debarred from appearing in final course examination"*) contradicts the physical PDF of `rules_ug_2019` page 7 (*"Even if the student's attendance falls below 75%, the student will be allowed to appear for the exams"*).
  - `q16` and `q17`: Questions leave the cohort year open, allowing the retriever to match valid but non-target revisions (2014 regulations for M.Tech thesis; 2022 regulations for M.Sc. credits).


