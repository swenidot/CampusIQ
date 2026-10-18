# CampusIQ: Architecture, Design Decisions & Build Log

> This document tracks every component built in CampusIQ, the architectural reasoning behind each design choice, and crisp talking points for engineering interviews.

---

## 1. High-Level Architecture Overview

CampusIQ is an enterprise-grade, grounded academic-help retrieval-augmented generation (RAG) and multi-agent system built for IIT Jodhpur public academic policies.

```
User Query (via REST API /ask)
   │
   ▼
FastAPI Gateway & Observability Layer (OpenTelemetry traces: latency, token metrics)
   │
   ▼
Multi-Agent Orchestrator (Google ADK)
   ├── Agent 1: Retriever / Router Agent
   │     │ (Invokes Tool via Model Context Protocol)
   │     ▼
   │   Authenticated Read-Only MCP Server
   │     │ (Token Authentication + Query Validation)
   │     ▼
   │   ChromaDB Vector Store (BAAI/bge-small-en-v1.5 embeddings)
   │     │ (Returns top-k chunks with page & doc citations)
   │     ▼
   ├── Draft Answer Generation (Gemini 2.5 Flash with strict citation instructions)
   │
   └── Agent 2: Verifier / Reflection Agent
         │ (Performs self-reflection: verifies claims against cited chunks)
         ├── [Supported] ──> Returns Final Answer + Verified Citations [doc_id, page]
         └── [Unsupported / Weak] ──> Query refinement (max 1 retry) OR Abstention
```

---

## 2. Core Design Choices & Trade-offs (Interview Cheat-Sheet)

| Design Choice | Selection | Why this choice? (Interview rationale) | Trade-off / Alternative considered |
|---|---|---|---|
| **Embeddings** | `BAAI/bge-small-en-v1.5` | SOTA performance on MTEB retrieval benchmark in the small model class (384 dimensions), runs completely locally on CPU with ultra-low latency and zero API cost. | OpenAI `text-embedding-3-small` requires paid API keys and network hops; `all-MiniLM-L6-v2` is slightly faster but scores lower on dense semantic retrieval. |
| **Vector DB** | ChromaDB (Local Persistent) | Zero-infrastructure, embedded SQLite + DuckDB/parquet backend. Perfect for 10-100k chunks without spinning up Docker clusters (Milvus/Qdrant). Free, zero billing. | Pinecone requires cloud credentials; Qdrant is heavier to spin up locally for a desktop test. |
| **PDF Parser** | `pypdf` | Fast, lightweight pure-Python parser that extracts page-by-page bounding boxes/text without heavy C dependencies. Perfect for official digital PDFs. | `pdfplumber` or `tesseract` OCR are 10-20x slower and only needed for scanned photocopies. |
| **Agent Framework** | Google Agent Development Kit (ADK) | Native Google Cloud multi-agent framework built for production LLM systems. Supports modular agents, structured tool dispatch, and clean session isolation. | LangChain/LangGraph or CrewAI have more boilerplate and higher abstraction overhead. |
| **Tool Protocol** | Model Context Protocol (MCP) | Anthropic & Linux Foundation standard for agent-to-tool communication. Decouples the vector store from LLM logic and provides an authenticated boundary. | Direct Python function calls offer no boundary enforcement, auth token check, or external tool reuse. |
| **Abstention Strategy** | Dual-tier (Retrieval Distance Cutoff + Prompt Guardrail) | LLMs hallucinate when context is absent. By checking distance scores before generation AND instructing Gemini to output refusal tokens when evidence is insufficient, false confidence is minimized. | Pure prompt guardrails still occasionally hallucinate; pure thresholding may reject questions that require multi-sentence synthesis. |

---

## 3. Build Log by Phase

### Phase 0: Project Setup, Corpus Definition & Evaluation Design
- **Date**: Oct 19, 2026
- **What was built**:
  1. Git repository structure established with clear domain separation (`ingestion`, `retrieval`, `rag`, `mcp_server`, `agents`, `api`, `eval`, `observability`).
  2. `.gitignore` ensuring zero secrets, no `.venv`, and ignoring raw/processed artifacts while tracking public source metadata.
  3. `.env.example` defining configuration contracts (`GEMINI_API_KEY`, `EMBEDDING_MODEL_NAME`, `MCP_SERVER_TOKEN`, etc.).
  4. Curated 10+ public IIT Jodhpur academic documents spanning B.Tech Ordinances, Academic Calendars, Exam Rules, Grade Systems, and Hostel policies in `data/sources.csv`.
  5. Standardized evaluation dataset `eval/questions.jsonl` (30+ ground truth questions including answerable and unanswerable edge cases) created **before** writing retrieval code, adhering to test-driven evaluation.
- **Why this was done**:
  - Setting up the ground-truth evaluation set *before* building RAG prevents test-set bias (you evaluate against actual user queries rather than queries tuned to match chunks).
  - Explicit secret management and architecture blueprints ensure code is immediately shareable on GitHub and reviewable for enterprise engineering roles.

### Phase 1: Robust PDF Ingestion, Text Cleaning & Chunking
- **Date**: Oct 19, 2026
- **What was built**:
  1. `ingestion/parser.py`: Page-by-page extractor using `pypdf` with:
     - Hyphenation repair (`regu-\nlation` -> `regulation`).
     - Header/footer artifact removal (e.g. `Page X of Y`, `- 12 -`).
     - Scanned / low-character page detection (`min_chars_per_page=30`).
  2. `ingestion/chunker.py`: Semantic chunker with:
     - Configurable target chunk size (500 tokens) and sliding-window overlap (50 tokens).
     - Natural split points at paragraph (`\n\n`) and sentence (`(?<=[.!?])\s+`) boundaries rather than crude character slices.
     - 1-based page metadata preservation: `chunk_id = {doc_id}_p{page}_c{idx}`.
  3. `ingestion/ingest.py`: End-to-end command-line runner that processed all 11 verified IIT Jodhpur PDFs, generating 321 chunks across 251 pages into `data/processed/chunks.jsonl`.
  4. `tests/test_phase1_ingestion.py`: Comprehensive test suite verifying hyphenation repair, metadata preservation, zero empty chunks, overlap validity, and JSONL integrity (all 9 tests passing).
- **Why this was done (Interview Talking Points)**:
  - *Page-level traceability*: In forward-deployed AI, grounding is useless if users can't independently verify citations. Preserving exact page numbers on every chunk allows the RAG verifier and human user to jump straight to the source paragraph.
  - *Boundary-aware chunking*: Fixed character slicing (e.g. every 1000 characters) cuts numbers, formulas, and sentences in half, causing semantic distortion in vector embeddings. Respecting paragraph and sentence boundaries keeps rules intact.
  - *Sliding-window overlap*: A 50-token overlap guarantees that if a rule spans across chunk boundaries (e.g. a condition stated in one sentence and the penalty in the next), both chunks retain sufficient semantic context for vector similarity matching.

