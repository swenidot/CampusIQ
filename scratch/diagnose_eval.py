import sys
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.stdout.reconfigure(encoding='utf-8')

from retrieval.vector_store import VectorStore
from pypdf import PdfReader

QUESTIONS_FILE = os.path.join(BASE_DIR, "eval", "questions.jsonl")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
SOURCES_CSV = os.path.join(BASE_DIR, "data", "sources.csv")

# Map doc_id to file_name
import csv
doc_files = {}
with open(SOURCES_CSV, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        doc_files[row["doc_id"]] = row["file_name"]

store = VectorStore()

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

answerable = [q for q in questions if q.get("answerable") is True]

# Cache pdf readers
pdf_cache = {}
def get_pdf(doc_id):
    if doc_id not in pdf_cache:
        fname = doc_files[doc_id]
        pdf_cache[doc_id] = PdfReader(os.path.join(RAW_DIR, fname))
    return pdf_cache[doc_id]

def get_page_text(doc_id, page_num):
    reader = get_pdf(doc_id)
    if 1 <= page_num <= len(reader.pages):
        return reader.pages[page_num - 1].extract_text() or ""
    return ""

def search_doc_for_text(doc_id, keywords):
    reader = get_pdf(doc_id)
    matches = []
    for idx, page in enumerate(reader.pages, 1):
        txt = page.extract_text() or ""
        if all(kw.lower() in txt.lower() for kw in keywords):
            matches.append(idx)
    return matches

print("=" * 80)
print(f"DIAGNOSING ALL {len(answerable)} ANSWERABLE EVAL QUESTIONS")
print("=" * 80)

diagnoses = []

for q in answerable:
    qid = q["id"]
    query = q["question"]
    exp_doc = q["source_doc"]
    exp_page = int(q["source_page"])
    ground_truth = q.get("ground_truth", "")

    retrieved = store.search(query, k=10)

    doc_ranks = [idx for idx, r in enumerate(retrieved, 1) if r.doc_id == exp_doc]
    first_doc_rank = doc_ranks[0] if doc_ranks else None

    page_ranks = [idx for idx, r in enumerate(retrieved, 1) if r.doc_id == exp_doc and r.page == exp_page]
    first_page_rank = page_ranks[0] if page_ranks else None

    exp_page_text = get_page_text(exp_doc, exp_page)

    diagnoses.append({
        "id": qid,
        "query": query,
        "expected_doc": exp_doc,
        "expected_page": exp_page,
        "ground_truth": ground_truth,
        "first_doc_rank": first_doc_rank,
        "first_page_rank": first_page_rank,
        "retrieved_top5": [(r.doc_id, r.page, r.score, r.content[:120]) for r in retrieved[:5]],
        "exp_page_len": len(exp_page_text),
        "exp_page_snippet": exp_page_text[:200].replace("\n", " ")
    })

# Output summaries for low page rank
for d in diagnoses:
    print(f"\n[{d['id']}] {d['query']}")
    print(f"  Expected: [{d['expected_doc']}, p.{d['expected_page']}] | First Doc Rank: {d['first_doc_rank']} | First Page Rank: {d['first_page_rank']}")
    print(f"  Exp Page Len: {d['exp_page_len']} chars | Snippet: {d['exp_page_snippet'][:100]}")
    print("  Top-3 Retrieved:")
    for doc, p, sc, snip in d['retrieved_top5'][:3]:
        print(f"    - [{doc}, p.{p}] score={sc} | {snip[:80]}...")
