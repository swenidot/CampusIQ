import sys
import json
import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.stdout.reconfigure(encoding='utf-8')

from retrieval.vector_store import VectorStore
from pypdf import PdfReader

QUESTIONS_FILE = os.path.join(BASE_DIR, "eval", "questions.jsonl")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
SOURCES_CSV = os.path.join(BASE_DIR, "data", "sources.csv")

doc_files = {}
with open(SOURCES_CSV, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        doc_files[row["doc_id"]] = row["file_name"]

store = VectorStore()

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

answerable = [q for q in questions if q.get("answerable") is True]

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

def find_pages_with_terms(doc_id, terms):
    reader = get_pdf(doc_id)
    hits = []
    for p_idx, page in enumerate(reader.pages, 1):
        txt = (page.extract_text() or "").lower()
        if all(t.lower() in txt for t in terms):
            hits.append(p_idx)
    return hits

results = []

for q in answerable:
    qid = q["id"]
    query = q["question"]
    exp_doc = q["source_doc"]
    exp_page = int(q["source_page"])
    gt = q.get("ground_truth", "")

    retrieved = store.search(query, k=10)

    first_doc_rank = None
    first_page_rank = None
    for idx, r in enumerate(retrieved, 1):
        if r.doc_id == exp_doc and first_doc_rank is None:
            first_doc_rank = idx
        if r.doc_id == exp_doc and r.page == exp_page and first_page_rank is None:
            first_page_rank = idx

    exp_page_text = get_page_text(exp_doc, exp_page)
    
    # Analyze if ground truth text is actually on exp_page
    # Extract key terms from ground_truth
    gt_terms = [w.strip(".,;:()") for w in gt.split() if len(w) > 4][:5]
    gt_in_exp_page = any(term.lower() in exp_page_text.lower() for term in gt_terms) if gt_terms else False

    results.append({
        "qid": qid,
        "query": query,
        "exp_doc": exp_doc,
        "exp_page": exp_page,
        "gt": gt,
        "doc_rank": first_doc_rank,
        "page_rank": first_page_rank,
        "exp_page_has_gt": gt_in_exp_page,
        "retrieved_ranks": [(r.doc_id, r.page, r.score) for r in retrieved[:5]],
    })

print(json.dumps(results, indent=2))
