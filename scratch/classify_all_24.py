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

def search_doc_for_snippets(doc_id, keywords):
    reader = get_pdf(doc_id)
    matches = []
    for idx, page in enumerate(reader.pages, 1):
        txt = page.extract_text() or ""
        score = sum(1 for kw in keywords if kw.lower() in txt.lower())
        if score > 0:
            matches.append((idx, score))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

classifications = []

for q in answerable:
    qid = q["id"]
    query = q["question"]
    exp_doc = q["source_doc"]
    exp_page = int(q["source_page"])
    exp_ans = q.get("expected_answer", "")

    retrieved = store.search(query, k=10)

    first_doc_rank = None
    first_page_rank = None
    for idx, r in enumerate(retrieved, 1):
        if r.doc_id == exp_doc and first_doc_rank is None:
            first_doc_rank = idx
        if r.doc_id == exp_doc and r.page == exp_page and first_page_rank is None:
            first_page_rank = idx

    exp_page_txt = get_page_text(exp_doc, exp_page)

    top5 = [(r.doc_id, r.page, r.score) for r in retrieved[:5]]

    classifications.append({
        "qid": qid,
        "query": query,
        "exp_doc": exp_doc,
        "exp_page": exp_page,
        "exp_ans": exp_ans,
        "first_doc_rank": first_doc_rank,
        "first_page_rank": first_page_rank,
        "exp_page_len": len(exp_page_txt),
        "exp_page_snippet": exp_page_txt[:150].replace("\n", " "),
        "top5": top5,
    })

with open("scratch/classification_raw.json", "w", encoding="utf-8") as out:
    json.dump(classifications, out, indent=2)

print(f"Dumped raw analysis for {len(classifications)} questions to scratch/classification_raw.json")
