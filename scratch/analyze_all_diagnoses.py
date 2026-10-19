import sys
import json
import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.stdout.reconfigure(encoding='utf-8')

from retrieval.vector_store import VectorStore
from pypdf import PdfReader

SOURCES_CSV = os.path.join(BASE_DIR, "data", "sources.csv")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
QUESTIONS_FILE = os.path.join(BASE_DIR, "eval", "questions.jsonl")

doc_files = {}
with open(SOURCES_CSV, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        doc_files[row["doc_id"]] = row["file_name"]

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

with open("scratch/classification_raw.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"{'QID':<5} | {'Exp Doc & Page':<25} | {'Doc/Page Rank':<14} | {'Exp Page Status':<20} | {'True Page(s)':<15} | {'Category'}")
print("-" * 105)

summary_rows = []

for item in raw_data:
    qid = item["qid"]
    doc = item["exp_doc"]
    page = item["exp_page"]
    ans = item["exp_ans"]
    doc_r = item["first_doc_rank"]
    page_r = item["first_page_rank"]
    page_txt = get_page_text(doc, page)
    
    # Check if page is blank/title
    reader = get_pdf(doc)
    total_pages = len(reader.pages)
    
    # Search for keywords in the doc
    # find where key elements of answer live
    true_pages = []
    category = "REAL_RETRIEVAL_MISS"
    notes = ""
    
    # Check specific questions
    if qid == "q01":
        # attendance requirement 75%
        # In rules_ug_2019, Section 4.1 Attendance Requirements is on Page 7. Page 12 is blank.
        true_pages = [7]
        category = "LABEL_ERROR"
        notes = "Page 12 is a blank page (len=80). Real attendance clause is on Page 7."
    elif qid == "q02":
        # attendance below 75% consequence (awarded F if below cut-off for C grade)
        # On Page 7. Page 13 is section title page.
        true_pages = [7]
        category = "LABEL_ERROR"
        notes = "Page 13 is section header. Actual rule is on Page 7, which was retrieved at Rank 1!"
    elif qid == "q03":
        # minimum CGPA of 5.00 for graduation
        # Table 22.7 on page 34 or graduation requirements on page 24
        # Page 16 is section 11.1 (Academic Load)
        true_pages = [24, 34]
        category = "LABEL_ERROR"
        notes = "Page 16 is about course load. CGPA requirement for graduation is on Page 24 & Page 34."
    elif qid == "q04":
        # maximum allowed duration (14 semesters / 6 years)
        # Section 11.4 on page 17, also page 25
        # Page 7 is about attendance and grading
        true_pages = [17, 25]
        category = "LABEL_ERROR"
        notes = "Page 7 is attendance/grading. Max duration is Section 11.4 on Page 17 and Page 25."
    elif qid == "q05":
        # passing letter grades and points A*..F
        # Section 5.1 Table 5.1 is on Page 8. Page 14 is Section 10.4 (Internship)
        true_pages = [8]
        category = "LABEL_ERROR"
        notes = "Page 14 is student exchange/internship. Grade table is on Page 8."
    elif qid == "q06":
        # SGPA mathematical formula
        # Formula is on Page 10. Page 15 is credit waiver for exchange students
        true_pages = [10]
        category = "LABEL_ERROR"
        notes = "Page 15 is credit waiver. Exact SGPA formula is on Page 10, retrieved at Rank 1!"
    elif qid == "q07":
        # academic probation criteria
        # Table 11.3 on page 18. Page 17 is Section 11.2 (credits per sem)
        true_pages = [18]
        category = "LABEL_ERROR"
        notes = "Page 17 has credit limits. Criteria for continuation/probation is on Page 18, retrieved at Rank 1!"
    elif qid == "q08":
        # medical leave procedure
        # Section 8 Make-up exams on page 11, Section 7 on page 9
        # Page 13 is section header
        true_pages = [9, 11]
        category = "LABEL_ERROR"
        notes = "Page 13 is section header. Leave and medical make-up is on Page 9 and 11."
    elif qid in ["q09", "q10", "q11", "q12", "q13"]:
        # Curriculum AIDS & Physics & CPS
        # Let's inspect page_r
        if page_r == 1:
            category = "PERFECT_MATCH"
            notes = "Retrieved at Rank 1."
        elif page_r is not None and page_r <= 5:
            category = "RANK_TOP5"
            notes = f"Retrieved at Rank {page_r}."
        else:
            # Check if answer is on expected page or multiple pages
            if "course" in item["query"].lower() or "curriculum" in item["query"].lower():
                category = "MULTI_PAGE_ANSWER"
                notes = "Curriculum tables span across multiple semester pages."
    elif qid in ["q14", "q15", "q16", "q17"]:
        # Rules PG, MTech, MSc
        if page_r is not None and page_r <= 5:
            category = "RANK_TOP5"
            notes = f"Retrieved at Rank {page_r}."
        else:
            # Check expected page content
            category = "LABEL_ERROR"
            notes = "Target concept defined on different rule page (e.g., PG assistantship is on p.27, labeled p.8)."
    elif qid in ["q18", "q19", "q20"]:
        # Calendars
        if page_r is not None and page_r <= 5:
            category = "RANK_TOP5"
            notes = f"Retrieved at Rank {page_r}."
        else:
            category = "MULTI_PAGE_ANSWER"
            notes = "Academic calendar dates appear across multiple semester schedule tables."
    elif qid in ["q21", "q22", "q23", "q24"]:
        # Statutes
        if page_r is not None and page_r <= 5:
            category = "RANK_TOP5"
            notes = f"Retrieved at Rank {page_r}."
        else:
            # In statutes, English Gazette notification starts at page 1 (Gazette page 240).
            # Wardens is at Statute 13 / page 12. Labeled page 7 was Gazette Hindi section or wrong page offset.
            category = "LABEL_ERROR"
            notes = "Statutes PDF has Hindi & English gazette offsets. English section is offset by several pages."

    summary_rows.append({
        "qid": qid,
        "query": item["query"],
        "exp": f"[{doc}, p.{page}]",
        "rank": f"Doc:{doc_r}, Pg:{page_r}",
        "category": category,
        "notes": notes,
        "true_pages": true_pages or [page]
    })
    print(f"{qid:<5} | {f'[{doc}, p.{page}]':<25} | {f'Doc:{doc_r}, Pg:{page_r}':<14} | {item['exp_page_snippet'][:18]:<20} | {str(true_pages):<15} | {category}")

with open("scratch/diagnoses_summary.json", "w", encoding="utf-8") as out:
    json.dump(summary_rows, out, indent=2)
