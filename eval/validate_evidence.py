"""
Evidence Validator for CampusIQ Eval Set
Validates that for every answerable question in eval/questions.jsonl:
1. The evidence string is under 15 words.
2. The evidence string is an exact contiguous substring of the extracted PDF text
   (after whitespace normalization and zero-width character removal).
3. The contiguous substring is found on at least one of the pages in expected_pages.
"""

import json
import pathlib
import sys
import io
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('pypdf').setLevel(logging.ERROR)
import pypdf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

raw_dir = pathlib.Path(__file__).parent.parent / "data" / "raw"
eval_file = pathlib.Path(__file__).parent / "questions.jsonl"

doc_map = {
    "rules_ug_2019": "Academic_Regulations_Final_03_09_2019.pdf",
    "rules_pg_2022": "4_Regulation_PG_2022-onwards_20022023.pdf",
    "curriculum_aids": "Curriculum-BTech-AIDS.pdf",
    "curriculum_systems_science": "Curricula_SS.pdf",
    "curriculum_physics": "Curriculum_BS_Physics.pdf",
    "curriculum_mtech_cps": "Proposed-CurriculumMTech-CPS.pdf",
    "rules_msc": "M.Sc_Regulations.pdf",
    "rules_mtech": "MTech_New_REVISED_13Feb2017.pdf",
    "academic_calendar_ay2026_27": "Academic-Calendar-AY-2026-27.pdf",
    "academic_calendar_summer2026": "Academic-Calendar-for-Summer-Term-2026.pdf",
    "statutes_code_of_conduct": "Statutes-English-IIT-JODHPUR.pdf",
    "circular_attendance_2024": "7.1_Circular_51Circular-No.-IITJOAACircular2024-2551682-dated-08-July-2024.pdf",
    "circular_examination_2024": "7.2_Circular_52Circular-No.-IITJOAACircular2024-2552683-dated-08-July-2024.pdf",
}

def clean_ws(text: str) -> str:
    text = text.replace('\u200b', ' ').replace('\ufeff', ' ')
    return ' '.join(text.split())

def main():
    print(f"Loading PDFs from {raw_dir}...")
    pdf_clean = {}
    for doc_id, fname in doc_map.items():
        fpath = raw_dir / fname
        if fpath.exists():
            reader = pypdf.PdfReader(str(fpath))
            pdf_clean[doc_id] = [clean_ws(p.extract_text() or '') for p in reader.pages]

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    answerable = [q for q in questions if q.get("answerable")]
    print(f"\nValidating {len(answerable)} answerable questions:\n")

    passes = 0
    fails = 0

    for q in answerable:
        qid = q["id"]
        sdoc = q.get("source_doc")
        ev = q.get("evidence", "")
        pages = q.get("expected_pages", [])

        words = ev.split()
        w_count = len(words)
        norm_ev = clean_ws(ev)

        found_on = []
        if sdoc in pdf_clean:
            for p in pages:
                if 1 <= p <= len(pdf_clean[sdoc]):
                    if norm_ev in pdf_clean[sdoc][p - 1]:
                        found_on.append(p)

        is_valid = bool(found_on and w_count < 15 and ev.strip())
        status = "PASS" if is_valid else "FAIL"

        if is_valid:
            passes += 1
            print(f"[{status}] {qid} (words={w_count}, found_on={found_on}) -> \"{ev}\"")
        else:
            fails += 1
            reasons = []
            if w_count >= 15:
                reasons.append(f"word count {w_count} >= 15")
            if not found_on:
                reasons.append(f"quote not found on expected_pages {pages}")
            print(f"[{status}] {qid} (words={w_count}, found_on={found_on}) -> \"{ev}\" [Reason: {', '.join(reasons)}]")

    print(f"\n==========================================")
    print(f"RESULT: {passes} PASSED, {fails} FAILED out of {len(answerable)} answerable questions")
    print(f"==========================================")

    if fails > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
