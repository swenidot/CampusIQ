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
