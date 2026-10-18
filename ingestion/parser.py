"""
CampusIQ PDF Parser
Extracts text page-by-page using pypdf, handles text cleaning,
hyphenation repair, and flags scanned/empty pages.
"""

import logging
import re
from typing import Dict, List, Optional
from pypdf import PdfReader

logger = logging.getLogger("campusiq.parser")


class PageData:
    def __init__(self, doc_id: str, title: str, page_number: int, text: str, is_scanned: bool = False):
        self.doc_id = doc_id
        self.title = title
        self.page_number = page_number
        self.text = text
        self.is_scanned = is_scanned

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "page_number": self.page_number,
            "text": self.text,
            "is_scanned": self.is_scanned,
        }


def clean_text(raw_text: str) -> str:
    """
    Cleans raw extracted PDF text:
    1. Replaces non-breaking spaces and special whitespace.
    2. Repairs hyphenated word splits across newlines (e.g., 'regula-\\ntion' -> 'regulation').
    3. Normalizes multiple empty lines and excessive spaces while preserving paragraph breaks.
    4. Strips repeated header/footer artifacts like 'Page X of Y'.
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")

    # Repair hyphenated words at line endings: e.g. "com-\nputer" -> "computer"
    text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", text)

    # Remove standard standalone page numbering footers: e.g. "Page 12 of 45" or "- 12 -"
    text = re.sub(r"(?i)\bpage\s+\d+(\s+of\s+\d+)?\b", "", text)
    text = re.sub(r"(?:^|\s+)-\s*\d+\s*-(?:\s+|$)", " ", text)

    # Normalize excessive horizontal whitespace (multiple spaces/tabs to a single space)
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive vertical whitespace (more than 2 consecutive newlines to 2)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def parse_pdf(file_path: str, doc_id: str, title: str, min_chars_per_page: int = 30) -> List[PageData]:
    """
    Parses a PDF file page by page.
    Returns a list of PageData objects with 1-indexed page numbers.
    Flags pages with fewer than min_chars_per_page as scanned or image-only.
    """
    logger.info(f"Parsing PDF: {file_path} (doc_id={doc_id})")
    reader = PdfReader(file_path)
    pages: List[PageData] = []

    total_pages = len(reader.pages)
    for idx, page in enumerate(reader.pages):
        page_number = idx + 1
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)

        is_scanned = len(cleaned) < min_chars_per_page
        if is_scanned and len(cleaned) == 0:
            logger.debug(f"[{doc_id}] Page {page_number}/{total_pages} has no extractable text (likely image/scanned).")

        pages.append(
            PageData(
                doc_id=doc_id,
                title=title,
                page_number=page_number,
                text=cleaned,
                is_scanned=is_scanned,
            )
        )

    logger.info(f"Parsed {len(pages)} pages for doc_id={doc_id}")
    return pages
