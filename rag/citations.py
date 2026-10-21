"""
CampusIQ Citation Parser and Validator
Parses grounded [Title, p.X] citation markers from generated answer text,
maps them to retrieved chunks (with applies_to and file_date),
and flags any citations not present in retrieved chunks as invalid_citation.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union


def normalize_title(title: str) -> str:
    """Normalize a document title for fuzzy matching."""
    t = re.sub(r"[^\w\s]", "", title).lower()
    return re.sub(r"\s+", " ", t).strip()


def parse_marker(marker_text: str) -> Tuple[str, List[int]]:
    """
    Parses a citation marker text like 'Academic Regulations, p.8, 10'
    into (title, [8, 10]).
    """
    match = re.search(r"^(?P<title>.+?),\s*(?:p\.|p|page|pages)\s*(?P<pages>[\d\s,\-]+)$", marker_text.strip(), re.IGNORECASE)
    if not match:
        # Fallback split on last comma
        if "," in marker_text:
            parts = marker_text.rsplit(",", 1)
            title = parts[0].strip()
            pages_raw = parts[1].replace("p.", "").replace("p", "").strip()
        else:
            return marker_text.strip(), []
    else:
        title = match.group("title").strip()
        pages_raw = match.group("pages").strip()

    pages: List[int] = []
    for item in re.split(r"[,;]+", pages_raw):
        item = item.strip()
        if "-" in item:
            try:
                start_p, end_p = item.split("-", 1)
                pages.extend(range(int(start_p.strip()), int(end_p.strip()) + 1))
            except ValueError:
                pass
        elif item.isdigit():
            pages.append(int(item))

    return title, pages


def extract_citations(
    answer_text: str,
    retrieved_chunks: List[Any],
) -> List[Dict[str, Any]]:
    """
    Finds all citation markers [Title, p.X] in answer_text.
    Maps each marker to the retrieved chunk that provides it.
    If a marker matches no retrieved chunk, sets invalid_citation=True.
    """
    if not answer_text or not answer_text.strip():
        return []

    # Find all bracketed tokens that look like citations
    pattern = re.compile(r"\[([^\]]+?,\s*(?:p\.|p|page|pages)\s*[^\]]+)\]", re.IGNORECASE)
    raw_markers = pattern.findall(answer_text)

    # Normalize retrieved chunks to a uniform dictionary format
    normalized_chunks = []
    for c in retrieved_chunks:
        if hasattr(c, "to_dict"):
            cd = c.to_dict()
        elif isinstance(c, dict):
            cd = c
        else:
            cd = {
                "chunk_id": getattr(c, "chunk_id", ""),
                "title": getattr(c, "title", ""),
                "page": getattr(c, "page", 0),
                "applies_to": getattr(c, "applies_to", ""),
                "file_date": getattr(c, "file_date", getattr(c, "document_date", "")),
            }
        normalized_chunks.append({
            "chunk_id": cd.get("chunk_id", ""),
            "title": cd.get("title", ""),
            "page": int(cd.get("page", 0)),
            "applies_to": cd.get("applies_to", ""),
            "file_date": cd.get("file_date", cd.get("document_date", "")),
            "norm_title": normalize_title(cd.get("title", "")),
        })

    citations: List[Dict[str, Any]] = []
    seen_keys = set()

    for marker in raw_markers:
        title, pages = parse_marker(marker)
        norm_title = normalize_title(title)

        matched_any = False

        # If pages were found, try matching each page
        target_pages = pages if pages else [None]

        for p in target_pages:
            matched_chunk = None
            for c in normalized_chunks:
                # Page match
                page_matches = (p is None or c["page"] == p)
                # Title match
                title_matches = (
                    norm_title == c["norm_title"]
                    or norm_title in c["norm_title"]
                    or c["norm_title"] in norm_title
                )
                if page_matches and title_matches:
                    matched_chunk = c
                    break

            if matched_chunk:
                matched_any = True
                cit_key = (matched_chunk["title"], matched_chunk["page"])
                if cit_key not in seen_keys:
                    seen_keys.add(cit_key)
                    formatted_parts = [f"[{matched_chunk['title']}, p.{matched_chunk['page']}]"]
                    if matched_chunk["file_date"]:
                        formatted_parts.append(f"Date: {matched_chunk['file_date']}")
                    if matched_chunk["applies_to"]:
                        formatted_parts.append(f"Applies to: {matched_chunk['applies_to']}")
                    
                    citations.append({
                        "marker": f"[{marker}]",
                        "doc_title": matched_chunk["title"],
                        "page": matched_chunk["page"],
                        "chunk_id": matched_chunk["chunk_id"],
                        "applies_to": matched_chunk["applies_to"],
                        "file_date": matched_chunk["file_date"],
                        "formatted": " | ".join(formatted_parts),
                        "invalid_citation": False,
                    })

        if not matched_any:
            # The cited marker matches no retrieved chunk -> flag invalid_citation
            raw_key = (title, tuple(pages))
            if raw_key not in seen_keys:
                seen_keys.add(raw_key)
                citations.append({
                    "marker": f"[{marker}]",
                    "doc_title": title,
                    "page": pages[0] if pages else None,
                    "pages": pages,
                    "chunk_id": None,
                    "applies_to": None,
                    "file_date": None,
                    "formatted": f"[{marker}]",
                    "invalid_citation": True,
                })

    return citations
