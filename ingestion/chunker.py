"""
CampusIQ Document Chunker
Splits document pages into semantically cohesive chunks using paragraph
and sentence boundaries with configurable token size and sliding overlap.
"""

import re
from typing import Dict, List, Optional
from ingestion.parser import PageData


def estimate_tokens(text: str) -> int:
    """
    Estimates token count for English text:
    Average English token in BGE / LLM tokenizers is ~0.75 words (or ~4 characters).
    We estimate as: max(1, int(len(text.split()) * 1.3))
    """
    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) * 1.3))


class TextChunk:
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        title: str,
        page: int,
        chunk_index: int,
        content: str,
        token_count: int,
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.title = title
        self.page = page
        self.chunk_index = chunk_index
        self.content = content
        self.token_count = token_count

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "token_count": self.token_count,
            "char_count": len(self.content),
        }


class DocumentChunker:
    def __init__(self, target_chunk_tokens: int = 500, overlap_tokens: int = 50):
        if overlap_tokens >= target_chunk_tokens:
            raise ValueError("overlap_tokens must be strictly less than target_chunk_tokens")
        self.target_chunk_tokens = target_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def _split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs if paragraphs else [text.strip()]

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        # Split on sentence boundaries: periods, exclamation marks, question marks
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_page(self, page_data: PageData) -> List[TextChunk]:
        """
        Chunks text from a single PageData object.
        Maintains page-level traceability and section cohesion.
        """
        if not page_data.text or page_data.is_scanned:
            return []

        raw_paragraphs = self._split_into_paragraphs(page_data.text)
        units: List[str] = []

        for p in raw_paragraphs:
            if estimate_tokens(p) > self.target_chunk_tokens:
                units.extend(self._split_large_paragraph(p))
            else:
                units.append(p)

        chunks: List[TextChunk] = []
        current_units: List[str] = []
        current_tokens = 0
        chunk_idx = 0

        for unit in units:
            unit_tokens = estimate_tokens(unit)
            if unit_tokens == 0:
                continue

            if current_tokens + unit_tokens > self.target_chunk_tokens and current_units:
                # Flush the current chunk
                content = "\n\n".join(current_units).strip()
                if content:
                    chunk_id = f"{page_data.doc_id}_p{page_data.page_number}_c{chunk_idx}"
                    chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            doc_id=page_data.doc_id,
                            title=page_data.title,
                            page=page_data.page_number,
                            chunk_index=chunk_idx,
                            content=content,
                            token_count=estimate_tokens(content),
                        )
                    )
                    chunk_idx += 1

                # Build overlap from the tail of current_units
                overlap_units: List[str] = []
                accumulated_overlap = 0
                for prev in reversed(current_units):
                    p_tokens = estimate_tokens(prev)
                    if accumulated_overlap + p_tokens <= self.overlap_tokens:
                        overlap_units.insert(0, prev)
                        accumulated_overlap += p_tokens
                    else:
                        break

                current_units = overlap_units + [unit]
                current_tokens = sum(estimate_tokens(u) for u in current_units)
            else:
                current_units.append(unit)
                current_tokens += unit_tokens

        # Flush any remaining text as the last chunk
        if current_units:
            content = "\n\n".join(current_units).strip()
            if content:
                chunk_id = f"{page_data.doc_id}_p{page_data.page_number}_c{chunk_idx}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        doc_id=page_data.doc_id,
                        title=page_data.title,
                        page=page_data.page_number,
                        chunk_index=chunk_idx,
                        content=content,
                        token_count=estimate_tokens(content),
                    )
                )

        return chunks

    def chunk_document(self, pages: List[PageData]) -> List[TextChunk]:
        """
        Chunks all pages of a document while preserving 1-based page metadata.
        """
        all_chunks: List[TextChunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page)
            all_chunks.extend(page_chunks)
        return all_chunks
