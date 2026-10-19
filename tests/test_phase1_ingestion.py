"""
Unit Tests for Phase 1: Document Parsing, Text Cleaning & Chunking
Validates:
1. clean_text repairs broken hyphenations and normalizes whitespace.
2. Scanned / empty page detection properly flags low-character pages.
3. Chunker produces no empty chunks.
4. Metadata (doc_id, title, page) is strictly preserved on every chunk.
5. Chunk overlap is present and correct across consecutive chunks.
"""

import os
import pytest
from ingestion.parser import clean_text, PageData
from ingestion.chunker import DocumentChunker, estimate_tokens, TextChunk


def test_clean_text_hyphenation_and_footers():
    raw_sample = (
        "The Institute follows strict regu-\nlation for exam-\ninations.\n\n"
        "Page 12 of 45\n\n"
        "Students must attend classes.  - 12 -  \n"
        "Next paragraph with multiple    spaces."
    )
    cleaned = clean_text(raw_sample)

    assert "regulation" in cleaned, "Hyphenated word 'regu-\\nlation' was not repaired"
    assert "examinations" in cleaned, "Hyphenated word 'exam-\\ninations' was not repaired"
    assert "Page 12 of 45" not in cleaned, "Footer 'Page 12 of 45' was not removed"
    assert "- 12 -" not in cleaned, "Footer '- 12 -' was not removed"
    assert "multiple    spaces" not in cleaned, "Multiple spaces were not normalized"
    assert "multiple spaces" in cleaned


def test_page_data_scanned_detection():
    normal_page = PageData(
        doc_id="test_doc",
        title="Test Document",
        page_number=1,
        text="This is a legitimate academic regulation page with plenty of extracted text.",
        is_scanned=False,
    )
    assert not normal_page.is_scanned

    scanned_page = PageData(
        doc_id="test_doc",
        title="Test Document",
        page_number=2,
        text="Scan",
        is_scanned=True,
    )
    assert scanned_page.is_scanned


def test_chunker_no_empty_chunks_and_metadata_preserved():
    chunker = DocumentChunker(target_chunk_tokens=50, overlap_tokens=10)

    # 4 paragraphs with diverse content
    paragraphs = [
        "Paragraph 1 contains important policy information regarding minimum required grades. " * 3,
        "Paragraph 2 discusses attendance criteria and medical leave submission procedures. " * 3,
        "Paragraph 3 outlines degree requirements and credits needed for graduation. " * 3,
        "Paragraph 4 describes hostel regulations, curfew times, and disciplinary penalties. " * 3,
    ]
    full_text = "\n\n".join(paragraphs)

    page = PageData(
        doc_id="policy_doc",
        title="IITJ Academic Policy",
        page_number=5,
        text=full_text,
        is_scanned=False,
        applies_to="Students enrolled from July 2019 onwards",
        document_date="2019-09-03",
    )

    chunks = chunker.chunk_page(page)

    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

    for i, c in enumerate(chunks):
        assert c.content.strip(), f"Chunk {i} must not be empty"
        assert c.doc_id == "policy_doc", "doc_id not preserved"
        assert c.title == "IITJ Academic Policy", "title not preserved"
        assert c.page == 5, "page number not preserved"
        assert c.chunk_index == i, f"Expected chunk_index {i}, got {c.chunk_index}"
        assert c.chunk_id == f"policy_doc_p5_c{i}", f"Invalid chunk_id: {c.chunk_id}"
        assert c.token_count > 0, "token_count must be positive"
        assert c.applies_to == "Students enrolled from July 2019 onwards", "applies_to not preserved"
        assert c.document_date == "2019-09-03", "document_date not preserved"


def test_chunker_overlap_presence():
    chunker = DocumentChunker(target_chunk_tokens=40, overlap_tokens=15)

    sentences = [
        f"Sentence number {i} explains academic ordinance clause {i} in great detail."
        for i in range(1, 15)
    ]
    page = PageData(
        doc_id="overlap_test",
        title="Overlap Test Document",
        page_number=1,
        text="\n\n".join(sentences),
    )

    chunks = chunker.chunk_page(page)
    assert len(chunks) >= 2, "Need at least 2 chunks to verify overlap"

    # Verify that words from the end of chunk 0 appear in chunk 1
    chunk0_words = chunks[0].content.split()
    tail_words = set(chunk0_words[-10:])

    chunk1_words = chunks[1].content.split()
    head_words = set(chunk1_words[:20])

    common_overlap = tail_words.intersection(head_words)
    assert len(common_overlap) > 0, f"Expected overlap between chunk 0 and chunk 1, but found common words: {common_overlap}"


def test_chunker_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        DocumentChunker(target_chunk_tokens=100, overlap_tokens=100)
    with pytest.raises(ValueError):
        DocumentChunker(target_chunk_tokens=100, overlap_tokens=150)


def test_processed_chunks_file_exists_and_valid():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_path = os.path.join(base_dir, "data", "processed", "chunks.jsonl")
    assert os.path.exists(chunks_path), f"Processed chunks file missing at: {chunks_path}"

    import json
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            chunks.append(item)
            assert item.get("chunk_id"), f"Missing chunk_id on line {idx}"
            assert item.get("doc_id"), f"Missing doc_id on line {idx}"
            assert item.get("title"), f"Missing title on line {idx}"
            assert item.get("page") is not None, f"Missing page on line {idx}"
            assert item.get("content", "").strip(), f"Empty chunk content on line {idx}"
            assert item.get("token_count", 0) > 0, f"Invalid token count on line {idx}"
            assert "applies_to" in item, f"Missing applies_to on line {idx}"
            assert "document_date" in item, f"Missing document_date on line {idx}"

    assert len(chunks) >= 300, f"Expected at least 300 chunks, found {len(chunks)}"

