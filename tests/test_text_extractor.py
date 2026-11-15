"""Tests for TXT extraction and common extraction interface."""

from __future__ import annotations

import pytest

from extraction.text_extractor import (
    ExtractionError,
    extract_document,
    extract_txt,
)


def test_extract_valid_txt() -> None:
    content = b"Hello world.\n\nSecond paragraph."
    result = extract_txt(content, filename="notes.txt")
    assert result.file_type == "txt"
    assert result.extracted_text == "Hello world.\n\nSecond paragraph."
    assert result.page_count == 1
    assert result.pages[0].page_number == 1
    assert result.extraction_warnings == []


def test_extract_empty_txt() -> None:
    result = extract_txt(b"", filename="empty.txt")
    assert result.extracted_text == ""
    assert result.page_count == 0
    assert result.pages == []
    assert any("empty" in w.lower() for w in result.extraction_warnings)


def test_extract_txt_utf8_bom() -> None:
    content = "\ufeffBOM text".encode("utf-8")
    result = extract_txt(content, filename="bom.txt")
    assert result.extracted_text == "BOM text"
    assert result.extraction_warnings == []


def test_extract_txt_non_utf8_fallback() -> None:
    # cp1252-encoded smart quotes
    content = "Café".encode("cp1252")
    result = extract_txt(content, filename="legacy.txt")
    assert "Caf" in result.extracted_text
    assert any("fallback" in w.lower() for w in result.extraction_warnings)


def test_extract_txt_preserves_paragraphs() -> None:
    content = b"Line one.\nLine two.\n\nNew paragraph."
    result = extract_txt(content, filename="p.txt")
    assert "\n\n" in result.extracted_text


def test_extract_document_dispatches_pdf() -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PDF text")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = extract_document("doc.pdf", pdf_bytes)
    assert result.file_type == "pdf"
    assert "PDF text" in result.extracted_text


def test_extract_document_dispatches_txt() -> None:
    result = extract_document("notes.txt", b"hi")
    assert result.file_type == "txt"
    assert result.extracted_text == "hi"


def test_extract_document_unsupported_type_raises() -> None:
    with pytest.raises(ExtractionError):
        extract_document("image.png", b"fake")


def test_extract_document_invalid_filename_raises() -> None:
    with pytest.raises(ExtractionError):
        extract_document("", b"hi")
    with pytest.raises(ExtractionError):
        extract_document("../etc/passwd", b"hi")


def test_extract_document_oversize_raises() -> None:
    big = b"a" * (10 * 1024 * 1024 + 1)
    with pytest.raises(ExtractionError):
        extract_document("big.txt", big)
