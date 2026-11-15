"""Tests for PDF extraction."""

from __future__ import annotations

import pymupdf
import pytest

from extraction.pdf_extractor import extract_pdf
from extraction.text_extractor import ExtractionError


def _make_pdf(page_texts: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_valid_single_page_pdf() -> None:
    data = _make_pdf(["Hello world"])
    result = extract_pdf(data, filename="sample.pdf")
    assert result.file_type == "pdf"
    assert result.page_count == 1
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert "Hello world" in result.extracted_text
    assert result.extraction_warnings == []


def test_extract_multi_page_pdf() -> None:
    data = _make_pdf(["Page one", "Page two", "Page three"])
    result = extract_pdf(data, filename="multi.pdf")
    assert result.page_count == 3
    assert [p.page_number for p in result.pages] == [1, 2, 3]
    assert "Page one" in result.pages[0].page_text
    assert "Page two" in result.pages[1].page_text
    assert "Page three" in result.pages[2].page_text


def test_extract_pdf_with_empty_page() -> None:
    data = _make_pdf(["Has text", "", "More text"])
    result = extract_pdf(data, filename="empty_page.pdf")
    assert result.page_count == 3
    assert result.pages[1].page_text.strip() == ""
    assert "Has text" in result.extracted_text
    assert "More text" in result.extracted_text


def test_extract_invalid_pdf_raises() -> None:
    with pytest.raises(ExtractionError):
        extract_pdf(b"this is not a pdf", filename="bad.pdf")


def test_extract_empty_pdf_content_raises() -> None:
    with pytest.raises(ExtractionError):
        extract_pdf(b"", filename="empty.pdf")


def test_extract_pdf_no_text_warns() -> None:
    data = _make_pdf(["", ""])
    result = extract_pdf(data, filename="scanned.pdf")
    assert result.extracted_text == ""
    assert any("No extractable text" in w for w in result.extraction_warnings)


def test_extract_pdf_oversize_raises() -> None:
    # Just over the limit
    big = b"%PDF-1.4\n" + b"x" * (10 * 1024 * 1024 + 1)
    with pytest.raises(ExtractionError):
        extract_pdf(big, filename="big.pdf")
