"""PDF text extraction using PyMuPDF."""

from __future__ import annotations

from typing import Any, Dict, List

import pymupdf

from .text_extractor import (
    DocumentResult,
    ExtractionError,
    MAX_FILE_SIZE_BYTES,
    PageResult,
)


def extract_pdf(content: bytes, filename: str = "document.pdf") -> DocumentResult:
    """Extract text from a PDF provided as bytes.

    Args:
        content: Raw PDF bytes.
        filename: Original filename (used for reporting only).

    Returns:
        DocumentResult with per-page text.

    Raises:
        ExtractionError: If the content is invalid, too large, or not a PDF.
    """
    if not isinstance(content, (bytes, bytearray)):
        raise ExtractionError("PDF content must be provided as bytes.")
    if len(content) == 0:
        raise ExtractionError("The uploaded PDF is empty.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ExtractionError(
            f"File exceeds the maximum allowed size of {MAX_FILE_SIZE_BYTES} bytes."
        )

    warnings: List[str] = []
    pages: List[PageResult] = []
    metadata: Dict[str, Any] = {}

    try:
        doc = pymupdf.open(stream=bytes(content), filetype="pdf")
    except Exception as exc:  # pymupdf raises various exceptions for bad input
        raise ExtractionError(f"Invalid or corrupted PDF: {exc}") from exc

    try:
        page_count = doc.page_count
        if page_count == 0:
            warnings.append("PDF contains no pages.")

        for index in range(page_count):
            try:
                page = doc.load_page(index)
                page_text = page.get_text("text") or ""
            except Exception as exc:  # pragma: no cover - defensive
                page_text = ""
                warnings.append(f"Failed to extract text from page {index + 1}: {exc}")
            pages.append(PageResult(page_number=index + 1, page_text=page_text))

        # Metadata (best effort)
        try:
            raw_meta = doc.metadata or {}
            for key in ("title", "author", "subject", "creator"):
                value = raw_meta.get(key)
                if value:
                    metadata[key] = value
        except Exception:  # pragma: no cover - defensive
            pass

        extracted_text = "\n\n".join(p.page_text for p in pages).strip()

        if not extracted_text:
            warnings.append(
                "No extractable text found. The document may be scanned or "
                "image-only; OCR is not performed in this stage."
            )
    finally:
        try:
            doc.close()
        except Exception:  # pragma: no cover - defensive
            pass

    return DocumentResult(
        filename=filename,
        file_type="pdf",
        extracted_text=extracted_text,
        pages=pages,
        page_count=page_count,
        extraction_warnings=warnings,
        metadata=metadata,
    )
