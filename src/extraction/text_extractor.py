"""Common text extraction interfaces and TXT extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


class ExtractionError(Exception):
    """Raised when a document cannot be extracted."""


@dataclass
class PageResult:
    """Text for a single page."""

    page_number: int
    page_text: str


@dataclass
class DocumentResult:
    """Consistent result structure for any extracted document."""

    filename: str
    file_type: str
    extracted_text: str
    pages: List[PageResult]
    page_count: int
    extraction_warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _validate_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename.strip():
        raise ExtractionError("A filename is required.")
    # Use only the basename; never trust directory components.
    safe_name = os.path.basename(filename.replace("\\", "/"))
    if safe_name in ("", ".", ".."):
        raise ExtractionError("Invalid filename.")
    return safe_name


def _validate_size(content: bytes) -> None:
    if not isinstance(content, (bytes, bytearray)):
        raise ExtractionError("Document content must be provided as bytes.")
    if len(content) == 0:
        # Empty files are allowed for TXT; size check only. PDF empty handled later.
        pass
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ExtractionError(
            f"File exceeds the maximum allowed size of {MAX_FILE_SIZE_BYTES} bytes."
        )


def _extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def extract_txt(content: bytes, filename: str = "document.txt") -> DocumentResult:
    """Extract text from a TXT file provided as bytes."""
    safe_name = _validate_filename(filename)
    _validate_size(content)

    if len(content) == 0:
        return DocumentResult(
            filename=safe_name,
            file_type="txt",
            extracted_text="",
            pages=[],
            page_count=0,
            extraction_warnings=["The text file is empty."],
            metadata={},
        )

    # Decode with sensible fallbacks.
    text: str
    used_fallback = False
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("cp1252")
            used_fallback = True
        except UnicodeDecodeError:
            text = content.decode("latin-1")
            used_fallback = True

    # Normalize line endings without collapsing paragraph boundaries.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    warnings: List[str] = []
    if used_fallback:
        warnings.append(
            "File was not valid UTF-8; decoded using a fallback encoding."
        )

    pages = [PageResult(page_number=1, page_text=text)]

    return DocumentResult(
        filename=safe_name,
        file_type="txt",
        extracted_text=text,
        pages=pages,
        page_count=1,
        extraction_warnings=warnings,
        metadata={},
    )


def extract_document(filename: str, content: bytes) -> DocumentResult:
    """Common extraction interface for PDF and TXT files.

    Dispatches based on the file extension. Raises ExtractionError for
    unsupported types or invalid input.
    """
    safe_name = _validate_filename(filename)
    _validate_size(content)

    ext = _extension(safe_name)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ExtractionError(
            f"Unsupported file type {ext!r}. Supported types: .pdf, .txt."
        )

    if ext == ".pdf":
        # Lazy import to avoid circular dependency.
        from .pdf_extractor import extract_pdf

        return extract_pdf(content, filename=safe_name)
    return extract_txt(content, filename=safe_name)
