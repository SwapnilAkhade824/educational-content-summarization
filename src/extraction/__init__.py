"""Document text extraction (PDF and TXT)."""

from .pdf_extractor import extract_pdf
from .text_extractor import (
    DocumentResult,
    ExtractionError,
    MAX_FILE_SIZE_BYTES,
    PageResult,
    extract_document,
    extract_txt,
)

__all__ = [
    "DocumentResult",
    "ExtractionError",
    "MAX_FILE_SIZE_BYTES",
    "PageResult",
    "extract_document",
    "extract_pdf",
    "extract_txt",
]
