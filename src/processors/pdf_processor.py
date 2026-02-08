"""
PDF processor.

Processing order per page:
  1. Attempt direct text extraction via PyMuPDF (fast, no model needed).
  2. If no text is found and OCR is enabled, render the page to a PNG
     and hand it to ImageProcessor for a description.
  3. Before any of the above, check whether the page is fully redacted
     (>80 % covered by dark rectangles).  Fully-redacted pages are
     skipped; if *every* page is redacted the document is flagged.
"""
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.base import AbstractProcessor, FileType, FileContent
from src.config import get_config


class PDFProcessor(AbstractProcessor):
    """Extracts text and/or descriptions from PDF files."""

    # Fraction of page area that must be covered by dark fill rectangles
    # before the page is considered fully redacted.
    REDACTION_THRESHOLD = 0.8

    def __init__(self, image_processor=None):
        """
        Args:
            image_processor: An ImageProcessor instance used as OCR
                             fallback.  If None, OCR is effectively disabled.
        """
        self.config = get_config()
        self.image_processor = image_processor

    # ---------------------------------------------------------- interface

    def can_process(self, file_type: FileType) -> bool:
        """Return True only for PDF files."""
        return file_type == FileType.PDF

    def process(self, file_path: Path) -> FileContent:
        """
        Extract text and metadata from a PDF.

        Args:
            file_path: Path to the PDF

        Returns:
            FileContent with extracted_text, description, page_count, and
            is_fully_redacted populated.
        """
        extracted_pages: list = []
        page_count = 0
        redacted_page_count = 0

        try:
            doc = fitz.open(str(file_path))
            page_count = len(doc)

            for page_num in range(page_count):
                page = doc[page_num]

                # --- redaction check first ---
                if self._is_page_redacted(page):
                    redacted_page_count += 1
                    continue

                # --- direct text extraction ---
                text = page.get_text()

                # --- OCR fallback for image-based pages ---
                if not text.strip() and self.config.ocr_enabled and self.image_processor:
                    text = self._ocr_page(page, page_num, file_path)

                if text.strip():
                    extracted_pages.append(text)

            doc.close()

            # Assemble results
            is_fully_redacted = (redacted_page_count == page_count) and page_count > 0
            full_text = '\n\n'.join(extracted_pages)

            description = self._build_description(full_text, page_count, is_fully_redacted)

            return FileContent(
                file_id=0,  # Set by Orchestrator
                extracted_text=full_text if full_text else None,
                description=description,
                is_fully_redacted=is_fully_redacted,
                page_count=page_count
            )

        except Exception as e:
            return FileContent(
                file_id=0,
                extracted_text=None,
                description=f"Failed to process PDF: {e}",
                is_fully_redacted=False,
                page_count=0
            )

    # ---------------------------------------------------------- helpers

    def _is_page_redacted(self, page) -> bool:
        """
        Heuristic: a page is 'fully redacted' when dark-filled rectangles
        cover more than REDACTION_THRESHOLD of its area.

        Args:
            page: A PyMuPDF Page object

        Returns:
            True if the page appears fully redacted
        """
        rect = page.rect
        page_area = rect.width * rect.height
        if page_area == 0:
            return False

        drawings = page.get_drawings()
        if not drawings:
            return False

        dark_area = 0.0
        for d in drawings:
            fill = d.get('fill')
            if not fill or len(fill) < 3:
                continue
            # "Dark" = all RGB channels below 0.2
            if all(c < 0.2 for c in fill[:3]):
                r = d.get('rect')
                if r:
                    dark_area += r.width * r.height

        return (dark_area / page_area) > self.REDACTION_THRESHOLD

    def _ocr_page(self, page, page_num: int, pdf_path: Path) -> str:
        """
        Render a single page to PNG and ask ImageProcessor for a description.

        The temporary PNG is deleted immediately after processing.

        Args:
            page: PyMuPDF Page object
            page_num: Zero-based page index (used for temp filename)
            pdf_path: Path to the parent PDF (used to place the temp file)

        Returns:
            OCR result string (may be empty on failure)
        """
        temp_path = pdf_path.parent / f"_ocr_tmp_page_{page_num}.png"
        try:
            pix = page.get_pixmap()
            pix.save(str(temp_path))

            content = self.image_processor.process(temp_path)
            return f"[OCR: {content.description}]" if content.description else ""
        except Exception as e:
            return f"[OCR failed: {e}]"
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _build_description(full_text: str, page_count: int, is_fully_redacted: bool) -> str:
        """Build a human-readable description for search results."""
        if is_fully_redacted:
            return f"PDF document ({page_count} pages) - FULLY REDACTED"

        if full_text:
            preview = full_text[:200].replace('\n', ' ').strip()
            if len(full_text) > 200:
                preview += '...'
            return f"PDF document ({page_count} pages): {preview}"

        return f"PDF document ({page_count} pages) - no extractable text"
