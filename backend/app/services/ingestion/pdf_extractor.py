"""
Bureaucracy AI — PDF Extractor

Extracts text from PDF files using PyMuPDF.
If a page returns no text (scanned/image PDF), automatically falls back
to Tesseract OCR via OCRService.

Returns a list of page dicts:
    [{"page_number": int, "text": str, "has_text": bool, "ocr_used": bool}]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pymupdf

logger = logging.getLogger(__name__)

# Minimum number of characters on a page to consider it "has text".
# Pages with fewer characters trigger the OCR fallback.
MIN_TEXT_CHARS = 20


class PDFExtractor:
    """
    Extracts text from PDF files, falling back to OCR for image-only pages.

    Usage::

        extractor = PDFExtractor()
        pages = extractor.extract_text("path/to/file.pdf")
        # pages = [{"page_number": 1, "text": "...", "has_text": True, "ocr_used": False}, ...]
    """

    def extract_text(self, file_path: str) -> List[Dict]:
        pdf_path = Path(file_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not pdf_path.is_file():
            raise ValueError("The provided path is not a file.")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        pages: List[Dict] = []

        try:
            with pymupdf.open(pdf_path) as document:
                if document.is_encrypted:
                    raise ValueError("Encrypted PDFs are not supported.")

                for page_number, page in enumerate(document, start=1):
                    text = page.get_text("text").strip()
                    ocr_used = False

                    if len(text) < MIN_TEXT_CHARS:
                        # Page has no selectable text — attempt OCR.
                        ocr_text = self._ocr_page(document, page, page_number)
                        if ocr_text:
                            text = ocr_text
                            ocr_used = True
                            logger.debug(
                                f"Page {page_number}: OCR produced "
                                f"{len(text)} chars"
                            )
                        else:
                            logger.debug(
                                f"Page {page_number}: no text from PyMuPDF or OCR"
                            )

                    pages.append({
                        "page_number": page_number,
                        "text": text,
                        "has_text": bool(text),
                        "ocr_used": ocr_used,
                    })

        except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
            raise ValueError(f"Unable to read PDF: {error}") from error

        return pages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ocr_page(self, document, page, page_number: int) -> str:
        """
        Render *page* to a PIL image and pass it to OCRService.

        Returns the OCR'd text string, or empty string on failure.
        """
        try:
            from backend.app.services.ingestion.ocr_service import get_ocr_service

            ocr_service = get_ocr_service()
            if not ocr_service.is_available():
                logger.debug("OCR service not available (Tesseract not installed)")
                return ""

            # Render the page as a pixmap at 2x scale for better OCR accuracy.
            matrix = pymupdf.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=matrix)

            # Convert pixmap bytes to a PIL Image for OCRService.
            try:
                from PIL import Image
                import io
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))
                return ocr_service.extract_image(image)
            except ImportError:
                # PIL not available — save pixmap to temp file and pass path.
                import tempfile, os
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                pix.save(tmp_path)
                try:
                    result = ocr_service.extract_image_path(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                return result

        except Exception as exc:
            logger.warning(f"OCR fallback failed for page {page_number}: {exc}")
            return ""
