"""
OCR service for Bureaucracy AI.

Extracts text from scanned document images using Tesseract OCR.
Handles multi-language documents (English, Kannada, Hindi).

Tesseract must be installed separately on the system:
  Windows: https://github.com/UB-Mannheim/tesseract/wiki
  
  After installation, add to PATH or set TESSERACT_CMD in .env:
    TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""

import io
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Language codes for Tesseract
_LANG_MAP = {
    "English": "eng",
    "Kannada": "kan",
    "Hindi": "hin",
    "English+Kannada": "eng+kan",
    "English+Hindi": "eng+hin",
}

# Allow overriding Tesseract binary path via environment variable
_TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")


def _configure_tesseract():
    """Point pytesseract to the correct binary if TESSERACT_CMD is set."""
    if _TESSERACT_CMD:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
        except ImportError:
            pass


class OCRService:
    """
    OCR service using Tesseract for image-based document text extraction.

    Gracefully degrades: if Tesseract is not installed, returns a helpful
    error message rather than crashing.
    """

    def __init__(self, default_language: str = "English"):
        self._default_language = default_language
        _configure_tesseract()
        self._tesseract_available = self._check_tesseract()

    def is_available(self) -> bool:
        """Return True if Tesseract is installed and functional."""
        return self._tesseract_available

    def extract_image(self, image, language: str = "English") -> str:
        """
        Extract text from a PIL Image object.
        Returns the extracted text string (empty string on failure).
        """
        try:
            import io as _io
            buf = _io.BytesIO()
            image.save(buf, format="PNG")
            text, _ = self.extract_from_bytes(buf.getvalue(), language=language)
            # Strip the OCR-unavailable placeholder so callers get empty string.
            if text.startswith("[OCR"):
                return ""
            return text
        except Exception as exc:
            logger.warning(f"extract_image failed: {exc}")
            return ""

    def extract_image_path(self, image_path: str, language: str = "English") -> str:
        """
        Extract text from an image file path.
        Returns the extracted text string (empty string on failure).
        """
        text, success = self.extract_from_path(image_path, language=language)
        if not success or text.startswith("[OCR"):
            return ""
        return text

    @staticmethod
    def _check_tesseract() -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_from_bytes(
        self,
        image_bytes: bytes,
        language: str = "English",
        preprocess: bool = True,
    ) -> Tuple[str, bool]:
        """
        Extract text from raw image bytes.

        Returns:
            (extracted_text, success_flag)
        """
        if not self._tesseract_available:
            return (
                "[OCR unavailable: Tesseract not installed. "
                "See https://github.com/UB-Mannheim/tesseract/wiki for installation.]",
                False,
            )

        try:
            from PIL import Image
            import pytesseract

            img = Image.open(io.BytesIO(image_bytes))
            img = self._preprocess(img) if preprocess else img

            lang_code = _LANG_MAP.get(language, "eng")
            text = pytesseract.image_to_string(img, lang=lang_code)
            return text.strip(), bool(text.strip())

        except Exception as exc:
            logger.error(f"OCR extraction failed: {exc}")
            return f"[OCR extraction failed: {exc}]", False

    def extract_from_path(
        self,
        image_path: str,
        language: str = "English",
    ) -> Tuple[str, bool]:
        """Extract text from an image file on disk."""
        path = Path(image_path)
        if not path.exists():
            return f"[Image file not found: {image_path}]", False
        with open(path, "rb") as f:
            return self.extract_from_bytes(f.read(), language=language)

    def extract_pdf_pages_ocr(
        self,
        pdf_bytes: bytes,
        language: str = "English",
        max_pages: int = 50,
    ) -> list:
        """
        For scanned PDFs (where PyMuPDF returns empty text),
        render each page as an image and run OCR on it.

        Returns a list of dicts: [{page_number, text, success}]
        """
        if not self._tesseract_available:
            return [{"page_number": 1, "text": "[OCR unavailable]", "success": False}]

        try:
            import pymupdf
        except ImportError:
            return [{"page_number": 1, "text": "[PyMuPDF not available]", "success": False}]

        results = []
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            for page_num, page in enumerate(doc, start=1):
                if page_num > max_pages:
                    break
                # Render at 2x zoom for better OCR accuracy
                mat = pymupdf.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                text, success = self.extract_from_bytes(img_bytes, language=language)
                results.append({
                    "page_number": page_num,
                    "text": text,
                    "success": success,
                })
        except Exception as exc:
            logger.error(f"PDF OCR failed: {exc}")
            results = [{"page_number": 1, "text": f"[PDF OCR failed: {exc}]", "success": False}]

        return results

    @staticmethod
    def _preprocess(img):
        """
        Basic image preprocessing to improve OCR accuracy.
        Converts to grayscale and enhances contrast.
        """
        try:
            from PIL import ImageEnhance, ImageFilter
            # Convert to grayscale
            img = img.convert("L")
            # Sharpen slightly
            img = img.filter(ImageFilter.SHARPEN)
            # Boost contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            return img
        except Exception:
            return img


# Module-level singleton for reuse
_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
