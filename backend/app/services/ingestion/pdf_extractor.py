from pathlib import Path
from typing import Dict, List

import pymupdf


class PDFExtractor:

    def extract_text(self, file_path: str) -> List[Dict]:
        pdf_path = Path(file_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {file_path}"
            )

        if not pdf_path.is_file():
            raise ValueError("The provided path is not a file.")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        pages = []

        try:
            with pymupdf.open(pdf_path) as document:

                if document.is_encrypted:
                    raise ValueError(
                        "Encrypted PDFs are not supported."
                    )

                for page_number, page in enumerate(document, start=1):
                    text = page.get_text("text").strip()

                    pages.append({
                        "page_number": page_number,
                        "text": text,
                        "has_text": bool(text)
                    })

        except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
            raise ValueError(
                f"Unable to read PDF: {error}"
            ) from error

        return pages