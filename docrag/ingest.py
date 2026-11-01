"""Read PDF pages and split them into source-aware chunks."""

import logging
import os
import shutil
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def find_pdfs(paths: list[Path]) -> list[Path]:
    """Find unique PDFs below files or directories."""
    found = {}
    for path in map(Path, paths):
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"not a PDF file: {path}")
            candidates = [path]
        elif path.is_dir():
            candidates = (p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
        else:
            raise ValueError(f"not a file or directory: {path}")
        for candidate in candidates:
            resolved = candidate.resolve()
            found[str(resolved)] = resolved
    return sorted(found.values(), key=lambda p: str(p))


def load_pages(pdf: Path, source: str, ocr: str = "auto", ocr_language: str = "eng") -> list[Document]:
    """Extract non-empty pages from a PDF, optionally using Tesseract OCR."""
    if ocr not in {"auto", "off"}:
        raise ValueError("ocr must be 'auto' or 'off'")
    try:
        pages = []
        with pymupdf.open(pdf) as doc:
            if doc.is_encrypted:
                raise ValueError(f"encrypted PDF: {pdf}")
            missing_ocr_logged = False
            tesseract_available = shutil.which("tesseract") is not None
            for number, page in enumerate(doc, start=1):
                text = page.get_text()
                if not text.strip() and ocr == "auto":
                    if tesseract_available:
                        tessdata = pymupdf.get_tessdata()
                        if tessdata:
                            os.environ.setdefault("TESSDATA_PREFIX", tessdata)
                        textpage = page.get_textpage_ocr(language=ocr_language, full=True, dpi=300)
                        text = page.get_text(textpage=textpage)
                    elif not missing_ocr_logged:
                        language_packages = " ".join(
                            f"tesseract-ocr-{language}" for language in ocr_language.split("+") if language
                        )
                        logger.warning(
                            "Tesseract is not installed; install it with "
                            "`sudo apt install tesseract-ocr %s` to enable OCR for %s",
                            language_packages,
                            pdf,
                        )
                        missing_ocr_logged = True
                text = " ".join(text.split())
                if text:
                    pages.append(Document(page_content=text, metadata={"source": source, "page": number}))
        return pages
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not read PDF {pdf}: {exc}") from exc


def split(pages: list[Document], chunk_size: int, overlap: int | None = None) -> list[Document]:
    """Split pages while preserving source and page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5 if overlap is None else overlap,
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks
