"""
document_parser.py

Responsibility: Extract clean text from PDF files and return
structured Document objects with text + metadata.

Why do we need this?
PDFs store rendering instructions, not clean text. This module
bridges the gap between raw binary PDF files and the clean text
strings that our chunker and embedder expect.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

import pypdf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    A parsed document with cleaned text and metadata.

    Why a dataclass?
    A dataclass is like a regular class but Python auto-generates
    __init__, __repr__, and __eq__ for you. It's cleaner than a
    plain dictionary because:
    - You get type hints (content: str)
    - You can't accidentally misspell a key
    - It's self-documenting
    - IDEs give you autocomplete

    Think of it as a named container for your data.
    """
    content: str                    # The cleaned text
    metadata: dict = field(default_factory=dict)  # Source info

    def __repr__(self):
        preview = self.content[:100].replace('\n', ' ')
        return (
            f"Document("
            f"source={self.metadata.get('source', 'unknown')}, "
            f"pages={self.metadata.get('total_pages', '?')}, "
            f"chars={len(self.content)}, "
            f"preview='{preview}...')"
        )


class PDFParser:
    """
    Extracts and cleans text from PDF files.

    Why a class and not just functions?
    The cleaning pipeline (the list of regex patterns and rules)
    is shared state across every document we parse. A class
    initializes that state once. Functions would rebuild it
    on every call.
    """

    def __init__(self):
        logger.info("PDFParser initialized")

    def _extract_raw_text(self, pdf_path: Path) -> tuple[list[str], int]:
        """
        Extract raw text from each page of a PDF.

        Returns:
            - List of raw text strings, one per page
            - Total page count

        Why page by page?
        Keeping pages separate lets us attach page numbers to
        metadata. If we concatenate everything first, we lose
        that information forever.
        """
        pages_text = []

        with open(pdf_path, 'rb') as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            total_pages = len(reader.pages)

            logger.info(
                f"Extracting text from {pdf_path.name} "
                f"({total_pages} pages)"
            )

            for page_num, page in enumerate(reader.pages):
                try:
                    raw_text = page.extract_text()

                    # Some pages extract as None (e.g. image-only pages)
                    if raw_text:
                        pages_text.append(raw_text)
                    else:
                        logger.warning(
                            f"Page {page_num + 1} of {pdf_path.name} "
                            f"returned no text — possibly an image page"
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to extract page {page_num + 1} "
                        f"of {pdf_path.name}: {e}"
                    )

        return pages_text, total_pages

    def _clean_text(self, raw_text: str) -> str:
        """
        Clean raw extracted PDF text.

        This is where the real work happens. PDFs are messy.
        We apply a series of cleaning steps in a specific order —
        order matters because some steps depend on previous ones.

        Each step is explained with WHY, not just what.
        """

        text = raw_text

        # Step 1: Fix hyphenated line breaks
        # PDFs split long words across lines with a hyphen: "knowl-\nedge"
        # We rejoin them: "knowledge"
        # Why regex? Because we only want to join hyphen+newline,
        # not all hyphens (we want to keep "state-of-the-art")
        text = re.sub(r'-\n', '', text)

        # Step 2: Remove lone newlines within paragraphs
        # PDFs insert newlines at line wrap points, not paragraph boundaries
        # "the cat\nsat on\nthe mat" → "the cat sat on the mat"
        # We preserve double newlines (actual paragraph breaks)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # Step 3: Normalize multiple spaces to single space
        # After step 2 we may have "word  word" with double spaces
        text = re.sub(r' +', ' ', text)

        # Step 4: Normalize multiple newlines to maximum two
        # We want to preserve paragraph breaks (double newline)
        # but not giant gaps (5+ newlines from headers/footers)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Step 5: Remove page numbers and common header/footer patterns
        # Matches things like "- 5 -" or "Page 5" or just "5" on its own line
        text = re.sub(r'\n\s*-?\s*\d+\s*-?\s*\n', '\n', text)
        text = re.sub(r'\nPage \d+\n', '\n', text, flags=re.IGNORECASE)

        # Step 6: Remove non-printable characters
        # PDFs sometimes contain control characters that cause
        # encoding issues downstream
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Step 7: Strip leading/trailing whitespace
        text = text.strip()

        return text

    def parse(self, pdf_path: Path) -> Document:
        """
        Parse a single PDF file into a Document object.

        This is the main public method — the one you call from
        outside this class. The private methods above (_extract,
        _clean) are implementation details.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Document with cleaned text and metadata
        """
        pdf_path = Path(pdf_path)  # Handle string input gracefully

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"Expected a PDF file, got: {pdf_path.suffix}")

        # Step 1: Extract raw text page by page
        pages_text, total_pages = self._extract_raw_text(pdf_path)

        if not pages_text:
            raise ValueError(f"No text could be extracted from {pdf_path.name}")

        # Step 2: Join all pages with a separator
        # We use double newline between pages to preserve
        # document structure
        full_raw_text = '\n\n'.join(pages_text)

        # Step 3: Clean the combined text
        cleaned_text = self._clean_text(full_raw_text)

        # Step 4: Build metadata
        file_size_kb = pdf_path.stat().st_size / 1024
        metadata = {
            "source":       pdf_path.name,
            "total_pages":  total_pages,
            "file_size_kb": round(file_size_kb, 1),
            "char_count":   len(cleaned_text),
            "word_count":   len(cleaned_text.split()),
        }

        logger.info(
            f"Parsed {pdf_path.name}: "
            f"{total_pages} pages, "
            f"{metadata['word_count']:,} words, "
            f"{metadata['char_count']:,} chars"
        )

        return Document(content=cleaned_text, metadata=metadata)

    def parse_all(self, pdf_dir: Path) -> list[Document]:
        """
        Parse all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDF files

        Returns:
            List of Document objects, one per successfully parsed PDF
        """
        pdf_dir = Path(pdf_dir)
        pdf_files = list(pdf_dir.glob('*.pdf'))

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return []

        logger.info(f"Parsing {len(pdf_files)} PDF files from {pdf_dir}")

        documents = []
        failed = 0

        for pdf_path in sorted(pdf_files):
            try:
                doc = self.parse(pdf_path)
                documents.append(doc)

            except Exception as e:
                logger.error(f"Failed to parse {pdf_path.name}: {e}")
                failed += 1

        logger.info(
            f"Successfully parsed {len(documents)} documents "
            f"({failed} failed)"
        )

        return documents


# --- Test block ---
if __name__ == '__main__':
    parser = PDFParser()

    # Test 1: Parse a single document
    print("=" * 60)
    print("TEST 1: Parse single document (RAG paper)")
    print("=" * 60)

    rag_path = Path('data/raw/RAG.pdf')
    doc = parser.parse(rag_path)

    print(f"\nDocument: {doc}")
    print(f"\nFirst 500 characters of cleaned text:")
    print("-" * 40)
    print(doc.content[:500])
    print("-" * 40)
    print(f"\nMetadata: {doc.metadata}")

    # Test 2: Parse all documents
    print("\n" + "=" * 60)
    print("TEST 2: Parse all 20 documents")
    print("=" * 60)

    all_docs = parser.parse_all(Path('data/raw'))

    print(f"\nResults:")
    print(f"  Total documents parsed: {len(all_docs)}")
    print(f"\nPer-document summary:")
    for d in all_docs:
        print(
            f"  {d.metadata['source']:<45} "
            f"{d.metadata['total_pages']:>3} pages  "
            f"{d.metadata['word_count']:>7,} words"
        )

    total_words = sum(d.metadata['word_count'] for d in all_docs)
    total_chars = sum(d.metadata['char_count'] for d in all_docs)
    print(f"\nTotal corpus:")
    print(f"  Words: {total_words:,}")
    print(f"  Characters: {total_chars:,}")