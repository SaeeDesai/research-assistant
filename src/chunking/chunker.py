"""
chunker.py

Responsibility: Split Document objects into smaller chunks
while preserving meaning and metadata.

Why do we chunk?
LLMs have context window limits — we can't send 326,000 words
at once. Chunking lets us send only the relevant pieces.
The art is in choosing chunk size and overlap so we preserve
meaning without introducing noise.
"""

import re
import logging
from dataclasses import dataclass, field
from src.ingestion.document_parser import Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    A single chunk of text with metadata inherited from its
    parent Document plus chunk-specific location info.

    Why inherit metadata?
    When your RAG system retrieves this chunk three weeks from
    now, it needs to know: which paper did this come from?
    What position in the paper? That's source attribution —
    the difference between a useful answer and an unverifiable one.
    """
    content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.content[:80].replace('\n', ' ')
        return (
            f"Chunk("
            f"source={self.metadata.get('source', '?')}, "
            f"chunk={self.metadata.get('chunk_index', '?')}/"
            f"{self.metadata.get('total_chunks', '?')}, "
            f"words={self.metadata.get('word_count', '?')}, "
            f"preview='{preview}...')"
        )


class RecursiveChunker:
    """
    Splits documents into overlapping chunks using recursive
    character splitting.

    The core idea: try to split on meaningful boundaries first
    (paragraphs, then sentences, then words) before resorting
    to arbitrary character splits.

    Parameters:
        chunk_size:    Target size of each chunk in characters.
                       Why characters not words? Because our
                       embedding model has a token limit, and
                       tokens correlate more closely with
                       characters than words.

        chunk_overlap: How many characters to repeat between
                       consecutive chunks. Preserves context
                       at chunk boundaries.
    """

    # The separators we try, in order of preference
    # We try paragraph breaks first, then sentences, then words
    SEPARATORS = ['\n\n', '\n', '. ', '! ', '? ', ' ', '']

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(
            f"RecursiveChunker initialized — "
            f"chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using a priority list of separators.

        This is the heart of the chunker. Read it carefully.

        Args:
            text:       The text to split
            separators: List of separators to try, in priority order

        Returns:
            List of text chunks, all <= chunk_size characters
        """

        # Base case: text already fits in one chunk
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Base case: no more separators to try — force split by character
        if not separators:
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                # Move forward by chunk_size minus overlap
                # so consecutive chunks share overlap characters
                start += self.chunk_size - self.chunk_overlap
            return chunks

        # Try the highest-priority separator
        separator = separators[0]
        remaining_separators = separators[1:]

        # Split on this separator
        if separator == '':
            splits = list(text)
        else:
            splits = text.split(separator)

        # Now merge small splits back together until we hit chunk_size
        # This prevents tiny chunks (e.g. one sentence per chunk when
        # sentences are short)
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split) + len(separator)

            # If adding this split would exceed chunk_size:
            if current_length + split_length > self.chunk_size:

                if current_chunk:
                    # Save what we have so far
                    chunk_text = separator.join(current_chunk)

                    # If even this chunk is too big, recurse with
                    # lower-priority separators
                    if len(chunk_text) > self.chunk_size:
                        sub_chunks = self._split_text(
                            chunk_text,
                            remaining_separators
                        )
                        chunks.extend(sub_chunks)
                    else:
                        if chunk_text.strip():
                            chunks.append(chunk_text)

                    # Start new chunk with overlap
                    # Keep the last few splits to create overlap
                    overlap_text = self._get_overlap_text(
                        current_chunk,
                        separator
                    )
                    current_chunk = overlap_text
                    current_length = sum(
                        len(s) + len(separator)
                        for s in current_chunk
                    )

            current_chunk.append(split)
            current_length += split_length

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) > self.chunk_size:
                sub_chunks = self._split_text(
                    chunk_text,
                    remaining_separators
                )
                chunks.extend(sub_chunks)
            elif chunk_text.strip():
                chunks.append(chunk_text)

        return chunks

    def _get_overlap_text(
        self,
        splits: list[str],
        separator: str
    ) -> list[str]:
        """
        Get the tail of the current chunk to use as overlap
        for the next chunk.

        We walk backwards through the splits, accumulating
        text until we have at least chunk_overlap characters.
        """
        overlap_splits = []
        overlap_length = 0

        for split in reversed(splits):
            split_length = len(split) + len(separator)
            if overlap_length + split_length > self.chunk_overlap:
                break
            overlap_splits.insert(0, split)
            overlap_length += split_length

        return overlap_splits

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Split a single Document into a list of Chunks.

        Each Chunk inherits the parent Document's metadata
        and adds its own position information.

        Args:
            document: A parsed Document from document_parser.py

        Returns:
            List of Chunk objects
        """
        logger.info(
            f"Chunking {document.metadata.get('source', 'unknown')} "
            f"({document.metadata.get('word_count', 0):,} words)"
        )

        # Split the text
        text_chunks = self._split_text(document.content, self.SEPARATORS)

        if not text_chunks:
            logger.warning(
                f"No chunks produced from "
                f"{document.metadata.get('source', 'unknown')}"
            )
            return []

        # Wrap each text chunk in a Chunk object with metadata
        chunks = []
        for i, text in enumerate(text_chunks):
            chunk_metadata = {
                # Inherit everything from parent document
                **document.metadata,
                # Add chunk-specific fields
                "chunk_index":  i,
                "total_chunks": len(text_chunks),
                "word_count":   len(text.split()),
                "char_count":   len(text),
            }
            chunks.append(Chunk(content=text, metadata=chunk_metadata))

        logger.info(
            f"  → {len(chunks)} chunks, "
            f"avg {sum(len(c.content) for c in chunks) // len(chunks)} "
            f"chars/chunk"
        )

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Chunk all documents in a corpus.

        Args:
            documents: List of Document objects

        Returns:
            Flat list of all Chunk objects across all documents
        """
        all_chunks = []

        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)

        total_words = sum(c.metadata['word_count'] for c in all_chunks)

        logger.info(
            f"Chunking complete — "
            f"{len(all_chunks)} total chunks across "
            f"{len(documents)} documents, "
            f"{total_words:,} total words"
        )

        return all_chunks


# --- Test block ---
if __name__ == '__main__':
    from src.ingestion.document_parser import PDFParser
    from pathlib import Path

    parser = PDFParser()
    chunker = RecursiveChunker(chunk_size=1500, chunk_overlap=200)

    # ── Test 1: Single document ──────────────────────────────────
    print("=" * 60)
    print("TEST 1: Chunk the RAG paper")
    print("=" * 60)

    doc = parser.parse(Path('data/raw/RAG.pdf'))
    chunks = chunker.chunk_document(doc)

    print(f"\nDocument: {doc.metadata['word_count']:,} words")
    print(f"Chunks produced: {len(chunks)}")
    print(f"Avg chunk size: {sum(len(c.content) for c in chunks) // len(chunks)} chars")

    print(f"\nFirst chunk:")
    print("-" * 40)
    print(chunks[0].content)
    print("-" * 40)
    print(f"Metadata: {chunks[0].metadata}")

    print(f"\nSecond chunk (check overlap with first):")
    print("-" * 40)
    print(chunks[1].content[:300])
    print("-" * 40)

    # ── Test 2: Verify overlap is working ───────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: Verify overlap between consecutive chunks")
    print("=" * 60)

    last_words_chunk0 = chunks[0].content[-200:]
    first_words_chunk1 = chunks[1].content[:200:]

    print(f"\nLast 200 chars of chunk 0:")
    print(f"  '{last_words_chunk0}'")
    print(f"\nFirst 200 chars of chunk 1:")
    print(f"  '{first_words_chunk1}'")
    print(f"\nDo they share text? "
          f"{'YES ✓' if any(word in chunks[1].content for word in chunks[0].content[-100:].split()) else 'NO — check overlap logic'}")

    # ── Test 3: Full corpus ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3: Chunk all 20 documents")
    print("=" * 60)

    all_docs = parser.parse_all(Path('data/raw'))
    all_chunks = chunker.chunk_documents(all_docs)

    print(f"\nFull corpus summary:")
    print(f"  Documents:    {len(all_docs)}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Total words:  {sum(c.metadata['word_count'] for c in all_chunks):,}")

    print(f"\nPer-document chunk counts:")
    from collections import defaultdict
    chunks_per_doc = defaultdict(int)
    for chunk in all_chunks:
        chunks_per_doc[chunk.metadata['source']] += 1

    for source, count in sorted(chunks_per_doc.items()):
        print(f"  {source:<45} {count:>4} chunks")