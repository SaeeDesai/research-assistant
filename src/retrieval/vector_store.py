"""
vector_store.py

Responsibility: Store chunk embeddings in a FAISS index
and provide fast similarity search.

Why FAISS?
- Developed by Facebook AI Research (same team as the FAISS paper
  in your corpus)
- Industry standard for vector similarity search
- Handles everything from 1K to 1B vectors
- Free, runs locally, no API needed
"""

import logging
import numpy as np
import faiss
import pickle
from pathlib import Path
from src.chunking.chunker import Chunk
from src.embeddings.embedder import Embedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Stores chunk embeddings in FAISS and enables semantic search.

    Two responsibilities:
    1. BUILD: take chunks + embedder, create searchable index
    2. SEARCH: take a query string, return top-k relevant chunks

    Why store chunks separately from the index?
    FAISS stores vectors (numbers) but not the original text.
    When FAISS says "vector 42 is most similar to your query",
    you need a way to look up "what was chunk 42's text?"
    We keep a parallel list of Chunk objects for this purpose.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index = None        # FAISS index (vectors only)
        self.chunks = []         # Parallel list (text + metadata)
        self.embedding_dim = embedder.embedding_dim

        logger.info(
            f"VectorStore initialized — "
            f"embedding_dim={self.embedding_dim}"
        )

    def build(self, chunks: list[Chunk]) -> None:
        """
        Build the FAISS index from a list of chunks.

        Steps:
        1. Embed all chunks → numpy array of shape (N, 384)
        2. Create FAISS index
        3. Add all vectors to the index
        4. Store chunks for later lookup

        Args:
            chunks: List of Chunk objects to index
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        logger.info(f"Building FAISS index for {len(chunks)} chunks...")

        # Step 1: Embed all chunks
        embeddings = self.embedder.embed_chunks(chunks)

        # Step 2: Create FAISS index
        # IndexFlatL2 = exact search using L2 (Euclidean) distance
        # "Flat" means no compression — stores full vectors
        # For normalized vectors, L2 distance and cosine similarity
        # produce the same ranking (just different scales)
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Step 3: Add vectors to the index
        # FAISS expects float32 — ensure correct dtype
        embeddings_f32 = embeddings.astype(np.float32)
        self.index.add(embeddings_f32)

        # Step 4: Store chunks for lookup
        self.chunks = chunks

        logger.info(
            f"FAISS index built — "
            f"{self.index.ntotal} vectors indexed"
        )

    def search(
        self,
        query: str,
        k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """
        Find the k most relevant chunks for a query.

        This is the core function your RAG system calls
        every time a user asks a question.

        Args:
            query: The user's question as a plain string
            k:     How many chunks to return (default 5)

        Returns:
            List of (Chunk, similarity_score) tuples,
            sorted by relevance (most relevant first)
        """
        if self.index is None:
            raise RuntimeError(
                "Index not built yet. Call build() first."
            )

        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty.")

        # Step 1: Embed the query
        # CRITICAL: use the same embedder that embedded the chunks
        query_embedding = self.embedder.embed_text(query)
        query_embedding_f32 = query_embedding.astype(np.float32)

        # FAISS expects shape (1, embedding_dim) for a single query
        query_vector = query_embedding_f32.reshape(1, -1)

        # Step 2: Search the index
        # Returns distances and indices of k nearest neighbors
        # distances shape: (1, k)
        # indices shape:   (1, k)
        distances, indices = self.index.search(query_vector, k)

        # Step 3: Convert to (Chunk, score) pairs
        # Lower L2 distance = more similar
        # We convert to similarity score: higher = better
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 for empty slots
                continue

            chunk = self.chunks[idx]
            # Convert L2 distance to similarity score
            # Score of 1.0 = identical, 0.0 = very different
            similarity = float(1 / (1 + distance))

            results.append((chunk, similarity))

        return results

    def save(self, save_dir: str = 'vector_store') -> None:
        """
        Save the FAISS index and chunks to disk.

        Why save?
        Building the index takes ~30 seconds for 1,953 chunks.
        You don't want to rebuild it every time you run the app.
        Save once, load instantly on subsequent runs.

        Saves two files:
        - vector_store/index.faiss  (the FAISS index)
        - vector_store/chunks.pkl   (the chunk objects)
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Save FAISS index
        index_path = save_path / 'index.faiss'
        faiss.write_index(self.index, str(index_path))

        # Save chunks using pickle
        chunks_path = save_path / 'chunks.pkl'
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)

        logger.info(
            f"Saved index ({self.index.ntotal} vectors) "
            f"and {len(self.chunks)} chunks to {save_dir}/"
        )

    def load(self, save_dir: str = 'vector_store') -> None:
        """
        Load a previously saved FAISS index and chunks.

        Call this instead of build() when you already have
        a saved index — much faster than rebuilding.
        """
        save_path = Path(save_dir)
        index_path = save_path / 'index.faiss'
        chunks_path = save_path / 'chunks.pkl'

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No saved index found in {save_dir}/. "
                f"Call build() first."
            )

        self.index = faiss.read_index(str(index_path))

        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)

        logger.info(
            f"Loaded index ({self.index.ntotal} vectors) "
            f"and {len(self.chunks)} chunks from {save_dir}/"
        )


# --- Test block ---
if __name__ == '__main__':
    from src.ingestion.document_parser import PDFParser
    from src.chunking.chunker import RecursiveChunker
    from pathlib import Path

    # ── Setup ────────────────────────────────────────────────────
    parser = PDFParser()
    chunker = RecursiveChunker()
    embedder = Embedder()

    # ── Test 1: Similarity sanity check ─────────────────────────
    embedder.verify_similarity()

    # ── Test 2: Build index on small corpus first ────────────────
    print("\n" + "=" * 55)
    print("TEST 2: Build index on 3 papers (quick test)")
    print("=" * 55)

    test_papers = [
        'data/raw/RAG.pdf',
        'data/raw/attention_is_all_you_need.pdf',
        'data/raw/lora.pdf',
    ]

    test_docs = [parser.parse(Path(p)) for p in test_papers]
    test_chunks = chunker.chunk_documents(test_docs)

    print(f"\nTest corpus: {len(test_chunks)} chunks from 3 papers")

    store = VectorStore(embedder)
    store.build(test_chunks)

    # ── Test 3: Search ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 3: Semantic search")
    print("=" * 55)

    queries = [
        "How does the attention mechanism work?",
        "What is LoRA and how does it reduce parameters?",
        "How does RAG combine retrieval with generation?",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)

        results = store.search(query, k=3)

        for i, (chunk, score) in enumerate(results):
            print(f"\n  Result {i+1} (score: {score:.4f})")
            print(f"  Source: {chunk.metadata['source']}")
            print(f"  Preview: {chunk.content[:200].replace(chr(10), ' ')}")

    # ── Test 4: Save and reload ──────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 4: Save and reload index")
    print("=" * 55)

    store.save('vector_store_test')
    print("Saved.")

    store2 = VectorStore(embedder)
    store2.load('vector_store_test')
    print(f"Reloaded — {store2.index.ntotal} vectors, "
          f"{len(store2.chunks)} chunks")

    # Verify search still works after reload
    results = store2.search("multi-head attention", k=2)
    print(f"\nSearch after reload works: {len(results)} results ✓")

    # ── Test 5: Full corpus ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("TEST 5: Build full corpus index (all 20 papers)")
    print("=" * 55)
    print("This will take 30-60 seconds...")

    all_docs = parser.parse_all(Path('data/raw'))
    all_chunks = chunker.chunk_documents(all_docs)

    full_store = VectorStore(embedder)
    full_store.build(all_chunks)
    full_store.save('vector_store')

    print(f"\nFull index built and saved.")
    print(f"  Vectors: {full_store.index.ntotal}")
    print(f"  Chunks:  {len(full_store.chunks)}")

    # Final search on full corpus
    print("\n" + "=" * 55)
    print("FINAL TEST: Search full corpus")
    print("=" * 55)

    final_queries = [
        "What datasets were used to evaluate RAG?",
        "How does LoRA work during inference?",
        "What is constitutional AI?",
        "How do transformers handle long sequences?",
    ]

    for query in final_queries:
        print(f"\nQuery: '{query}'")
        results = full_store.search(query, k=2)
        for i, (chunk, score) in enumerate(results):
            print(
                f"  [{score:.3f}] {chunk.metadata['source']} — "
                f"{chunk.content[:120].replace(chr(10), ' ')}..."
            )