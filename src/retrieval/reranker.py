"""
reranker.py

A cross-encoder reranker that re-scores retrieved chunks for
more precise relevance ranking.

Why? Bi-encoder retrieval (our FAISS search) has high recall
but imperfect top-1 ranking — the right chunk is retrieved but
sometimes ranked below a lookalike. A cross-encoder reads the
query and chunk TOGETHER and produces a precise relevance score,
fixing the ranking without needing to re-retrieve.

Two-stage design: retrieve broad (bi-encoder), rerank narrow
(cross-encoder).
"""

import logging
from sentence_transformers import CrossEncoder
from src.chunking.chunker import Chunk

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)


class Reranker:
    """
    Re-scores (query, chunk) pairs with a cross-encoder and
    returns chunks sorted by precise relevance.

    Loaded once, reused across queries — same pattern as the
    Embedder. The model is small (~80MB) but loading still has
    overhead we don't want to repeat per query.
    """

    MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    def __init__(self):
        logger.info(f"Loading cross-encoder reranker: {self.MODEL_NAME}")
        self.model = CrossEncoder(self.MODEL_NAME)
        logger.info("Reranker loaded")

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """
        Re-score chunks against the query and return the top_k.

        Args:
            query:  the user's question
            chunks: candidate chunks from the first-stage retrieval
                    (more than top_k — e.g. 15 candidates)
            top_k:  how many chunks to return after reranking

        Returns:
            List of (Chunk, rerank_score) tuples, sorted by
            relevance (most relevant first), length top_k
        """
        if not chunks:
            return []

        # Build (query, chunk_text) pairs for the cross-encoder.
        # Unlike the bi-encoder, the cross-encoder needs the query
        # paired with EACH chunk — it scores them together.
        pairs = [(query, chunk.content) for chunk in chunks]

        # Score all pairs. Returns one relevance score per pair.
        scores = self.model.predict(pairs)

        # Pair each chunk with its new score
        scored_chunks = list(zip(chunks, scores))

        # Sort by score, highest relevance first
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return only the top_k
        return [(chunk, float(score)) for chunk, score in scored_chunks[:top_k]]


# --- Test block ---
if __name__ == '__main__':
    from src.retrieval.vector_store import VectorStore
    from src.embeddings.embedder import Embedder

    embedder = Embedder()
    store = VectorStore(embedder)
    store.load('vector_store')
    reranker = Reranker()

    # The RAG question — our known top-1 failure case
    query = "What is the main idea behind retrieval-augmented generation?"

    print("=" * 65)
    print("RERANKING TEST — the RAG-vs-RAGAS failure case")
    print("=" * 65)

    # Stage 1: retrieve broad (15 candidates instead of 5)
    initial = store.search(query, k=15)
    initial_chunks = [chunk for chunk, score in initial]

    print(f"\nBEFORE reranking — top 5 by bi-encoder:")
    for i, (chunk, score) in enumerate(initial[:5], 1):
        print(f"  {i}. [{score:.3f}] {chunk.metadata['source']}")

    # Stage 2: rerank narrow (cross-encoder picks true top 5)
    reranked = reranker.rerank(query, initial_chunks, top_k=5)

    print(f"\nAFTER reranking — top 5 by cross-encoder:")
    for i, (chunk, score) in enumerate(reranked, 1):
        print(f"  {i}. [{score:.3f}] {chunk.metadata['source']}")