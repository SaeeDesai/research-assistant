"""
embedder.py

Responsibility: Turn text chunks into embedding vectors
using sentence-transformers.

Why sentence-transformers?
- Free, runs locally, no API costs
- all-MiniLM-L6-v2 is fast and high quality
- Same model for chunks and queries = same vector space
- Industry standard for semantic search tasks
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from src.chunking.chunker import Chunk

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)


class Embedder:
    """
    Converts text chunks into dense vector embeddings.

    Why a class?
    Loading the embedding model takes 2-3 seconds and uses
    ~90MB of memory. We load it once in __init__ and reuse
    it for every embedding call. If this were a function,
    you'd reload the model on every call — wasteful.
    """

    # We hardcode the model name as a class constant.
    # This makes it visible and easy to change in one place.
    # If you ever switch models, change it here and everything
    # downstream updates automatically.
    MODEL_NAME = 'all-MiniLM-L6-v2'

    def __init__(self):
        logger.info(f"Loading embedding model: {self.MODEL_NAME}")
        logger.info("This takes 5-10 seconds on first run...")

        # SentenceTransformer downloads the model on first use
        # and caches it locally (~90MB). Subsequent runs are instant.
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Model loaded — embedding dimension: {self.embedding_dim}"
        )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single piece of text.

        Used for embedding queries at search time.
        Must use the same model as embed_chunks() —
        they need to live in the same vector space.

        Args:
            text: Any string of text

        Returns:
            1D numpy array of shape (embedding_dim,)
            i.e. (384,) for all-MiniLM-L6-v2
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True  # Normalize to unit length
                                       # so cosine similarity = dot product
                                       # (faster computation)
        )
        return embedding

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed a list of chunks in batches.

        Why batches?
        Embedding one chunk at a time is slow — the model
        has overhead per call. Batching sends multiple chunks
        through the model at once, using GPU/CPU parallelism.
        batch_size=32 is a good default for CPU inference.

        Args:
            chunks:        List of Chunk objects to embed
            batch_size:    How many chunks to process at once
            show_progress: Whether to show a progress bar

        Returns:
            2D numpy array of shape (num_chunks, embedding_dim)
            i.e. (1953, 384) for your full corpus
        """
        if not chunks:
            raise ValueError("Cannot embed empty chunk list")

        logger.info(
            f"Embedding {len(chunks)} chunks "
            f"(batch_size={batch_size})..."
        )

        # Extract just the text content from each Chunk object
        texts = [chunk.content for chunk in chunks]

        # Embed all texts in batches
        # show_progress_bar gives you a tqdm progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )

        logger.info(
            f"Embedding complete — "
            f"shape: {embeddings.shape}, "
            f"dtype: {embeddings.dtype}"
        )

        return embeddings

    def verify_similarity(self):
        """
        Quick sanity check — embed pairs of similar and
        different sentences and verify the similarity scores
        make intuitive sense.

        This is the test that makes embeddings 'click'.
        Run this once and you'll understand what embeddings
        actually do.
        """
        print("\n" + "=" * 55)
        print("EMBEDDING SIMILARITY SANITY CHECK")
        print("=" * 55)

        pairs = [
            (
                "How does the attention mechanism work in transformers?",
                "Explain the self-attention operation in neural networks.",
                "Similar — both about attention"
            ),
            (
                "What is the learning rate used during training?",
                "How do you fine-tune a language model?",
                "Related — both about training"
            ),
            (
                "The cat sat on the mat.",
                "Retrieval augmented generation combines parametric memory.",
                "Different — completely unrelated topics"
            ),
            (
                "LoRA reduces trainable parameters using low-rank matrices.",
                "LoRA is a parameter-efficient fine-tuning method.",
                "Very similar — same concept, different words"
            ),
        ]

        for text_a, text_b, label in pairs:
            vec_a = self.embed_text(text_a)
            vec_b = self.embed_text(text_b)

            # Cosine similarity = dot product when vectors are normalized
            similarity = float(np.dot(vec_a, vec_b))

            print(f"\n{label}")
            print(f"  A: '{text_a[:60]}'")
            print(f"  B: '{text_b[:60]}'")
            print(f"  Similarity: {similarity:.4f}")

        print("\n" + "=" * 55)
        print("Expected pattern:")
        print("  Very similar  → score close to 1.0")
        print("  Related       → score 0.5 - 0.8")
        print("  Different     → score close to 0.0 or negative")
        print("=" * 55)