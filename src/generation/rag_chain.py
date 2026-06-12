"""
rag_chain.py

Responsibility: Tie retrieval and generation together.
Takes a user question, retrieves relevant chunks, builds a
grounded prompt, and generates an answer using an LLM.

This is the heart of the RAG system — where everything
from Days 1-4 finally produces a real answer.
"""

import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq

from src.retrieval.vector_store import VectorStore
from src.chunking.chunker import Chunk

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    A complete RAG response: the answer plus the sources
    that were used to generate it.

    Why bundle answer + sources together?
    A RAG answer without sources is unverifiable. The whole
    advantage of RAG over a plain LLM is that you can cite
    where the answer came from. We return both so the user
    can check the answer against the original papers.
    """
    answer: str
    sources: list[Chunk]
    question: str

    def display(self) -> str:
        """Format the response for human-readable printing."""
        output = [
            f"\nQuestion: {self.question}",
            f"\nAnswer:\n{self.answer}",
            f"\nSources ({len(self.sources)} chunks):"
        ]
        # Show unique source documents
        seen = set()
        for chunk in self.sources:
            source = chunk.metadata.get('source', 'unknown')
            if source not in seen:
                seen.add(source)
                output.append(f"  • {source}")
        return "\n".join(output)


class RAGChain:
    """
    Retrieval-Augmented Generation chain.

    Combines a VectorStore (retrieval) with an LLM (generation)
    to answer questions grounded in your document corpus.
    """

    # The model we use for generation.
    # llama-3.3-70b-versatile is Groq's flagship — fast and capable.
    LLM_MODEL = 'llama-3.3-70b-versatile'

    # The system prompt defines the LLM's behavior.
    # This is the single most important piece of text in the
    # whole RAG system — it's our main defense against hallucination.
    SYSTEM_PROMPT = """You are a precise research assistant that answers questions about AI and machine learning papers.

Your rules:
1. Answer ONLY using the context provided below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say "I don't have enough information in the provided context to answer that." Do not guess or make up information.
3. Be concise and accurate. Cite specific details from the context when relevant.
4. If you reference a specific claim, mention which source it came from when possible.

You must stay grounded in the provided context at all times."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. "
                "Did you add it to your .env file?"
            )

        self.client = Groq(api_key=api_key)
        logger.info(f"RAGChain initialized — LLM: {self.LLM_MODEL}")

    def _build_context(self, chunks: list[Chunk]) -> str:
        """
        Format retrieved chunks into a context string for the prompt.

        Why number the chunks and label sources?
        It helps the LLM cite which chunk it used, and it helps
        us debug — when we read the LLM's answer, we can trace
        it back to specific context.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get('source', 'unknown')
            context_parts.append(
                f"[Context {i} — from {source}]\n{chunk.content}"
            )
        return "\n\n".join(context_parts)

    def answer(
        self,
        question: str,
        k: int = 5,
        temperature: float = 0.1
    ) -> RAGResponse:
        """
        Answer a question using retrieval-augmented generation.

        Args:
            question:    The user's question
            k:           How many chunks to retrieve for context
            temperature: LLM randomness (0 = deterministic,
                         1 = creative). We use 0.1 because we want
                         factual, grounded answers — not creative ones.

        Returns:
            RAGResponse with answer and sources
        """
        logger.info(f"Answering: '{question}'")

        # Step 1: Retrieve relevant chunks
        results = self.vector_store.search(question, k=k)
        retrieved_chunks = [chunk for chunk, score in results]

        if not retrieved_chunks:
            return RAGResponse(
                answer="No relevant context found to answer this question.",
                sources=[],
                question=question
            )

        # Step 2: Build the context string from retrieved chunks
        context = self._build_context(retrieved_chunks)

        # Step 3: Build the user message (context + question)
        user_message = f"""Context:
{context}

Question: {question}

Answer the question using only the context above."""

        # Step 4: Call the LLM
        logger.info(f"Sending prompt to LLM ({len(retrieved_chunks)} chunks of context)")

        response = self.client.chat.completions.create(
            model=self.LLM_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=1024
        )

        answer_text = response.choices[0].message.content

        logger.info("Answer generated")

        return RAGResponse(
            answer=answer_text,
            sources=retrieved_chunks,
            question=question
        )


# --- Test block ---
if __name__ == '__main__':
    from src.embeddings.embedder import Embedder

    # Load the pre-built vector store from disk
    print("Loading vector store...")
    embedder = Embedder()
    store = VectorStore(embedder)
    store.load('vector_store')

    # Build the RAG chain
    rag = RAGChain(store)

    # Test questions — mix of answerable and unanswerable
    test_questions = [
        # Should answer well — directly in the papers
        "What is LoRA and how does it reduce the number of trainable parameters?",
        "How does the attention mechanism work in transformers?",
        "What is the main idea behind retrieval-augmented generation?",

        # Should answer — tests synthesis across context
        "What datasets were used to train the original transformer?",

        # Should refuse — not in the papers
        "What is the best pizza topping?",
    ]

    print("\n" + "=" * 60)
    print("RAG CHAIN — END TO END TEST")
    print("=" * 60)

    for question in test_questions:
        response = rag.answer(question, k=5)
        print(response.display())
        print("\n" + "-" * 60)