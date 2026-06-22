"""
harness.py

Runs each test case through the system and collects what the agent produced
along with the ground-truth references. The output of the harness is the raw material
that the scorers then evaluate.
"""

import logging
from dataclasses import dataclass, field

from src.evaluation.test_set import TestCase, TEST_CASES
from src.retrieval.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.generation.rag_chain import RAGChain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EvalResult:
    """
    Everything we collected for one test case — what the system
    produced next to what it should have produced.

    This bundles the question, the ground-truth references, and
    the agent's actual output so the scorers have everything they
    need in one place.
    """
    question: str
    category: str

    reference_answer: str
    expected_source: str

    generated_answer: str
    retrieved_sources: list = field(default_factory=list)

    retrieval_correct: bool = False
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0

def run_test_cases(test_cases: list[TestCase] = None) -> list[EvalResult]:
    """
    Run each test case through the retrieval + generation system
    and collect the results.

    For each case we capture:
    - the sources the retriever actually returned
    - the answer the system generated

    We test the RAG core (retriever + generator) directly rather
    than the full agent, to measure retrieval and answer quality
    in isolation from the routing/web-search layer.

    Returns a list of EvalResult, one per test case, with the
    'produced' fields filled in but scores not yet computed.
    """
    if test_cases is None:
        test_cases = TEST_CASES

    # Build the RAG system once
    embedder = Embedder()
    store = VectorStore(embedder)
    store.load('vector_store')
    rag = RAGChain(store)

    results = []

    for i, case in enumerate(test_cases, 1):
        logger.info(f"Running test case {i}/{len(test_cases)}: '{case.question[:50]}...'")

        # Off-topic cases: we don't run RAG, we just note the
        # expectation. The agent's refusal behavior is tested
        # separately; here we record the case for completeness.
        if case.category == "offtopic":
            results.append(EvalResult(
                question=case.question,
                category=case.category,
                reference_answer=case.reference_answer,
                expected_source=case.expected_source,
                generated_answer="(off-topic — handled by agent refusal)",
                retrieved_sources=[],
            ))
            continue

        # Run the RAG system: retrieve + generate
        response = rag.answer(case.question, k=5)

        # Extract the unique sources that were retrieved
        retrieved_sources = []
        for chunk in response.sources:
            src = chunk.metadata.get("source", "unknown")
            if src not in retrieved_sources:
                retrieved_sources.append(src)

        # Bundle everything into an EvalResult
        results.append(EvalResult(
            question=case.question,
            category=case.category,
            reference_answer=case.reference_answer,
            expected_source=case.expected_source,
            generated_answer=response.answer,
            retrieved_sources=retrieved_sources,
        ))

    logger.info(f"Completed {len(results)} test cases")
    return results


# --- Test block ---
if __name__ == '__main__':
    results = run_test_cases()

    print("\n" + "=" * 65)
    print("HARNESS RESULTS (raw — not yet scored)")
    print("=" * 65)

    for r in results:
        print(f"\nQ: {r.question}")
        print(f"  category:          {r.category}")
        print(f"  expected source:   {r.expected_source}")
        print(f"  retrieved sources: {r.retrieved_sources}")
        print(f"  answer preview:    {r.generated_answer[:150]}")