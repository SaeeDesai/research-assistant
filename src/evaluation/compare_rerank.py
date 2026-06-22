"""
compare_rerank.py

Measures the effect of reranking on retrieval quality.
Scores top-1 precision both WITH and WITHOUT reranking on the
same test cases, so we can see the aggregate before/after.
"""

from src.evaluation.test_set import TEST_CASES
from src.retrieval.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.retrieval.reranker import Reranker


def top_source(results):
    """Get the source of the top retrieved chunk."""
    if not results:
        return None
    chunk = results[0][0]
    return chunk.metadata.get("source", "unknown")


def run_comparison():
    embedder = Embedder()
    store = VectorStore(embedder)
    store.load('vector_store')
    reranker = Reranker()

    # Only answerable cases have an expected source
    cases = [c for c in TEST_CASES if c.category != "offtopic"]

    baseline_hits = 0
    reranked_hits = 0

    print("\n" + "=" * 70)
    print("  RERANKING COMPARISON — top-1 source match")
    print("=" * 70)

    for case in cases:
        expected = case.expected_source

        # Baseline: bi-encoder only, top 5
        baseline = store.search(case.question, k=5)
        baseline_top = top_source(baseline)
        baseline_ok = baseline_top == expected
        baseline_hits += baseline_ok

        # Reranked: retrieve 15, rerank to 5
        reranked = store.search_with_rerank(
            case.question, reranker, initial_k=15, final_k=5
        )
        reranked_top = top_source(reranked)
        reranked_ok = reranked_top == expected
        reranked_hits += reranked_ok

        print(f"\n  Q: {case.question[:50]}")
        print(f"     expected:  {expected}")
        print(f"     baseline:  {baseline_top}  {'✓' if baseline_ok else '✗'}")
        print(f"     reranked:  {reranked_top}  {'✓' if reranked_ok else '✗'}")

    n = len(cases)
    print("\n" + "=" * 70)
    print(f"  Top-1 precision WITHOUT reranking: {baseline_hits}/{n} = {100*baseline_hits/n:.0f}%")
    print(f"  Top-1 precision WITH reranking:    {reranked_hits}/{n} = {100*reranked_hits/n:.0f}%")
    print("=" * 70)


if __name__ == '__main__':
    run_comparison()