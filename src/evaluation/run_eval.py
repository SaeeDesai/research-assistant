"""
run_eval.py

The top-level evaluation runner. Ties together the harness and
the scorers, computes aggregate metrics per category and overall,
and prints a report card.

Run with: python -m src.evaluation.run_eval
"""

import time
import logging
from collections import defaultdict

from src.evaluation.harness import run_test_cases
from src.evaluation.scorers import (
    score_retrieval,
    score_faithfulness,
    score_relevance,
)
from src.retrieval.vector_store import VectorStore
from src.embeddings.embedder import Embedder

logging.basicConfig(level=logging.WARNING)  # quiet the noise for a clean report


def run_full_evaluation():
    """
    Run the complete evaluation pipeline and print a report card.

    Steps:
    1. Run all test cases through the system (harness)
    2. Score retrieval (both top-1 and recall)
    3. Score answer quality (faithfulness + relevance) via LLM judge
    4. Aggregate into per-category and overall metrics
    """
    print("Running evaluation — this makes several LLM calls, please wait...\n")

    # We need the vector store to rebuild context for faithfulness scoring
    embedder = Embedder()
    store = VectorStore(embedder)
    store.load('vector_store')

    # Step 1: run the harness
    results = run_test_cases()

    # Step 2 & 3: score each answerable result
    # We collect scores into lists so we can average them
    retrieval_top1 = []
    retrieval_recall = []
    faithfulness = []
    relevance = []

    # Per-category tracking
    by_category = defaultdict(lambda: {
        "top1": [], "recall": [], "faith": [], "rel": []
    })

    for r in results:
        if r.category == "offtopic":
            continue  # scored separately / not part of RAG metrics

        # Retrieval scoring (objective, no LLM)
        top1, recall = score_retrieval(r)
        retrieval_top1.append(top1)
        retrieval_recall.append(recall)

        # Rebuild the context shown to the generator (for faithfulness)
        search_results = store.search(r.question, k=5)
        context = "\n\n".join(c.content for c, s in search_results)

        # Answer quality scoring (LLM-as-judge)
        faith = score_faithfulness(r, context)
        rel = score_relevance(r)
        faithfulness.append(faith)
        relevance.append(rel)

        # Store scores back on the result object
        r.retrieval_correct = top1
        r.faithfulness_score = faith
        r.relevance_score = rel

        # Track per category
        by_category[r.category]["top1"].append(top1)
        by_category[r.category]["recall"].append(recall)
        by_category[r.category]["faith"].append(faith)
        by_category[r.category]["rel"].append(rel)

        # Small delay to be gentle on the API rate limit
        time.sleep(1)

    # Step 4: compute and print aggregates
    _print_report(
        results,
        retrieval_top1, retrieval_recall,
        faithfulness, relevance,
        by_category
    )


def _pct(bool_list) -> str:
    """Format a list of booleans as a percentage."""
    if not bool_list:
        return "n/a"
    return f"{100 * sum(bool_list) / len(bool_list):.0f}%"


def _avg(num_list) -> str:
    """Format a list of numbers as an average out of 5."""
    if not num_list:
        return "n/a"
    return f"{sum(num_list) / len(num_list):.2f}/5"


def _print_report(results, top1, recall, faith, rel, by_category):
    """Print the evaluation report card."""
    n = len(top1)

    print("\n" + "=" * 60)
    print("  EVALUATION REPORT CARD")
    print("=" * 60)
    print(f"\n  Answerable test cases evaluated: {n}")

    print("\n  RETRIEVAL")
    print(f"    Top-1 precision:  {_pct(top1)}   (right source ranked #1)")
    print(f"    Recall@5:         {_pct(recall)}   (right source in top 5)")

    print("\n  ANSWER QUALITY (LLM-as-judge, 1-5)")
    print(f"    Faithfulness:     {_avg(faith)}   (grounded in context)")
    print(f"    Relevance:        {_avg(rel)}   (answers the question)")

    print("\n  BY CATEGORY")
    for cat, scores in by_category.items():
        print(f"    {cat}:")
        print(f"      top-1: {_pct(scores['top1'])}  "
              f"recall: {_pct(scores['recall'])}  "
              f"faith: {_avg(scores['faith'])}  "
              f"rel: {_avg(scores['rel'])}")

    print("\n" + "=" * 60)
    print("  PER-QUESTION DETAIL")
    print("=" * 60)
    for r in results:
        if r.category == "offtopic":
            continue
        print(f"\n  {r.question[:55]}")
        print(f"    retrieval: {'✓' if r.retrieval_correct else '✗'}  "
              f"faith: {r.faithfulness_score}/5  "
              f"rel: {r.relevance_score}/5")


if __name__ == '__main__':
    run_full_evaluation()