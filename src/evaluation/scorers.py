"""
scorers.py

Scoring functions for evaluation. The harness gathers what the
system produced; the scorers judge it against the ground truth.

Two kinds of scoring:
1. Retrieval — objective comparison (did we retrieve the right source?)
2. Answer quality — LLM-as-judge (faithfulness + relevance)
"""

import logging
import os
import re
from dotenv import load_dotenv
from groq import Groq
from src.evaluation.harness import EvalResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

#---JUDGE CLIENT SETUP---
load_dotenv()
_judge_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
_JUDGE_MODEL = 'llama-3.3-70b-versatile'

def score_retrieval(result: EvalResult) -> tuple[bool, bool]:
    """
    Score retrieval for one result, two ways:

    - top1_correct (strict): is the expected source the FIRST
      retrieved source? Measures ranking quality.
    - recall_hit (lenient): does the expected source appear
      ANYWHERE in the retrieved sources? Measures whether the
      right document was retrieved at all.

    The gap between these two metrics is diagnostic: high recall
    but low top-1 means a ranking problem (fix with reranking);
    low recall means a retrieval problem (fix with better
    embeddings / query expansion).

    Off-topic cases are excluded — there's no expected source
    to retrieve.

    Returns:
        (top1_correct, recall_hit)
    """
    # Off-topic cases have no retrieval to score
    if result.category == "offtopic":
        return (False, False)

    expected = result.expected_source
    retrieved = result.retrieved_sources

    if not retrieved:
        return (False, False)

    # Option B: expected source is ranked first
    top1_correct = retrieved[0] == expected

    # Option A: expected source appears anywhere in the list
    recall_hit = expected in retrieved

    return (top1_correct, recall_hit)

def _extract_score(text: str) -> float:
    """
    Pull a 1-5 score out of the judge's response

    The judge is prompted to answer with 'SCORE:N', but LLMs are imperfect, 
    so we defensively search for the first number 1-5 in responses. Default to 0.0 if none found 
    """
    match = re.search(r'SCORE:\s*([1-5])', text)
    if match:
        return float(match.group(1))
    # Fallback: any digit 1-5 in the text
    match = re.search(r'\b([1-5])\b', text)
    if match:
        return float(match.group(1))
    return 0.0
    

def score_faithfulness(result: EvalResult, retrieved_context: str) -> float:
    """
    Judge whether the answer is supported by the retrieved context.

    Score 1-5:
        5 = fully grounded, every claim supported by context
        1 = mostly fabricated, claims not in context

    Args:
        result:            the EvalResult (has question + generated answer)
        retrieved_context: the actual chunks shown to the generator
    """
    prompt = f"""You are an evaluation judge. Rate how FAITHFUL the answer is to the provided context — that is, whether the answer's claims are supported by the context and not made up.

Context:
{retrieved_context}

Answer to evaluate:
{result.generated_answer}

Scoring guide:
5 = every claim in the answer is supported by the context
4 = mostly supported, minor unsupported details
3 = partially supported, some claims not in context
2 = largely unsupported by the context
1 = answer contradicts or ignores the context entirely

Note: an answer that says "I don't have enough information" should score 5 if the context genuinely lacks the info — refusing to fabricate IS faithful.

Respond with exactly: SCORE: N
Then one sentence of justification."""

    response = _judge_client.chat.completions.create(
        model=_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )

    text = response.choices[0].message.content
    return _extract_score(text)


def score_relevance(result: EvalResult) -> float:
    """
    Judge whether the answer actually addresses the question.

    Independent of faithfulness — an answer can be faithful to
    context but not answer the question (e.g. "I don't know").

    Score 1-5:
        5 = directly and completely answers the question
        1 = does not address the question at all
    """
    prompt = f"""You are an evaluation judge. Rate how well the answer ADDRESSES the question — regardless of whether it's correct, just whether it actually responds to what was asked.

Question:
{result.question}

Answer to evaluate:
{result.generated_answer}

Scoring guide:
5 = directly and fully answers the question
4 = answers the question with minor gaps
3 = partially answers the question
2 = barely addresses the question
1 = does not answer the question at all (e.g. "I don't have enough information")

Respond with exactly: SCORE: N
Then one sentence of justification."""

    response = _judge_client.chat.completions.create(
        model=_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )

    text = response.choices[0].message.content
    return _extract_score(text)