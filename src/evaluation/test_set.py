"""
test_set.py

A ground-truth evaluation dataset: questions paired with reference answers and the source we expect the answer to come from.

This is the foundation of evaluation - we can only measure
answer quality by comparing against known-good references.
"""

from dataclasses import dataclass

@dataclass
class TestCase:
    """
    A single evaluation example.

    Fields:
     - question -> the question to ask the agent
     - reference answer -> a known-good answer(the gorund truth)
     - expected_source -> which paper the answer should come from
     - category -> what kind of question this is, so we can measure performance per category
    """

    question: str
    reference_answer: str
    expected_source: str
    category: str

# The ground-truth evaluation set.
# Each reference answer is based on what the paper actually says.
# We include answerable questions across categories, plus a few
# that test the system's edges (off-topic, current-info).

TEST_CASES = [
    # --- Conceptual questions (core capability) ---
    TestCase(
        question="What is LoRA and how does it reduce trainable parameters?",
        reference_answer=(
            "LoRA adds trainable pairs of low-rank decomposition matrices "
            "in parallel to the existing weight matrices, rather than "
            "updating the full weight matrices. The number of trainable "
            "parameters is determined by the chosen rank r and the shape of "
            "the weights, so a small rank yields far fewer trainable "
            "parameters than full fine-tuning."
        ),
        expected_source="lora.pdf",
        category="conceptual",
    ),
    TestCase(
        question="How does the self-attention mechanism work in transformers?",
        reference_answer=(
            "Self-attention is the core mechanism in the Transformer's "
            "encoder and decoder stacks. The encoder is built from 6 "
            "identical layers, each containing a multi-head self-attention "
            "sub-layer followed by a position-wise feed-forward sub-layer, "
            "allowing each position to attend to all positions in the "
            "previous layer."
        ),
        expected_source="attention_is_all_you_need.pdf",
        category="conceptual",
    ),
    # KNOWN RETRIEVAL FAILURE (documented intentionally):
    # This question retrieves ragas.pdf and self-rag.pdf above the
    # original RAG.pdf chunk that defines the concept — because four
    # papers in the corpus all discuss "retrieval augmented generation"
    # (semantic overlap). Candidate fixes: sharpen the query, add
    # metadata/source filtering, or add a reranking step.
    TestCase(
        question="What is the main idea behind retrieval-augmented generation?",
        reference_answer=(
            "RAG combines a parametric memory (a pre-trained seq2seq model) "
            "with a non-parametric memory (a dense vector index of documents "
            "accessed via a retriever). It retrieves relevant passages and "
            "conditions generation on them, grounding answers in retrieved "
            "knowledge and reducing hallucination."
        ),
        expected_source="RAG.pdf",
        category="conceptual",
    ),

    # --- Specific-detail questions (harder for retrieval) ---
    # KNOWN RETRIEVAL FAILURE (documented intentionally):
    # The WMT 2014 training-data chunk exists in the index but does not
    # rank in the top-k. The query "what datasets was X trained on"
    # semantically matches OTHER papers' training-data discussions
    # (e.g. Llama2's "2 trillion tokens") above the transformer's actual
    # answer. This is the failure first observed on Day 5 of development.
    # Candidate fixes: query expansion, metadata/source filtering,
    # increasing k, or a reranking step.
    TestCase(
        question="What datasets was the original transformer trained on?",
        reference_answer=(
            "The transformer was trained on the WMT 2014 English-German "
            "dataset (about 4.5 million sentence pairs) and the larger WMT "
            "2014 English-French dataset (about 36 million sentences)."
        ),
        expected_source="attention_is_all_you_need.pdf",
        category="specific_detail",
    ),
    TestCase(
        question="How many layers are in the encoder of the original transformer?",
        reference_answer=(
            "The encoder is composed of a stack of 6 identical layers, each "
            "with a multi-head self-attention sub-layer and a position-wise "
            "feed-forward sub-layer."
        ),
        expected_source="attention_is_all_you_need.pdf",
        category="specific_detail",
    ),

    # --- Off-topic (should be refused) ---
    TestCase(
        question="What's a good recipe for pasta carbonara?",
        reference_answer="REFUSAL_EXPECTED",
        expected_source="none",
        category="offtopic",
    ),
]