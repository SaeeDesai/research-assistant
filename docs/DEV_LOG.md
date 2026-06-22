## Day 12 — Improving retrieval precision

**Problem (from Day 11 eval):** Top-1 precision 80%, recall@5 100%.
The gap = a ranking problem: right doc retrieved but sometimes
ranked below a semantically-similar lookalike (RAGAS above RAG).

**Experiment 1 — Cross-encoder reranking (ms-marco-MiniLM-L-6-v2):**
Two-stage retrieve-15-rerank-to-5. MEASURED RESULT: top-1 precision
*dropped* 80% → 60%. The general-purpose reranker, trained on MS MARCO
web-search data, underperforms on a homogeneous corpus where every
candidate is a dense, topically-similar AI paper. It broke the WMT
case (promoted bert.pdf over the attention paper). Conclusion:
reranking is not suited to this corpus. Kept the code as a documented
experiment; not used in the default path.

**Experiment 2 — Metadata source filtering:**
When the target paper is known, restrict retrieval to that source so
lookalikes are excluded. MEASURED RESULT: fixed both known failures —
RAG question returns RAG.pdf chunks; WMT question surfaces the
newstest2014 dataset chunk. Caveat: current implementation uses
keyword→source matching, which works because test questions contain
paper-identifying keywords. Production version would have the planner
LLM infer the target source.

**Takeaway:** Decomposed eval metrics correctly identified a ranking
problem; the first "obvious" fix (reranking) regressed and was rejected
with evidence; the corpus-appropriate fix (source filtering) succeeded.