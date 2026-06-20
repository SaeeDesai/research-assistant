"""
graph.py

An Agentic RAG system built with LangGraph.
The agent looks at each question and routes it:
- Questions about AI/ML papers -> answered via RAG
- Off-topic questions + politely declined

This routing decision is what makes it an agent rather than a fixed pipeline

"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

from src.retrieval.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.generation.rag_chain import RAGChain

from src.agent.web_search import WebSearchTool

class AgentState(TypedDict):
    """
    State for the multi-tool agent

    New fields for Day 9;
     - needs_papers: did the planner decide we need the paper corpus?
     - needs_web: did the planner decide we need web search?
     - paper_context: text gathered from RAG retreival
     - web_context: textr gathered from web search
    """

    question: str      # The user's question
    needs_papers: bool # planner decision: use papers?
    needs_web: bool    # planner decision: needs web?
    paper_context: str # context gathered from the papers
    web_context: str   # context gathered from web
    answer: str        # the final answer
    sources: list      # source documents used


_embedder = Embedder()
_store = VectorStore(_embedder)
_store.load('vector_store')
_rag = RAGChain(_store)
_web = WebSearchTool()

def planner_node(state: AgentState) -> AgentState:
    """
    Decides which tools the question needs — possibly more than one.

    Unlike the Day 8 router (which picked ONE path), the planner
    makes two independent yes/no decisions: papers? web? This lets
    it handle questions that need both sources.

    Writes needs_papers and needs_web to the state.
    """
    question = state["question"]
    print(f"  [planner] planning tools for: '{question}'")

    planning_prompt = f"""You are a planning assistant. Decide which information sources are needed to answer the question.

Question: "{question}"

Two sources are available:
1. PAPERS — a corpus of foundational AI/ML research papers (transformers, BERT, RAG, LoRA, RLHF, diffusion, etc.). Use for conceptual or theoretical questions about established AI/ML methods.
2. WEB — live web search. Use for current/recent information: latest model releases, recent news, current versions, anything time-sensitive.

A question may need ONE source, BOTH, or NEITHER (if it's completely off-topic, not about AI/ML at all).

Respond in exactly this format, with yes or no for each:
PAPERS: yes/no
WEB: yes/no

Examples:
- "What is LoRA?" → PAPERS: yes, WEB: no
- "What's the newest Llama model?" → PAPERS: no, WEB: yes
- "How does the newest Llama compare to the original transformer?" → PAPERS: yes, WEB: yes
- "What's a good pasta recipe?" → PAPERS: no, WEB: no

Now answer for the question above:"""

    response = _rag.client.chat.completions.create(
        model=_rag.LLM_MODEL,
        messages=[{"role": "user", "content": planning_prompt}],
        temperature=0.0,
        max_tokens=30
    )

    decision_text = response.choices[0].message.content.lower()

    # Parse the two decisions from the response
    needs_papers = "papers: yes" in decision_text
    needs_web = "web: yes" in decision_text

    print(f"  [planner] needs_papers={needs_papers}, needs_web={needs_web}")

    return {
        "needs_papers": needs_papers,
        "needs_web": needs_web
    }
def gather_papers_node(state: AgentState) -> AgentState:
    """
    Gathers context from the paper corpus IF the planner
    decided papers are needed.

    Note: this node does NOT generate an answer. It only
    retrieves context and stores it. Generation happens
    later, once all context from all sources is gathered.
    """
    if not state.get("needs_papers"):
        # Planner said papers aren't needed — skip, store nothing
        print(f"  [gather_papers] skipped (not needed)")
        return {"paper_context": ""}

    question = state["question"]
    print(f"  [gather_papers] retrieving from papers")

    # Use the vector store directly to get chunks (no generation)
    results = _store.search(question, k=5)

    # Format chunks into context text
    context_parts = []
    sources = []
    for chunk, score in results:
        source = chunk.metadata.get("source", "unknown")
        context_parts.append(f"[From {source}]\n{chunk.content}")
        if source not in sources:
            sources.append(source)

    paper_context = "\n\n".join(context_parts)

    return {
        "paper_context": paper_context,
        "sources": sources
    }


def gather_web_node(state: AgentState) -> AgentState:
    """
    Gathers context from web search IF the planner decided
    web is needed.

    Like gather_papers_node, this only collects context —
    no answer generation here.
    """
    if not state.get("needs_web"):
        print(f"  [gather_web] skipped (not needed)")
        return {"web_context": ""}

    question = state["question"]
    print(f"  [gather_web] searching the web")

    web_context = _web.search(question, max_results=5)

    # Append "web search" to existing sources
    existing_sources = state.get("sources", [])
    updated_sources = existing_sources + ["web search"]

    return {
        "web_context": web_context,
        "sources": updated_sources
    }

def generate_node(state: AgentState) -> AgentState:
    """
    Generates the final answer from all gathered context.

    This is the synthesis step. It combines paper_context and
    web_context (whichever were gathered) into one prompt and
    produces a single unified answer.

    If neither source was gathered, the question was off-topic,
    so we refuse.
    """
    question = state["question"]
    paper_context = state.get("paper_context", "")
    web_context = state.get("web_context", "")

    # Case: neither source gathered → off-topic → refuse
    if not paper_context and not web_context:
        print(f"  [generate] no context — refusing")
        return {
            "answer": (
                "I'm a research assistant focused on AI and machine "
                "learning. That question falls outside what I can help "
                "with. Try asking about AI/ML concepts or recent "
                "developments in the field."
            ),
            "sources": []
        }

    print(f"  [generate] synthesizing answer from gathered context")

    # Build a combined context section, labeling each source type
    context_sections = []
    if paper_context:
        context_sections.append(
            f"=== From research papers ===\n{paper_context}"
        )
    if web_context:
        context_sections.append(
            f"=== From web search ===\n{web_context}"
        )
    combined_context = "\n\n".join(context_sections)

    # Build the generation prompt
    prompt = f"""You are a research assistant. Answer the question using only the context below, which may come from research papers, web search, or both. Synthesize a single coherent answer. If the context combines historical and current information, connect them clearly. Cite which source type information came from when relevant. If the context doesn't fully answer the question, say so.

{combined_context}

Question: {question}

Answer:"""

    response = _rag.client.chat.completions.create(
        model=_rag.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    return {"answer": answer}

    
def build_graph():
    """
    Assemble the multi-tool agent graph.

    Flow (linear, but gather nodes are conditionally active):
        START → planner → gather_papers → gather_web → generate → END

    Unlike Day 8's branching graph, here the nodes run in sequence.
    The planner sets the booleans, and each gather node checks those
    booleans to decide whether to do work or skip. Generation then
    synthesizes whatever was gathered.
    """
    builder = StateGraph(AgentState)

    # Register all four nodes
    builder.add_node("planner", planner_node)
    builder.add_node("gather_papers", gather_papers_node)
    builder.add_node("gather_web", gather_web_node)
    builder.add_node("generate", generate_node)

    # Wire them in sequence
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "gather_papers")
    builder.add_edge("gather_papers", "gather_web")
    builder.add_edge("gather_web", "generate")
    builder.add_edge("generate", END)

    graph = builder.compile()
    return graph


# --- Test block ---
if __name__ == '__main__':
    graph = build_graph()

    test_questions = [
        "What is LoRA and how does it reduce trainable parameters?",      # papers only
        "What is the newest Llama model released in 2026?",               # web only
        "How does the newest Llama model compare to the original transformer architecture?",  # BOTH
        "What's a good recipe for pasta?",                                # neither → refuse
    ]

    for question in test_questions:
        print("\n" + "=" * 65)
        print(f"QUESTION: {question}")
        print("=" * 65)

        initial_state = {
            "question": question,
            "needs_papers": False,
            "needs_web": False,
            "paper_context": "",
            "web_context": "",
            "answer": "",
            "sources": []
        }

        final_state = graph.invoke(initial_state)

        print(f"\n  PLAN: papers={final_state['needs_papers']}, web={final_state['needs_web']}")
        print(f"  ANSWER: {final_state['answer'][:350]}")
        if final_state['sources']:
            print(f"  SOURCES: {final_state['sources']}")