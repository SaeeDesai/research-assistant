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

class AgentState(TypedDict):
    """
    The state that flows through the graph.

    - route: the router's decision ('papers' and 'off-topic')
    - sources: which documents the answer came from
    """

    question: str      # The user's question
    route: str         # router's decision
    answer: str        # the final answer
    sources: list      # source documents used


_embedder = Embedder()
_store = VectorStore(_embedder)
_store.load('vector_store')
_rag = RAGChain(_store)

def router_node(state: AgentState) -> AgentState:
    """
    Looks at the question and decides where to route it.

    Uses the LLM as a classifier: is this question about AI/ML research papers, or is it off-topic?
    Writes the decision to state['route'] as either 'papers' or 'off-topic'. The conditional edge reads this field to decide the next node
    """

    question = state["question"]
    print(f" [router] classifying: '{question}'")

    # We ask LLM to classify the question
    # The prompt is tight and forces a one-word answer
    # so we can route it reliably

    classification_prompt = f""" You are a classifier. Decide if the following question is about AI, machine learning, deep learning, NLP, transformers or related research topics.

Question: "{question}"

Answer with ONLY one word:
- "papers" if the question is about AI/ML/deep learning research topics
- "offtopic" if it is about anything else

Answer: """
    
    response = _rag.client.chat.completions.create(
        model=_rag.LLM_MODEL,
        messages=[{"role": "user", "content": classification_prompt}],
        temperature=0.0,  # zero randomess, we want consistent classification
        max_tokens=10
    )

    decision = response.choices[0].message.content.strip().lower()

    # Defensive: if the LLM says anything unexpected, default to papers
    # (better to try answering than to wrongly refuse)
    if "offtopic" in decision:
        route = "offtopic"
    else:
        route = "papers"

    print(f" [router] decision: {route}")

    return {"route": route}


def rag_node(state: AgentState) -> AgentState:
    """
    Answers the question using the RAG system from week 1

    Called only when the router decided the question is about AI/ML papers.
    Reuses the RAGChain we built - the agent doesn't reimplement RAG, it delegates to it
    """

    question = state["question"]
    print(f" [rag] answering from papers: '{question}'")

    # Delegate to the Week 1 RAG chain
    response = _rag.answer(question, k=5)

    # Pull out the unique source filenames for citation
    source_names = []
    for chunk in response.sources:
        name = chunk.metadata.get("source", "unknown")
        if name not in source_names:
            source_names.append(name)

    return {
        "answer": response.answer,
        "sources": source_names
    }

def refuse_node(state: AgentState) -> AgentState:
    """
    Politely declines off-topic questions.

    Called only when the router decides the question is not about AI/ML papers.
    No LLM call, no retreival - just a fixed polite response. This saves compute on questions we can't answer anyway.

    """
    question = state["question"]
    print(f" [refuse] declining off-topic: '{question}'")

    answer = (
        "I'm a research assistant focused on AI and machine learning "
        "papers. That question falls outside the papers I have access to, "
        "so I can't answer it. Try asking me about transformers, RAG, "
        "LoRA, fine-tuning, or other AI/ML topics."
    )

    return {
        "answer": answer,
        "sources": []
    }

def route_decision(state: AgentState) -> AgentState:
    """
    The conditional edge function.

    This is NOT a node - it doesn't do work or change state.
    It reads the router's decision from state and returns the NAME of the next node to run

    LangGraph calls this function after the router node, looks at what it returns, and sends the flow to that node.

    Return value must match a node name we registered in the graph.
    """

    route = state["route"]
    print(f" [route_decision] routing to: {route}")

    if route == "offtopic":
        return "refuse"
    else:
        return "rag"
    

def build_graph():
    """
    Assemble the agentic RAG graph.
    Flow: START -> router -> (conditional) -> rag -> END
                                   |
                                 refuse -> END
    The router always runs first. Then the conditional edge sends the flow to either 
    rag or refuse based on the router's decision.
    """
    builder = StateGraph(AgentState)

    # Register the three nodes
    builder.add_node("router", router_node)
    builder.add_node("rag", rag_node)
    builder.add_node("refuse", refuse_node)

    # Start always goes to the router first
    builder.add_edge(START, "router")

    # THE CONDITIONAL EDGE - this is the new mechanism
    # After 'router' runs, call route_decision to choose
    # the next node. The dictionary maps the function's
    # return values to actual node names.
    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "rag",
            "refuse": "refuse",
        }
    )

    # Both destination nodes lead to END
    builder.add_edge("rag", END)
    builder.add_edge("refuse", END)

    graph = builder.compile()
    return graph

if __name__ == '__main__':
    graph = build_graph()

    test_questions = [
        "What is LoRA and how does it reduce trainable parameters?",
        "How does the attention mechanism work?",
        "What's a good recipe for pasta?",
        "Who won the football match last night?",
    ]

    for question in test_questions:
        print("\n" + "=" * 60)
        print(f"QUESTION:{question}")
        print("=" * 60)

        # Run the agent, answer the sources start empty;
        # the nodes fill them in depending on the path taken.
        initial_state = {
            "question": question,
            "route": "",
            "answer": "",
            "sources": []
        }

        final_state = graph.invoke(initial_state)

        print(f"\n ROUTE TAKEN: {final_state['route']}")
        print(f" ANSWER: {final_state['answer'][:250]}")
        if final_state['sources']:
            print(f" SOURCES: {final_state['sources']}")