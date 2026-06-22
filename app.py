"""
app.py

Gradio web interface for the Agentic RAG Research Assistant.
Wraps the LangGraph agent in a chat UI.

Run locally:  python app.py
"""

import gradio as gr
from src.agent.graph import build_graph

# Build the agent once, when the app starts.
# This loads the embedder, vector store, LLM, and web search —
# all the heavy setup happens here, not on every message.
print("Building agent (loading models and index)...")
agent = build_graph()
print("Agent ready.")

def respond(message, history):
    """
    Run one user question through the agent and return the answer.

    Gradio's ChatInterface calls this for each message:
      message — the user's question (string)
      history — past turns (managed by Gradio; we don't need it
                since each question is answered independently)

    Returns the agent's answer as a formatted string, including
    which sources were used.
    """
    # Build the initial state the agent expects
    initial_state = {
        "question": message,
        "needs_papers": False,
        "needs_web": False,
        "paper_context": "",
        "web_context": "",
        "answer": "",
        "sources": [],
    }

    # Run the agent
    final_state = agent.invoke(initial_state)

    answer = final_state["answer"]
    sources = final_state.get("sources", [])

    # Append a sources line so users see where the answer came from
    if sources:
        source_label = ", ".join(sources)
        answer = f"{answer}\n\n---\n*Sources: {source_label}*"

    return answer

# Build the chat interface
demo = gr.ChatInterface(
    fn=respond,
    title="Agentic RAG Research Assistant",
    description=(
        "Ask questions about 20 foundational AI/ML research papers "
        "(transformers, RAG, LoRA, RLHF, diffusion, and more). "
        "For current questions, the agent searches the live web. "
        "It cites its sources and declines off-topic questions."
    ),
    examples=[
        "What is LoRA and how does it reduce trainable parameters?",
        "How does the self-attention mechanism work?",
        "What is the newest Llama model in 2026?",
        "How does the newest Llama compare to the original transformer?",
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()