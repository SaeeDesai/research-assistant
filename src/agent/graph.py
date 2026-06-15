"""
graph.py

The LangGraph agent. Starts as a trivial two-node graph 
so we understand the mechanics, 
then grows into a real agent with 
routing and tools

"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    """
    The state that flows through the graph.

    Every node receives this, can read from it, and returns
    an updated version. Think of it as the backpack the agent carries from node to node.

    For now it hold two things. We will add more fields
    (retrieved chunks, chosen tool etc) as the tool grows.
    """

    question: str      # The user's question
    answer: str        # The answer we build up as we go

def greet_node(state: AgentState) -> AgentState:
    """
    First node. Reads the question from state and starts building an answer.

    A node eceives the current state, does it's work and returns a dict with the fields it wants to update.
    LangGraph merges that duct back into the state.
    """

    question = state["question"]
    print(f" [greet_node] received question: '{question}'")

    # We return only the field we want to update.
    # LangGraph merges this into the existing state

    return {"answer": f"You asked: '{question}'. "} # type: ignore

def respond_node(state: AgentState) -> AgentState:
    """
    Second node. Takes the answer started by greet_node and adds to it.
    Notice it can read 'amswer' - which the previous node wrote. That is the state
    flowing through the graph: each node sees what previous nodes leftf behind
    """

    current_answer = state["answer"]
    print(f" [respond_node] current answer so far: '{current_answer}'")

    # Append to the answer the previous node started
    updated = current_answer + "This is a response from the second node."

    return {"answer": updated} # type: ignore


def build_graph():
    """
    Assemble the nodes into a runnable graph.

    Three things happen here:
    1) Create a graph that uses our AgentState
    2) Add the nodes to it
    3) Connect them with edges (define the flow) 
    """

    # Create the graph, telling it what state shape to use
    builder = StateGraph(AgentState)

    # Add our two nodes. The first argument is a name (a string label for this node), the second is the actual function.
    builder.add_node("greet", greet_node)
    builder.add_node("respond", respond_node)

    # Now we define the flow with edges.
    # START is a special marker for where the graph begins
    # END is a special marker for where it finishes
    builder.add_edge(START, "greet")
    builder.add_edge("greet", "respond")
    builder.add_edge("respond", END)

    # Compile turns the builder into a runnable graph
    graph = builder.compile()

    return graph

# --- Test Block ---
if __name__ == '__main__':
    # Build the graph
    graph = build_graph()

    # Define the starting state.
    # We provide the question. The amswer starts empty - the nodes will fill it in
    # as the graph runs.

    initial_state = {
        "question": "How does attention work?",
        "answer": ""
    }

    print("="*50)
    print("RUNNING THE SCRIPT")
    print("="*50)
    print(f"\nInitial State: {initial_state}\n")

    # invoke() runs the graph from START to END, passing the state through each node in order.
    final_state = graph.invoke(initial_state)

    print(f"Final state: {final_state}")
    print(f"\nFinal answer: {final_state['answer']}")