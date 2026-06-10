
from langgraph.graph import (
    StateGraph,
    END
)

from graph.state import (
    GraphState
)

from graph.nodes import (
    rewrite_node,
    router_node,
    retrieval_node,
    rerank_node,
    refine_node,
    generate_node,
    web_search_node,
    routing_logic,
    confidence_logic
)

# ------------------------
# GRAPH
# ------------------------

workflow = StateGraph(
    GraphState
)

# ------------------------
# NODES
# ------------------------

workflow.add_node(
    "rewrite",
    rewrite_node
)

workflow.add_node(
    "router",
    router_node
)

workflow.add_node(
    "retrieve",
    retrieval_node
)

workflow.add_node(
    "rerank",
    rerank_node
)

workflow.add_node(
    "refine",
    refine_node
)

workflow.add_node(
    "generate",
    generate_node
)

workflow.add_node(
    "web_search",
    web_search_node
)

# ------------------------
# ENTRY
# ------------------------

workflow.set_entry_point(
    "rewrite"
)

# ------------------------
# FLOW
# ------------------------

workflow.add_edge(
    "rewrite",
    "router"
)

workflow.add_conditional_edges(
    "router",
    routing_logic,
    {
        "RAG":
        "retrieve",

        "WEB_SEARCH":
        "web_search"
    }
)

workflow.add_edge(
    "retrieve",
    "rerank"
)

workflow.add_edge(
    "rerank",
    "refine"
)

workflow.add_conditional_edges(
    "refine",
    confidence_logic,
    {
        "GENERATE":
        "generate",

        "WEB_SEARCH":
        "web_search"
    }
)

workflow.add_edge(
    "generate",
    END
)

workflow.add_edge(
    "web_search",
    END
)

# ------------------------
# COMPILE
# ------------------------

app = workflow.compile()

print(
    "\nLangGraph Ready!"
)
