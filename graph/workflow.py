from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import (
    rewrite_node,
    router_node,
    team_profile_node,
    batting_stats_node,
    bowling_stats_node,
    h2h_node,
    venue_node,
    form_node,
    records_node,
    season_node,
    validation_node,
    synthesis_node,
    web_search_node,
    routing_logic,
    dream11_node
)

# ------------------------
# PIPELINE EDGE ROUTERS
# ------------------------

def route_batting(state: GraphState):
    qtype = state.get("query_type", "")
    if qtype == "dream11":
        return "bowling"
    return "validation"

def route_h2h(state: GraphState):
    qtype = state.get("query_type", "")
    if qtype == "prediction":
        return "venue"
    return "validation"

def route_venue(state: GraphState):
    qtype = state.get("query_type", "")
    if qtype == "prediction":
        return "form"
    return "validation"

def route_form(state: GraphState):
    qtype = state.get("query_type", "")
    if qtype == "dream11":
        return "batting"
    return "validation"

# ------------------------
# BUILD STATE GRAPH
# ------------------------

workflow = StateGraph(GraphState)

# Add all specialized agent nodes
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("router", router_node)
workflow.add_node("team", team_profile_node)
workflow.add_node("batting", batting_stats_node)
workflow.add_node("bowling", bowling_stats_node)
workflow.add_node("h2h", h2h_node)
workflow.add_node("venue", venue_node)
workflow.add_node("form", form_node)
workflow.add_node("records", records_node)
workflow.add_node("season", season_node)
workflow.add_node("validation", validation_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("dream11", dream11_node)

# Set Entry Point
workflow.set_entry_point("rewrite")

# Rewrite connects directly to Router
workflow.add_edge("rewrite", "router")

# Router routes conditionally to specialized agents or Web Search
workflow.add_conditional_edges(
    "router",
    routing_logic,
    {
        "team": "team",
        "batting": "batting",
        "bowling": "bowling",
        "h2h": "h2h",
        "venue": "venue",
        "form": "form",
        "records": "records",
        "season": "season",
        "fallback": "web_search",
        "dream11": "dream11"
    }
)

# Connect single-agent and multi-agent flows
workflow.add_edge("team", "validation")
workflow.add_edge("records", "validation")
workflow.add_edge("season", "validation")
workflow.add_edge("bowling", "validation")
workflow.add_edge("dream11", "validation")

# Batting routing (conditional for Dream11 multi-step flow)
workflow.add_conditional_edges(
    "batting",
    route_batting,
    {
        "bowling": "bowling",
        "validation": "validation"
    }
)

# Head-to-Head routing (conditional for Prediction flow)
workflow.add_conditional_edges(
    "h2h",
    route_h2h,
    {
        "venue": "venue",
        "validation": "validation"
    }
)

# Venue routing (conditional for Prediction flow)
workflow.add_conditional_edges(
    "venue",
    route_venue,
    {
        "form": "form",
        "validation": "validation"
    }
)

# Recent Form routing (conditional for Dream11 flow)
workflow.add_conditional_edges(
    "form",
    route_form,
    {
        "batting": "batting",
        "validation": "validation"
    }
)

# Validation always goes to final Synthesis
workflow.add_edge("validation", "synthesis")

# End of flow paths
workflow.add_edge("synthesis", END)
workflow.add_edge("web_search", END)

# Compile Graph
app = workflow.compile()

print("\n[Multi-Agent LangGraph] Workflow Ready & Compiled!")
