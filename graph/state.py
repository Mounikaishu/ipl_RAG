from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict, total=False):
    query: str
    rewritten_query: str
    query_type: str
    entities: List[str]

    # Context fields for each specialized agent
    team_context: List[Document]
    batting_context: List[Document]
    bowling_context: List[Document]
    h2h_context: List[Document]
    venue_context: List[Document]
    form_context: List[Document]
    season_context: List[Document]
    records_context: List[Document]
    validation_context: List[Document]

    # Outputs
    conflict_detected: bool
    final_answer: str
    confidence: float
    score_gap: float

    # Legacy & framework compatibility fields
    chunks: list
    documents: list
    chat_history: list
    last_context: str
    previous_docs: list
    resolved_entities: list
    resolved_entity_type: str
    computed_answer: str
    answer: str
    route: str
