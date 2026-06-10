
from typing import (
    TypedDict,
    List
)


class GraphState(
    TypedDict
):

    query: str

    rewritten_query: str

    route: str

    chunks: List

    chat_history: List

    retrieved_docs: List

    reranked_docs: List

    refined_docs: List

    answer: str

    confidence: float

    score_gap: float

    web_result: str

