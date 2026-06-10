import sys
from pathlib import Path


# --------------------------
# CONNECT FRAMEWORK
# --------------------------

framework_path = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "rag-pipeline-framework"
)

sys.path.insert(
    0,
    str(framework_path)
)


from generation.llm import (
    LLMModel
)


class QueryRouter:

    def __init__(self):

        self.llm = (
            LLMModel()
            .get_llm()
        )

    def route(
        self,
        query
    ):

        prompt = f"""
You are a query router.

Your job is to classify
the user query into:

1. RAG
2. WEB_SEARCH

Rules:

Use RAG if question can
be answered from IPL dataset.

Examples:
- highest runs in IPL
- Virat Kohli form
- RCB vs CSK head to head
- venue stats
- team captain

Use WEB_SEARCH for:
- live data
- CEO names
- stock prices
- chairman names
- current news
- anything not in IPL dataset

User Query:
{query}

Return ONLY:

RAG

or

WEB_SEARCH
"""

        response = (
            self.llm.invoke(
                prompt
            )
        )

        result = (
            response.content
            .strip()
            .upper()
        )

        if (
            "WEB_SEARCH"
            in result
        ):
            return (
                "WEB_SEARCH"
            )

        return "RAG"