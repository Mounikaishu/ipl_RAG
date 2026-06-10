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


class QueryClassifier:

    def __init__(self):

        self.llm = (
            LLMModel()
            .get_llm()
        )

    def classify(
        self,
        query
    ):

        prompt = f"""
You are a query routing agent.

Your job is to classify
the user's query into
ONE category only.

Categories:

1. RAG
Use this if answer can
be found in IPL dataset.

Examples:
- highest runs
- wickets
- player stats
- batting
- bowling
- team records
- venue reports
- head to head
- recent form
- season trends

2. MULTI_DOC_RAG
Use this when answer
needs multiple IPL
documents together.

Examples:
- predict match
- compare teams
- best playing XI
- RCB vs CSK analysis
- who has advantage

3. WEB_SEARCH
Use this for:
- live info
- latest updates
- today match
- current IPL table
- CEO/chairman
- news
- anything not in dataset

Return ONLY one word:
RAG
MULTI_DOC_RAG
WEB_SEARCH

Query:
{query}

Category:
"""

        response = (
            self.llm.invoke(
                prompt
            )
        )

        category = (
            response.content
            .strip()
            .upper()
        )

        return category
