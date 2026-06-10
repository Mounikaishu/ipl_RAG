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


class QueryRewriter:

    def __init__(self):

        self.llm = (
            LLMModel()
            .get_llm()
        )

    def rewrite(
        self,
        query
    ):

        prompt = f"""
Rewrite the query for better retrieval.

Rules:
1. Expand abbreviations
2. Fix vague wording
3. Keep original meaning
4. Return ONLY rewritten query

User Query:
{query}
"""

        response = (
            self.llm.invoke(
                prompt
            )
        )

        rewritten_query = (
            response.content
            .strip()
        )

        print(
            "\nRewritten Query:",
            rewritten_query
        )

        return (
            rewritten_query
        )