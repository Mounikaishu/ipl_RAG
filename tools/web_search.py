
import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from tavily import TavilyClient


# --------------------------
# LOAD ENV VARIABLES
# --------------------------

load_dotenv()


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


class WebSearchTool:

    def __init__(self):

        api_key = os.getenv(
            "TAVILY_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "TAVILY_API_KEY "
                "not found in .env"
            )

        self.client = (
            TavilyClient(
                api_key=api_key
            )
        )

        self.llm = (
            LLMModel()
            .get_llm()
        )

    def search(
        self,
        query
    ):

        try:

            response = (
                self.client.search(
                    query=query,
                    search_depth=
                    "advanced",
                    max_results=5
                )
            )

            print(
                "\nTavily Response:\n"
            )

            print(
                response
            )

            results = (
                response.get(
                    "results",
                    []
                )
            )

            if not results:

                return (
                    "No web result found."
                )

            context = (
                "\n\n".join(
                    [
                        f"Title: "
                        f"{r.get('title')}\n"
                        f"Content: "
                        f"{r.get('content')}"
                        for r
                        in results
                    ]
                )
            )

            prompt = f"""
You are a factual assistant.

Answer ONLY from
the web search context.

STRICT RULES:

1. Give short,
accurate answers.

2. Prefer
latest/current info.

3. Do NOT dump
article text.

4. Answer in
1-3 lines only.

5. If multiple
sources agree,
choose the
most likely
current answer.

Web Context:
{context}

Question:
{query}

Answer:
"""

            llm_response = (
                self.llm.invoke(
                    prompt
                )
            )

            return (
                llm_response.content
                .strip()
            )

        except Exception as e:

            return (
                f"Web search error: {e}"
            )
