
import sys
from pathlib import (
    Path
)


# --------------------------
# CONNECT FRAMEWORK
# --------------------------

framework_path = (
    Path(__file__)
    .resolve()
    .parent.parent.parent
    / "rag-pipeline-framework"
)

sys.path.append(
    str(framework_path)
)

print(
    "\nFramework Path:",
    framework_path
)


# --------------------------
# IMPORTS
# --------------------------

from query_processing.rewrite import (
    QueryRewriter
)

from routing.query_router import (
    QueryRouter
)

from retrieval.hybrid_retriever import (
    HybridRetriever
)

from rerank.reranker import (
    Reranker
)

from refine.refiner import (
    Refiner
)

from ipl_generation.generator import (
    IPLGenerator
)

from tools.web_search import (
    WebSearchTool
)

from memory.conversation_memory import (
    ConversationMemory
)


# --------------------------
# COMPONENTS
# --------------------------

rewriter = (
    QueryRewriter()
)

router = (
    QueryRouter()
)

retriever = (
    HybridRetriever(
        persist_directory=
        "chroma_db"
    )
)

reranker = (
    Reranker()
)

refiner = (
    Refiner()
)

generator = (
    IPLGenerator()
)

web_tool = (
    WebSearchTool()
)

memory = (
    ConversationMemory()
)


# --------------------------
# REWRITE NODE
# --------------------------

def rewrite_node(
    state
):

    history = (
        memory.get_context()
    )

    query = (
        state["query"]
    )

    prompt = f"""
You are an intelligent
query rewriter.

Conversation History:
{history}

Current User Question:
{query}

Your task:

Rewrite the question
using conversation history.

Rules:

1. Resolve pronouns:
he, she, his, her,
they, them.

2. Resolve references:
that player,
that team,
that match,
that stadium.

3. If previous answer
mentions a player,
replace vague terms.

Example:

History:
User:
Who scored highest runs?

Assistant:
Virat Kohli scored
highest runs.

Question:
How many centuries
does he have?

Rewrite:
How many centuries
does Virat Kohli
have in IPL?

Return ONLY the
rewritten query.
"""

    rewritten_query = (
        rewriter.llm
        .invoke(prompt)
        .content
        .strip()
    )

    print(
        "\nRewritten Query:",
        rewritten_query
    )

    return {
        "rewritten_query":
        rewritten_query
    }

# --------------------------
# ROUTER NODE
# --------------------------

def router_node(
    state
):

    route = (
        router.route(
            state[
                "rewritten_query"
            ]
        )
    )

    print(
        "\nRoute Chosen:",
        route
    )

    return {
        "route":
        route
    }


# --------------------------
# RETRIEVAL NODE
# --------------------------

def retrieval_node(
    state
):

    retrieved_docs = (
        retriever.retrieve(
            query=
            state[
                "rewritten_query"
            ],

            chunks=
            state[
                "chunks"
            ],

            k=5
        )
    )

    return {

        "retrieved_docs":
        retrieved_docs
    }


# --------------------------
# RERANK NODE
# --------------------------

def rerank_node(
    state
):

    rerank_result = (
        reranker.rerank(
            state[
                "rewritten_query"
            ],

            state[
                "retrieved_docs"
            ]
        )
    )

    return {

        "reranked_docs":
        rerank_result[
            "docs"
        ],

        "confidence":
        rerank_result[
            "confidence"
        ],

        "score_gap":
        rerank_result[
            "score_gap"
        ]
    }


# --------------------------
# REFINE NODE
# --------------------------

def refine_node(
    state
):

    refined_docs = (
        refiner.refine(
            state[
                "reranked_docs"
            ]
        )
    )

    return {

        "refined_docs":
        refined_docs
    }


# --------------------------
# GENERATE NODE
# --------------------------

def generate_node(
    state
):

    answer = (
        generator
        .generate_answer(
            query=
            state[
                "rewritten_query"
            ],

            retrieved_docs=
            state[
                "refined_docs"
            ]
        )
    )

    return {

        "answer":
        answer
    }


# --------------------------
# WEB SEARCH NODE
# --------------------------

def web_search_node(
    state
):

    web_result = (
        web_tool.search(
            state[
                "rewritten_query"
            ]
        )
    )

    answer = (
        generator
        .summarize_web_answer(
            state[
                "rewritten_query"
            ],

            web_result
        )
    )

    return {

        "answer":
        answer
    }


# --------------------------
# MEMORY UPDATE NODE
# --------------------------

def memory_update_node(
    state
):

    memory.add_message(
        "user",

        state[
            "query"
        ]
    )

    memory.add_message(
        "assistant",

        state[
            "answer"
        ]
    )

    return {}


# --------------------------
# ROUTING LOGIC
# --------------------------

def routing_logic(
    state
):

    return state[
        "route"
    ]


def confidence_logic(
    state
):

    if (
        state[
            "confidence"
        ]
        < 8
    ):

        return (
            "WEB_SEARCH"
        )

    return (
        "GENERATE"
    )
