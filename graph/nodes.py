
from rewrite.query_rewriter import (
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


# ------------------------
# COMPONENTS
# ------------------------

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


# ------------------------
# REWRITE NODE
# ------------------------

def rewrite_node(
    state
):

    rewritten_query = (
        rewriter.rewrite(
            state["query"],
            state.get(
                "chat_history",
                []
            )
        )
    )

    print(
        "\nRewritten Query:",
        rewritten_query
    )

    return {
        "rewritten_query":
        rewritten_query
    }


# ------------------------
# ROUTER NODE
# ------------------------

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


# ------------------------
# RETRIEVAL NODE
# ------------------------

def retrieval_node(
    state
):

    original_query = (
        state[
            "query"
        ]
        .lower()
    )

    followup_words = [

        "he",
        "his",
        "him",
        "she",
        "her",
        "they",
        "them",
        "that player",
        "this player",
        "those players",
        "those teams",
        "their",
        "its"
    ]

    print(
        "\nOriginal Query:",
        original_query
    )

    print(
        "Last Context Size:",
        len(
            str(
                state.get(
                    "last_context",
                    ""
                )
            )
        )
    )

    # ------------------------
    # FOLLOW-UP DETECTION
    # ------------------------

    is_followup = any(

        word in original_query

        for word

        in followup_words
    )

    # ------------------------
    # NORMAL RETRIEVAL
    # ------------------------

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


# ------------------------
# RERANK NODE
# ------------------------

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


# ------------------------
# REFINE NODE
# ------------------------

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


# ------------------------
# GENERATE NODE
# ------------------------

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
            ],

            last_context=
            state.get(
                "last_context",
                ""
            )
        )
    )

    return {
        "answer":
        answer,

        "retrieved_docs":
        state[
            "refined_docs"
        ]
    }


# ------------------------
# WEB SEARCH NODE
# ------------------------

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


# ------------------------
# ROUTING LOGIC
# ------------------------

def routing_logic(
    state
):

    return state[
        "route"
    ]


# ------------------------
# CONFIDENCE LOGIC
# ------------------------

def confidence_logic(
    state
):

    confidence = (
        state.get(
            "confidence",
            100
        )
    )

    print(
        "\nConfidence:",
        confidence
    )

    if confidence < 5:

        return (
            "WEB_SEARCH"
        )

    return (
        "GENERATE"
    )
