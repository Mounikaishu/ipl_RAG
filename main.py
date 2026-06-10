import sys
from pathlib import Path


# --------------------------
# CONNECT FRAMEWORK
# --------------------------

framework_path = (
    Path(__file__)
    .resolve()
    .parent.parent
    / "rag-pipeline-framework"
)

sys.path.append(
    str(framework_path)
)

print(
    framework_path
)


# --------------------------
# IMPORTS
# --------------------------

from data_ingestion.load_dataset import (
    DatasetLoader
)

from data_ingestion.prepare_chunks import (
    IPLChunkPreparer
)

from ipl_retrieval.vector_store import (
    IPLVectorStore
)

from ipl_generation.generator import (
    IPLGenerator
)

from routing.query_router import (
    QueryRouter
)

from tools.web_search import (
    WebSearchTool
)

from rewrite.query_rewriter import (
    QueryRewriter
)

print(
    QueryRewriter.__module__
)

from memory.cache import (
    CacheMemory
)

from rerank.reranker import (
    Reranker
)

from refine.refiner import (
    Refiner
)

from graph.workflow import (
    app
)


# ------------------------
# LOAD DATASET
# ------------------------

loader = (
    DatasetLoader()
)

documents = (
    loader.load_documents()
)

print(
    "\nLoaded documents:",
    len(documents)
)


# ------------------------
# PREPARE CHUNKS
# ------------------------

chunk_preparer = (
    IPLChunkPreparer()
)

chunks = (
    chunk_preparer
    .prepare_chunks(
        documents
    )
)

print(
    "\nFinal chunks:",
    len(chunks)
)


# ------------------------
# STORE EMBEDDINGS
# ------------------------

vector_store = (
    IPLVectorStore()
)

vector_store.store_documents(
    chunks
)

print(
    "\nIPL Vector DB Ready!"
)


# ------------------------
# COMPONENTS
# ------------------------

cache = (
    CacheMemory()
)

generator = (
    IPLGenerator()
)

router = (
    QueryRouter()
)

web_tool = (
    WebSearchTool()
)

rewriter = (
    QueryRewriter()
)

reranker = (
    Reranker()
)

refiner = (
    Refiner()
)


# ------------------------
# CHAT HISTORY
# ------------------------

chat_history = []


# ------------------------
# QUERY LOOP
# ------------------------

while True:

    query = input(
        "\nAsk IPL Question (or type exit): "
    )

    if (
        query.lower()
        == "exit"
    ):
        break

    normalized_query = (
        query.lower()
        .strip()
    )

    # ------------------------
    # CACHE CHECK
    # ------------------------

    cached_answer = (
        cache.get(
            normalized_query
        )
    )

    if cached_answer:

        print(
            "\nCACHE HIT!"
        )

        print(
            "\nFinal Answer:\n"
        )

        print(
            cached_answer
        )

        continue

    # ------------------------
    # USE PREVIOUS CONTEXT
    # ------------------------

    last_context = []

    if len(chat_history) > 0:

        last_context = (
            chat_history[-1]
            .get(
                "retrieved_docs",
                []
            )
        )

    print(
        "\nOriginal Query:",
        normalized_query
    )

    print(
        "Last Context Size:",
        len(last_context)
    )

    # ------------------------
    # LANGGRAPH RUN
    # ------------------------

    result = (
        app.invoke(
            {
                "query":
                query,

                "documents":
                documents,

                "chunks":
                chunks,

                "chat_history":
                chat_history[-1:],

                "last_context":
                last_context
            }
        )
    )

    answer = (
        result.get(
            "answer",
            "No answer found."
        )
    )

    # ------------------------
    # FINAL ANSWER
    # ------------------------

    print(
        "\nFinal Answer:\n"
    )

    print(
        answer
    )

    # ------------------------
    # DEBUG INFO
    # ------------------------

    if (
        "confidence"
        in result
    ):

        print(
            f"\nConfidence:"
            f" {result['confidence']:.1f}%"
        )

    if (
        "score_gap"
        in result
    ):

        print(
            f"Score Gap:"
            f" {result['score_gap']:.2f}"
        )

    # ------------------------
    # SAVE TO CACHE
    # ------------------------

    cache.set(
        normalized_query,
        answer
    )

    print(
        "\nAnswer Cached!"
    )

    # ------------------------
    # SAVE CHAT HISTORY
    # ------------------------

    chat_history.append(
        {
            "user":
            query,

            "assistant":
            answer,

            "retrieved_docs":
            result.get(
                "refined_docs",
                []
            )
        }
    )

    print(
        "\nConversation History Saved!"
    )