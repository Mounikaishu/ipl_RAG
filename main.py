
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

print(framework_path)


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

from retrieval.hybrid_retriever import (
    HybridRetriever
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

from query_processing.rewrite import (
    QueryRewriter
)

from rerank.reranker import (
    Reranker
)

from refine.refiner import (
    Refiner
)


# ------------------------
# LOAD DATASET
# ------------------------

loader = DatasetLoader()

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

retriever = (
    HybridRetriever(
        persist_directory=
        "chroma_db"
    )
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

    # ------------------------
    # REWRITE
    # ------------------------

    rewritten_query = (
        rewriter.rewrite(
            query
        )
    )

    print(
        "\nRewritten Query:",
        rewritten_query
    )

    # ------------------------
    # ROUTING
    # ------------------------

    route = (
        router.route(
            rewritten_query
        )
    )

    print(
        "\nRoute Chosen:",
        route
    )

    # ------------------------
    # DIRECT WEB SEARCH
    # ------------------------

    if (
        route
        == "WEB_SEARCH"
    ):

        web_result = (
            web_tool.search(
                rewritten_query
            )
        )

        answer = (
            generator
            .summarize_web_answer(
                rewritten_query,
                web_result
            )
        )

        print(
            "\nFinal Answer:\n"
        )

        print(answer)

        continue

    # ------------------------
    # HYBRID RETRIEVAL
    # ------------------------

    retrieved_docs = (
        retriever.retrieve(
            query=
            rewritten_query,
            chunks=
            chunks,
            k=5
        )
    )

    print(
        "\nRetrieved Docs:",
        len(
            retrieved_docs
        )
    )

    # ------------------------
    # RERANK
    # ------------------------

    rerank_result = (
        reranker.rerank(
            rewritten_query,
            retrieved_docs
        )
    )

    reranked_docs = (
        rerank_result[
            "docs"
        ]
    )

    best_score = (
        rerank_result[
            "best_score"
        ]
    )

    confidence = (
        rerank_result[
            "confidence"
        ]
    )

    score_gap = (
        rerank_result[
            "score_gap"
        ]
    )

    print(
        "\nAfter Rerank:",
        len(
            reranked_docs
        )
    )

    print(
        f"\nBest Score:"
        f" {best_score:.2f}"
    )

    print(
        f"Confidence:"
        f" {confidence:.1f}%"
    )

    print(
        f"Score Gap:"
        f" {score_gap:.2f}"
    )

    # ------------------------
    # REFINE
    # ------------------------

    refined_docs = (
        refiner.refine(
            reranked_docs
        )
    )

    print(
        "\nAfter Refine:",
        len(
            refined_docs
        )
    )

    # ------------------------
    # LOW CONFIDENCE
    # ------------------------

    if confidence < 8:

        print(
            "\nLow confidence."
            " Switching to"
            " WEB SEARCH..."
        )

        web_result = (
            web_tool.search(
                rewritten_query
            )
        )

        answer = (
            generator
            .summarize_web_answer(
                rewritten_query,
                web_result
            )
        )

    # ------------------------
    # HIGH CONFIDENCE
    # ------------------------

    else:

        answer = (
            generator
            .generate_answer(
                query=
                rewritten_query,
                retrieved_docs=
                refined_docs
            )
        )

    # ------------------------
    # FINAL ANSWER
    # ------------------------

    print(
        "\nFinal Answer:\n"
    )

    print(answer)

    print(
        f"\nConfidence:"
        f" {confidence:.1f}%"
    )

    print(
        f"Score Gap:"
        f" {score_gap:.2f}"
    )
