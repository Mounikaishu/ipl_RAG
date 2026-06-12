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

from data_ingestion.docling_loader import (
    DoclingPDFLoader
)

from data_ingestion.prepare_chunks import (
    IPLChunkPreparer
)

from ipl_retrieval.vector_store import (
    IPLVectorStore
)

from memory.cache import (
    CacheMemory
)

from graph.workflow import (
    app
)


# ------------------------
# LOAD DATASET
# ------------------------

loader = (
    DoclingPDFLoader(
        dataset_path="dataset"
    )
)

documents = (
    loader.load_documents()
)

print(
    "\nLoaded documents:",
    len(documents)
)

if len(documents) == 0:

    print(
        "\nNo documents loaded!"
    )

    exit()


# ------------------------
# PREPARE CHUNKS
# ------------------------

chunk_preparer = (
    IPLChunkPreparer()
)

chunks = (
    chunk_preparer.prepare_chunks(
        documents
    )
)

print(
    "\nFinal chunks:",
    len(chunks)
)

if len(chunks) == 0:

    print(
        "\nNo chunks created!"
    )

    exit()


# ------------------------
# STORE EMBEDDINGS
# ------------------------

vector_store = (
    IPLVectorStore()
)

try:

    vector_store.store_documents(
        chunks
    )

    print(
        "\nIPL Vector DB Ready!"
    )

except Exception as e:

    print(
        "\nEmbedding Error:"
    )

    print(e)

    exit()


# ------------------------
# COMPONENTS
# ------------------------

cache = (
    CacheMemory()
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

    try:

        cached_answer = (
            cache.get(
                f"v2_{normalized_query}"
            )
        )

    except Exception:

        cached_answer = None

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
    # PREVIOUS CONTEXT
    # ------------------------

    last_context = ""

    retrieved_docs = []

    resolved_entities = []
    resolved_entity_type = None

    if len(
        chat_history
    ) > 0:

        last_context = (
            chat_history[-1]
            .get(
                "assistant",
                ""
            )[:500]
        )

        retrieved_docs = (
            chat_history[-1]
            .get(
                "retrieved_docs",
                []
            )
        )

        resolved_entities = (
            chat_history[-1]
            .get(
                "resolved_entities",
                []
            )
        )

        resolved_entity_type = (
            chat_history[-1]
            .get(
                "resolved_entity_type",
                None
            )
        )

    print(
        "\nOriginal Query:",
        normalized_query
    )

    print(
        "Last Context Size:",
        len(
            last_context
        )
    )

    print(
        "Previous Saved Entities:",
        resolved_entities
    )

    # ------------------------
    # LANGGRAPH RUN
    # ------------------------

    try:

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
                    chat_history[-3:],

                    "last_context":
                    last_context,

                    "previous_docs":
                    retrieved_docs,

                    "resolved_entities":
                    resolved_entities,

                    "resolved_entity_type":
                    resolved_entity_type
                }
            )
        )

    except Exception as e:

        print(
            "\nGraph Error:"
        )

        print(e)

        continue

    print(
        "\nGraph Result Keys:",
        result.keys()
    )

    answer = (
        result.get(
            "answer",
            "No answer found."
        )
    )

    entities = (
        result.get(
            "resolved_entities",
            []
        )
    )

    print(
        "\nSaving Entities:",
        entities
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

    try:

        cache.set(
            f"v2_{normalized_query}",
            answer
        )

        print(
            "\nAnswer Cached!"
        )

    except Exception:

        print(
            "\nCache skipped."
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
                "retrieved_docs",
                []
            ),

            "resolved_entities":
            entities,

            "resolved_entity_type":
            result.get("resolved_entity_type")
        }
    )

    print(
        "\nConversation History Saved!"
    )