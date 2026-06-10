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

from graph.workflow import (
    app
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

    result = app.invoke(

        {

            "query":
            query,

            "chunks":
            chunks
        }
    )

    print(
        "\nFinal Answer:\n"
    )

    print(
        result[
            "answer"
        ]
    )

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