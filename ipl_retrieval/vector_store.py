
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


from langchain_chroma import (
    Chroma
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)


class IPLVectorStore:

    def __init__(
        self,
        persist_directory=
        "./chroma_db"
    ):

        self.persist_directory = (
            persist_directory
        )

        self.embeddings = (
            HuggingFaceEmbeddings(
                model_name=
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    def store_documents(
        self,
        chunks
    ):

        # ------------------------
        # LOAD EXISTING DB
        # ------------------------

        vector_store = (
            Chroma(
                persist_directory=
                self.persist_directory,

                embedding_function=
                self.embeddings
            )
        )

        # ------------------------
        # CLEAR OLD EMBEDDINGS
        # ------------------------

        existing = (
            vector_store.get()
        )

        if (
            existing["ids"]
        ):

            vector_store.delete(
                ids=
                existing["ids"]
            )

            print(
                "\nOld embeddings cleared!"
            )

        # ------------------------
        # ADD NEW DOCUMENTS
        # ------------------------

        vector_store.add_documents(
            chunks
        )

        print(
            "\nEmbeddings stored!"
        )

        return vector_store

