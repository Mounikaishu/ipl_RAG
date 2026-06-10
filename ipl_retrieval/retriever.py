from langchain_chroma import (
    Chroma
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)


class IPLRetriever:

    def __init__(
        self,
        persist_directory=
        "./chroma_db"
    ):

        self.embeddings = (
            HuggingFaceEmbeddings(
                model_name=
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

        self.vector_store = (
            Chroma(
                persist_directory=
                persist_directory,

                embedding_function=
                self.embeddings
            )
        )

    def get_metadata_filter(
        self,
        query
    ):

        query_lower = (
            query.lower()
        )

        if any(
            word in query_lower
            for word in [
                "runs",
                "strike rate",
                "batting",
                "century",
                "fifty"
            ]
        ):

            return {
                "source":
                "batting_stats"
            }

        elif any(
            word in query_lower
            for word in [
                "wickets",
                "economy",
                "bowling"
            ]
        ):

            return {
                "source":
                "bowling_stats"
            }

        elif any(
            word in query_lower
            for word in [
                "vs",
                "head to head",
                "h2h"
            ]
        ):

            return {
                "source":
                "h2h_records"
            }

        elif any(
            word in query_lower
            for word in [
                "recent form",
                "last 5",
                "form"
            ]
        ):

            return {
                "source":
                "recent_form"
            }

        elif any(
            word in query_lower
            for word in [
                "venue",
                "pitch"
            ]
        ):

            return {
                "source":
                "venue_reports"
            }

        return None

    def retrieve(
        self,
        query,
        k=3
    ):

        metadata_filter = (
            self.get_metadata_filter(
                query
            )
        )

        print(
            "\nMetadata Filter:",
            metadata_filter
        )

        docs = (
            self.vector_store
            .similarity_search(
                query=
                query,

                k=k,

                filter=
                metadata_filter
            )
        )

        print(
            "\nRetrieved Docs:",
            len(docs)
        )

        return docs

