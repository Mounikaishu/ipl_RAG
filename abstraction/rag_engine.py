import sys
from pathlib import Path


# --------------------------------
# CONNECT FRAMEWORK REPO
# --------------------------------

framework_path = (
    Path(__file__)
    .resolve()
    .parent.parent.parent
    / "rag-pipeline-framework"
)

sys.path.append(
    str(framework_path)
)


# --------------------------------
# FRAMEWORK IMPORTS
# --------------------------------

from retrieval.hybrid_retriever import (
    HybridRetriever
)

from rerank.reranker import (
    Reranker
)

from refine.refiner import (
    Refiner
)

from generation.generator import (
    Generator
)

from rewrite.query_rewriter import (
    QueryRewriter
)


# --------------------------------
# RAG ENGINE
# --------------------------------

class RAGEngine:

    def __init__(
        self,
        persist_directory
    ):

        self.retriever = (
            HybridRetriever(
                persist_directory=
                persist_directory
            )
        )

        self.reranker = (
            Reranker()
        )

        self.refiner = (
            Refiner()
        )

        self.generator = (
            Generator()
        )

        self.rewriter = (
            QueryRewriter()
        )

    def ask(
        self,
        query,
        chunks,
        k=10
    ):

        # -------------------
        # QUERY REWRITE
        # -------------------

        rewritten_query = (
            self.rewriter
            .rewrite(query)
        )

        print(
            "\nRewritten Query:",
            rewritten_query
        )

        # -------------------
        # RETRIEVE
        # -------------------

        retrieved_docs = (
            self.retriever
            .retrieve(
                query=
                rewritten_query,

                chunks=
                chunks,

                k=k
            )
        )

        print(
            "\nRetrieved docs:",
            len(retrieved_docs)
        )

        # -------------------
        # RERANK
        # -------------------

        reranked_docs = (
            self.reranker
            .rerank(
                query=
                rewritten_query,

                retrieved_docs=
                retrieved_docs
            )
        )

        print(
            "\nReranked docs:",
            len(reranked_docs)
        )

        # -------------------
        # REFINE
        # -------------------

        refined_docs = (
            self.refiner
            .refine(
                reranked_docs
            )
        )

        print(
            "\nRefined docs:",
            len(refined_docs)
        )

        # -------------------
        # GENERATE
        # -------------------

        answer = (
            self.generator
            .generate_answer(
                query=
                rewritten_query,

                retrieved_docs=
                refined_docs
            )
        )

        return answer