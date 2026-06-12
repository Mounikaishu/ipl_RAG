from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from config.settings import Settings


class IPLHybridRetriever:

    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=getattr(Settings, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
    def retrieve(self, query, chunks, k=10, metadata_filter=None):
        print(f"\n[IPLHybridRetriever] Query: {query}")
        print(f"[IPLHybridRetriever] Filter: {metadata_filter}")

        # -------------------
        # VECTOR SEARCH WITH METADATA FILTER
        # -------------------
        try:
            chroma_filter = None
            if metadata_filter:
                if len(metadata_filter) == 1:
                    chroma_filter = metadata_filter
                else:
                    chroma_filter = {
                        "$and": [
                            {k: v}
                            for k, v in metadata_filter.items()
                        ]
                    }

            vector_results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=chroma_filter
            )
            vector_docs = [doc for doc, score in vector_results]
            vector_scores = {doc.page_content: (1 / (rank + 1)) for rank, doc in enumerate(vector_docs)}
            print(f"[IPLHybridRetriever] Vector retrieved: {len(vector_docs)}")
        except Exception as e:
            print(f"[IPLHybridRetriever] Vector search error: {e}")
            vector_docs = []
            vector_scores = {}

        # -------------------
        # BM25 SEARCH (ONLY ON FILTERED CHUNKS FOR PRECISION)
        # -------------------
        filtered_chunks = chunks
        if metadata_filter:
            filtered_chunks = []
            for chunk in chunks:
                match = True
                for key, val in metadata_filter.items():
                    if chunk.metadata.get(key) != val:
                        match = False
                        break
                if match:
                    filtered_chunks.append(chunk)

        print(f"[IPLHybridRetriever] BM25 Candidates: {len(filtered_chunks)}")

        bm25_docs = []
        if filtered_chunks:
            try:
                tokenized_chunks = [doc.page_content.lower().split() for doc in filtered_chunks]
                bm25 = BM25Okapi(tokenized_chunks)
                tokenized_query = query.lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)
                ranked_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )[:k]
                bm25_docs = [filtered_chunks[i] for i in ranked_indices]
                print(f"[IPLHybridRetriever] BM25 retrieved: {len(bm25_docs)}")
            except Exception as e:
                print(f"[IPLHybridRetriever] BM25 error: {e}")

        # -------------------
        # MERGE & RRF SCORE
        # -------------------
        combined_docs = vector_docs + bm25_docs
        unique_docs = []
        seen = set()
        for doc in combined_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        combined_scores = {}
        for rank, doc in enumerate(unique_docs):
            text = doc.page_content
            vector_score = vector_scores.get(text, 0)
            
            bm25_score = 0
            if doc in bm25_docs:
                bm25_rank = bm25_docs.index(doc)
                bm25_score = 1 / (bm25_rank + 1)

            # Reciprocal Rank Fusion weights
            final_score = 0.7 * vector_score + 0.3 * bm25_score
            combined_scores[text] = final_score

        # Sort documents by score
        final_docs = sorted(
            unique_docs,
            key=lambda doc: combined_scores.get(doc.page_content, 0),
            reverse=True
        )[:k]

        return final_docs
