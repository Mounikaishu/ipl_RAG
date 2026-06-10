class IPLChunkPreparer:

    def __init__(self):

        pass

    def prepare_chunks(
        self,
        documents
    ):

        cleaned_docs = []

        # -------------------
        # REMOVE INSTRUCTION NOISE
        # -------------------

        noise_patterns = [
            "LangGraph Node",
            "This node",
            "This is a multi-document node",
            "This is a key LangGraph idea",
            "Students must implement",
            "ValidationNode",
            "Implementation Hint",
            "The graph must route",
            "ComparisonNode",
            "downstream"
            "This is a key LangGraph insight",
            "metadata-filtered retrieval",
            "Chunks from",
            "must have metadata tag",
        ]

        for doc in documents:

            content = (
                doc.page_content
            )

            cleaned_lines = []

            for line in (
                content.split("\n")
            ):

                should_skip = any(
                    pattern.lower()
                    in line.lower()
                    for pattern
                    in noise_patterns
                )

                if (
                    not should_skip
                ):

                    cleaned_lines.append(
                        line
                    )

            cleaned_text = (
                "\n".join(
                    cleaned_lines
                ).strip()
            )

            # -------------------
            # UPDATE DOCUMENT
            # -------------------

            doc.page_content = (
                cleaned_text
            )

            source = (
                doc.metadata.get(
                    "source",
                    "unknown"
                )
            )

            doc.metadata[
                "dataset_type"
            ] = source

            cleaned_docs.append(
                doc
            )

        print(
            "\nPrepared chunks:",
            len(cleaned_docs)
        )

        return cleaned_docs

