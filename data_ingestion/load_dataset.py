import sys
from pathlib import Path

from langchain_core.documents import (
    Document
)

from data_ingestion.docling_loader import (
    DoclingPDFLoader
)


class DatasetLoader:

    def __init__(
        self,
        dataset_path="./dataset"
    ):

        self.dataset_path = (
            Path(dataset_path)
        )

        self.loader = (
            DoclingPDFLoader()
        )

    def load_documents(
        self
    ):

        all_documents = []

        pdf_files = list(
            self.dataset_path.glob(
                "*.pdf"
            )
        )

        print(
            "\nFound PDFs:"
        )

        for pdf in pdf_files:

            print(
                pdf.name
            )

            try:

                # -----------------
                # LOAD PDF
                # -----------------

                text = (
                    self.loader
                    .load_pdf(
                        str(pdf)
                    )
                )

                docs = [
                    Document(
                        page_content=
                        text,

                        metadata={}
                    )
                ]

                # -----------------
                # SOURCE METADATA
                # -----------------

                source_name = (
                    pdf.stem
                )

                for doc in docs:

                    if (
                        doc.metadata
                        is None
                    ):

                        doc.metadata = {}

                    doc.metadata[
                        "source"
                    ] = source_name

                all_documents.extend(
                    docs
                )

                print(
                    f"Loaded: {source_name}"
                )

            except Exception as e:

                print(
                    f"\nFailed: {pdf.name}"
                )

                print(
                    "Error:",
                    e
                )

        print(
            "\nTotal Documents Loaded:",
            len(
                all_documents
            )
        )

        return all_documents

