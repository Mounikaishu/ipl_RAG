from langchain_core.documents import Document
from pathlib import Path
from docling.document_converter import DocumentConverter


class DoclingPDFLoader:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.converter = DocumentConverter()

    def load_documents(self):

        documents = []

        print("\nSearching PDFs in:")
        print(self.dataset_path)

        pdf_files = list(
            self.dataset_path.glob("*.pdf")
        )

        print("\nFound PDFs:")

        for pdf in pdf_files:

            print(pdf.name)

            try:

                result = (
                    self.converter.convert(
                        str(pdf)
                    )
                )

                text = (
                    result.document.export_to_markdown()
                )

                documents.append(

                    Document(
                        page_content=text,

                        metadata={

                            "source":
                            pdf.stem
                        }
                    )
                )

                print(
                    f"Loaded: {pdf.stem}"
                )

            except Exception as e:

                print(
                    f"Failed: {pdf.name}"
                )

                print(e)

        print(
            "\nTotal Documents Loaded:",
            len(documents)
        )

        return documents