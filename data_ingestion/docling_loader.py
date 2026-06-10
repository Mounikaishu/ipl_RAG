from docling.document_converter import (
    DocumentConverter
)


class DoclingPDFLoader:

    def __init__(self):

        self.converter = (
            DocumentConverter()
        )

    def load_pdf(
        self,
        file_path
    ):

        result = (
            self.converter
            .convert(
                file_path
            )
        )

        document = (
            result.document
        )

        markdown_text = (
            document.export_to_markdown()
        )

        return markdown_text

