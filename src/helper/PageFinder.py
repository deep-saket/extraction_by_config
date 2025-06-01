from vector_retrieve import PDFProcessor
from common import CallableComponent

class PageFinder(CallableComponent):
    """
    Uses ColPali (or any retrieval backend) to find relevant pages for a query.
    """
    def __init__(self, pdf_processor: PDFProcessor) -> None:
        self.pdf_processor = pdf_processor

    def retrieve_pages(self, embeddings, field_name, description):
        query = f"{field_name}: {description}"
        return self.pdf_processor.retrieve_relevant_pages(embeddings, query)

    def __call__(self, embeddings, field_name, description):
        return self.retrieve_pages(embeddings, field_name, description)