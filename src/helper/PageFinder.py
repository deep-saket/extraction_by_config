from typing import List, Optional
from common import CallableComponent, ExtractionState
from vector_retrieve import PDFProcessor
from extraction_io.ExtractionItems import ExtractionItem


class PageFinder(CallableComponent):
    """
    Uses only embedding‐based retrieval to find relevant pages for a query.
    If ExtractionItem.search_keys is nonempty, we join those phrases into a single string
    and send that to the embedding retriever. Otherwise, we default to "field_name + description".
    """

    def __init__(self, pdf_processor: PDFProcessor) -> None:
        super().__init__()
        self.pdf_processor = pdf_processor

    def retrieve_pages(
        self,
        embeddings: List[tuple[int, any]],
        extraction_item: ExtractionItem
    ) -> List[int]:
        """
        Given all page embeddings and a validated ExtractionItem,
        decide on an embedding‐query string, then return the top‐k pages.
        """
        # Determine the embedding query
        if extraction_item.search_keys:
            # Join all search_keys into one query string
            query = " ".join(extraction_item.search_keys)
            self.logger.info(f"[PageFinder] Using search_keys: {extraction_item.search_keys}")
        else:
            # Fallback to combining field_name and description
            query = f"{extraction_item.field_name}: {extraction_item.description}"
            self.logger.info(f"[PageFinder] Using default embedding query: '{query}'")

        # Use PDFProcessor to retrieve the most relevant pages
        return self.pdf_processor.retrieve_relevant_pages(embeddings, query)

    def __call__(
        self,
        embeddings: Optional[List[tuple[int, any]]] = None,
        extraction_item: Optional[ExtractionItem] = None
    ) -> List[int]:
        """
        Entry point:
          1) Fetch embeddings from state if not passed in.
          2) Fetch current ExtractionItem from state if not passed in.
          3) Raise an error if either is missing.
          4) Return the list of relevant page numbers.
        """
        if embeddings is None:
            embeddings = ExtractionState.get_embeddings()
        if extraction_item is None:
            extraction_item = ExtractionState.get_current_extraction_item()

        if not embeddings or extraction_item is None:
            raise ValueError(
                "PageFinder requires both embeddings and a current extraction_item."
            )

        return self.retrieve_pages(embeddings, extraction_item)