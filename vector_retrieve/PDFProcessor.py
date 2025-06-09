import os
import fitz  # PyMuPDF
from PIL import Image
import torch
from common import CallableComponent, ExtractionState


class PDFProcessor(CallableComponent):
    """
    A class for processing PDFs and generating embeddings for their pages.
    
    This class extracts pages as images from a PDF, saves them to a temporary directory,
    generates embeddings for those images using the ColPali model, and retrieves the most 
    relevant pages based on a text query.
    """
    def __init__(self, colpali_infer, checkbox_infer=None, override=False):
        """
        Initializes the PDFProcessor by creating an instance of ColPaliInfer.
        """
        super().__init__()
        self.colpali_infer = colpali_infer
        self.override = override
        self.checkbox_infer = checkbox_infer

    def __call__(self, pdf_path: str):
        # populate state
        ExtractionState.images = self.pdf_to_images(pdf_path)
        ExtractionState.embeddings = self.generate_embeddings(ExtractionState.images)
        if ExtractionState.extraction_items.has_checkbox_items():
            ExtractionState.checkboxes = self.process_checkboxes(ExtractionState.images)

    def process_checkboxes(self, images):
        """
        Process images to detect checkboxes if checkbox items are present in extraction items.
        
        Args:
            images (list): List of tuples containing page number and image path
            
        Returns:
            dict: Dictionary with page numbers as keys and checkbox details as values
        """
        checkboxes = {}
        if self.checkbox_infer:
            for page_num, image_path in images:
                checkbox_list = self.checkbox_infer.infer(image_data=image_path)
                if checkbox_list:
                    checkboxes[page_num] = {
                        'image_path': image_path,
                        'checkbox_count': len(checkbox_list),
                        'checkbox_data': checkbox_list,
                    }
        return checkboxes

    
    def pdf_to_images(self, pdf_path):
        """
        Converts a PDF into a list of images, one per page, and saves them to a temporary directory.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            list of tuples: Each tuple contains the page number (int) and the file path (str) to the saved image.
        """
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        # Open the PDF
        doc = fitz.open(pdf_path)
        images = []

        self.logger.info(f"total number of pages in PDF: {len(doc)}")
        # Create a unique temporary directory using uuid
        tmp_dir = os.path.join("./tmp", pdf_name)

        if not self.override and os.path.exists(tmp_dir):
            for page_num, page in enumerate(doc):
                file_path = os.path.join(tmp_dir, f"page_{page_num + 1}.png")
                images.append((page_num + 1, file_path))
            return images

        os.makedirs(tmp_dir, exist_ok=True)
        # Process each page in the PDF
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            file_path = os.path.join(tmp_dir, f"page_{page_num + 1}.png")
            pix.save(file_path)
            images.append((page_num + 1, file_path))
        return images

    def generate_embeddings(self, images):
        """
        Generates embeddings for a list of page images stored as file paths.

        Args:
            images (list of tuples): Each tuple contains a page number (int) and an image file path (str).

        Returns:
            list of tuples: Each tuple contains the page number (int) and its embedding (torch.Tensor).
        """
        embeddings = []
        for page_num, image_path in images:
            # Open the image file
            image = Image.open(image_path).convert("RGB")
            # Get the image embedding using ColPaliInfer
            if not self.override:
                pickle_path = image_path.replace('.png', '.pkl')
                if os.path.exists(pickle_path):
                    with open(pickle_path, 'rb') as f:
                        embedding = torch.load(f)
                    embeddings.append((page_num, embedding.cpu()))
                    continue

            embedding = self.colpali_infer.get_image_embedding(image)
            # If embedding has shape (1, embed_dim), squeeze to (embed_dim,)
            if embedding.dim() == 2 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)

            # Save embedding as pickle file
            pickle_path = image_path.replace('.png', '.pkl')
            with open(pickle_path, 'wb') as f:
                torch.save(embedding, f)

            embeddings.append((page_num, embedding.cpu()))

        return embeddings

    def retrieve_relevant_pages(self, embeddings, query, top_k=3):
        """
        Retrieves the top relevant pages for a given query using similarity scores.

        This method generates a text embedding for the query, stacks the page embeddings,
        and uses the processor's `score_multi_vector` method to compute similarity scores.

        Args:
            embeddings (list of tuples): Each tuple contains the page number (int) and its embedding (torch.Tensor).
            query (str): The text query to evaluate.
            top_k (int): The number of top relevant pages to return.

        Returns:
            list of int: The page numbers of the most relevant pages, sorted by relevance.
        """
        # Get the query embedding (shape: (1, embed_dim))
        query_embedding = self.colpali_infer.get_text_embedding(query)
        if query_embedding.dim() == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding.squeeze(0).unsqueeze(0)  # Ensure shape remains (1, embed_dim)
        
        # Stack all page embeddings to form a tensor of shape (num_pages, embed_dim)
        page_embeddings = torch.stack([emb for _, emb in embeddings], dim=0).to(self.colpali_infer.model.device)
        page_embeddings = page_embeddings.squeeze(1)

        # Compute similarity scores using the processor's score_multi_vector method.
        # Expected shape for a single query: (1, num_pages)
        scores = self.colpali_infer.processor.score_multi_vector(query_embedding, page_embeddings)
        scores = scores.squeeze(0)  # Now shape: (num_pages,)

        # Sort scores in descending order and get indices of top_k scores.
        sorted_scores, indices = torch.sort(scores, descending=True)
        top_indices = indices[:top_k]

        # Map indices back to the corresponding page numbers.
        relevant_pages = [embeddings[i][0] for i in top_indices]
        return relevant_pages
