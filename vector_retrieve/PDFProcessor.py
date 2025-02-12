import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
from models.ColPaliInfer import ColPaliInfer
import torch

class PDFProcessor:
    """
    A class for processing PDFs and generating embeddings for their pages.
    
    This class extracts pages as images from a PDF, generates embeddings for those images,
    and retrieves relevant pages based on a text query using embeddings.
    """
    def __init__(self):
        """
        Initializes the PDFProcessor class.

        This involves setting up the ColPaliInfer instance for handling embeddings.
        """
        self.colpali_infer = ColPaliInfer()

    def pdf_to_images(self, pdf_path):
        """
        Converts a PDF into a list of images, one for each page.

        Args:
            pdf_path (str): The file path of the PDF to process.

        Returns:
            list of tuples: Each tuple contains the page number (int) and the image data (bytes).
        """
        # Open the PDF and convert each page to an image
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes()
            images.append((page_num + 1, img_data))
        return images

    def generate_embeddings(self, images):
        """
        Generates embeddings for a list of images.

        Args:
            images (list of tuples): A list where each tuple contains the page number (int)
                                     and the image data (bytes).

        Returns:
            list of tuples: Each tuple contains the page number (int) and its embedding (torch.Tensor).
        """
        # Generate embeddings for each page image
        embeddings = []
        for page_num, img_data in images:
            image = Image.open(BytesIO(img_data)).convert("RGB")
            embedding = self.colpali_infer.get_image_embedding(image)
            embeddings.append((page_num, embedding))
        return embeddings

    def retrieve_relevant_pages(self, embeddings, query, top_k=3):
        """
        Retrieves the top relevant pages for a given query based on embeddings.

        Args:
            embeddings (list of tuples): A list where each tuple contains the page number (int)
                                         and its embedding (torch.Tensor).
            query (str): The query to match against the page embeddings.
            top_k (int): The number of top relevant pages to retrieve.

        Returns:
            list of int: The page numbers of the most relevant pages, sorted by relevance.
        """
        # Generate an embedding for the query text
        query_embedding = self.colpali_infer.get_text_embedding(query)
        print('query_embedding.shape', query_embedding.shape)
        print('embeddings =', embeddings)
        print('embeddings.shape', embeddings[0][1].shape)

        # Compute similarity scores between the query and each page embedding
        scores = []
        for page_num, page_embedding in embeddings:
            score = torch.matmul(query_embedding, page_embedding.T).item()
            scores.append((page_num, score))

        # Sort scores by relevance and return the top page numbers
        scores.sort(key=lambda x: x[1], reverse=True)
        relevant_pages = [page_num for page_num, _ in scores[:top_k]]
        return relevant_pages
