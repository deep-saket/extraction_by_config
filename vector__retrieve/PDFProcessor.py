import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
from models.ModelManager import ModelManager

class PDFProcessor:
    def __init__(self):
        self.colpali_infer = ModelManager.colpali_infer

    def pdf_to_images(self, pdf_path):
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes()
            images.append((page_num + 1, img_data))
        return images

    def generate_embeddings(self, images):
        embeddings = []
        for page_num, img_data in images:
            image = Image.open(BytesIO(img_data)).convert("RGB")
            embedding = self.colpali_infer.get_image_embedding(image)
            embeddings.append((page_num, embedding.cpu()))
        return embeddings

    def retrieve_relevant_pages(self, embeddings, query, top_k=3):
        query_embedding = self.colpali_infer.get_text_embedding(query)
        scores = []
        for page_num, page_embedding in embeddings:
            score = torch.matmul(query_embedding, page_embedding.T).item()
            scores.append((page_num, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        relevant_pages = [page_num for page_num, _ in scores[:top_k]]
        return relevant_pages
