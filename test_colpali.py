from models.ColPaliInfer import ColPaliInfer
from vector_retrieve.PDFProcessor import PDFProcessor
import fitz  # PyMuPDF

def main():
    # Initialize the inference class
    colpali_infer = ColPaliInfer()

    # Initialize the PDFProcessor
    pdf_processor = PDFProcessor()

    # Provide the path to your PDF file
    pdf_path = "/home/saket/Documents/Final User Undertaking for Remote Work.pdf"

    # Convert PDF pages to images
    print("Converting PDF to images...")
    page_images = pdf_processor.pdf_to_images(pdf_path)

    # Generate embeddings for each page image
    print("Generating embeddings for each page...")
    page_embeddings = pdf_processor.generate_embeddings(page_images)
    print(page_embeddings)

    # Define queries to test
    queries = [
        "What is the organizational structure?",
        "Who is the prime minister?",
        "Provide a summary of the financial performance."
    ]

    # Retrieve relevant pages for each query
    for query in queries:
        print(f"Evaluating query: {query}")
        relevant_pages = pdf_processor.retrieve_relevant_pages(page_embeddings, query)
        print(f"Relevant pages for query '{query}': {relevant_pages}")

def unit_colpalit():
    import torch
    from PIL import Image

    from colpali_engine.models import ColQwen2, ColQwen2Processor

    model_name = "vidore/colqwen2-v0.1"

    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
    ).eval()

    processor = ColQwen2Processor.from_pretrained(model_name)

    # Your inputs
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black"),
    ]
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the inputs
    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    # Forward pass
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)


if __name__ == "__main__":
    # main()
    unit_colpalit()

