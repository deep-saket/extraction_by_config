from models.ColPaliInfer import ColPaliInfer
from vector_retrieve.PDFProcessor import PDFProcessor
import fitz  # PyMuPDF

def main():
    # Initialize the PDFProcessor
    pdf_processor = PDFProcessor()

    # Provide the path to your PDF file
    pdf_path = "/home/saket/Documents/2028048054.Owner_Policy.1054423.pdf"

    # Convert PDF pages to images
    print("Converting PDF to images...")
    page_images = pdf_processor.pdf_to_images(pdf_path)

    # Generate embeddings for each page image
    print("Generating embeddings for each page...")
    page_embeddings = pdf_processor.generate_embeddings(page_images)

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

if __name__ == "__main__":
    print('*' * 50)
    main()

