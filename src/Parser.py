import json
from models.model_manager import ModelManager
from vector_and_retrieve.pdf_processor import PDFProcessor

class Parser:
    def __init__(self, extraction_config):
        """
        Initializes the Parser with the extraction configuration.

        Args:
            extraction_config (list): A list of dictionaries containing field extraction details.
        """
        self.pdf_processor = PDFProcessor()
        self.extraction_config = extraction_config
        self.qwen_infer = ModelManager.qwen_infer

    def perform_de(self, pdf_path, output_json_path):
        """
        Performs document extraction on the given PDF and saves the results to a JSON file.

        Args:
            pdf_path (str): Path to the input PDF file.
            output_json_path (str): Path to save the extracted data as a JSON file.
        """
        images = self.pdf_processor.pdf_to_images(pdf_path)
        embeddings = self.pdf_processor.generate_embeddings(images)
        extracted_data = {}

        for field in self.extraction_config:
            field_name = field["field_name"]
            description = field["description"]
            probable_pages = field.get("probable_pages", [])
            extraction_type = field["type"]

            if probable_pages:
                field_values = self._process_specified_pages(images, probable_pages, field_name, description, extraction_type)
            else:
                field_values = self._retrieve_with_colpali(embeddings, images, field_name, description, extraction_type)

            if extraction_type == "key_value":
                extracted_data[field_name] = field_values[0] if field_values else None
            else:
                extracted_data[field_name] = field_values

        self._create_json(extracted_data, output_json_path)

    def _process_specified_pages(self, images, pages, field_name, description, extraction_type):
        """
        Processes specified pages to extract the desired field.

        Args:
            images (list): List of tuples containing page numbers and image data.
            pages (list): List of page numbers to process.
            field_name (str): The name of the field to extract.
            description (str): Description of the field.
            extraction_type (str): The type of extraction (e.g., "key_value", "bullet_points").

        Returns:
            list: Extracted content from the specified pages.
        """
        field_values = []
        for page_num, image_data in images:
            if page_num in pages:
                prompt = f"Extract the {field_name} ({description}) from page {page_num} as {extraction_type}."
                content = self.qwen_infer.infer(image_data, prompt)
                field_values.append(content)
        return field_values

    def _retrieve_with_colpali(self, embeddings, images, field_name, description, extraction_type):
        """
        Uses ColPali to retrieve relevant pages and extract the desired field.

        Args:
            embeddings (list): List of tuples containing page numbers and embeddings.
            images (list): List of tuples containing page numbers and image data.
            field_name (str): The name of the field to extract.
            description (str): Description of the field.
            extraction_type (str): The type of extraction (e.g., "key_value", "bullet_points").

        Returns:
            list: Extracted content from the retrieved pages.
        """
        query = f"{field_name}: {description}"
        relevant_pages = self.pdf_processor.retrieve_relevant_pages(embeddings, query)
        field_values = []
        for page_num in relevant_pages:
            image_data = next((img for num, img in images if num == page_num), None)
            if image_data:
                prompt = f"Extract the {field_name} ({description}) from page {page_num} as {extraction_type}."
                content = self.qwen_infer.infer(image_data, prompt)
                field_values.append(content)
        return field_values

    def _create_json(self, extracted_data, output_path):
        """
        Saves the extracted data to a JSON file.

        Args:
            extracted_data (dict): The data extracted from the PDF.
            output_path (str): Path to save the JSON file.
        """
        with open(output_path, 'w') as json_file:
            json.dump(extracted_data, json_file, indent=4)
