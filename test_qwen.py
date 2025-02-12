import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from io import BytesIO
import requests

# Import the QwenV25Infer class (ensure it's in the same directory or properly imported)
from models.QwenV25Infer import QwenV25Infer

def load_image_from_path(image_path):
    """Load image data from a local file path."""
    with open(image_path, 'rb') as f:
        return f.read()

def load_image_from_url(image_url):
    """Load image data from a URL."""
    response = requests.get(image_url)
    response.raise_for_status()
    return response.content

def main():
    # Configuration
    use_api = False  # Set to True to use API inference
    image_path = '/home/saket/Documents/photo.png'  # Path to your local image file
    prompt = "Describe the image."

    if use_api:
        # API inference configuration
        api_endpoint = 'https://your-endpoint-url'  # Replace with your endpoint URL
        api_token = 'your-huggingface-api-token'  # Replace with your Hugging Face API token
        inferer = QwenV25Infer(api_endpoint=api_endpoint, api_token=api_token)
    else:
        # Local inference configuration
        model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'  # Replace with your model name
        inferer = QwenV25Infer(model_name=model_name, device='cpu')

    # Load image data
    with open(image_path, 'rb') as f:
        image_data = f.read()

    # Perform inference
    result = inferer.infer(image_data, prompt)
    print("Inference Result:", result)


if __name__ == "__main__":
    main()
