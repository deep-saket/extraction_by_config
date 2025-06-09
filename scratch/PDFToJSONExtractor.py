import io
import os
import json
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF for PDF rendering into images  [oai_citation:1‡youtube.com](https://www.youtube.com/watch?v=9gDJ6PhvVck&utm_source=chatgpt.com)
from PIL import Image  # PIL for in-memory image handling (convert Pixmap bytes to PIL.Image)  [oai_citation:2‡python.langchain.com](https://python.langchain.com/docs/integrations/document_loaders/docling/?utm_source=chatgpt.com)
import torch  # PyTorch for model inference
from transformers import AutoProcessor, AutoModelForVision2Seq  # SmolDocling APIs

# Import DoclingDocument to parse DocTags into JSON
from docling_core.types.doc import DoclingDocument # type: ignore
from docling_core.types.doc.document import DocTagsDocument


class PDFSmolDoclingExtractor:
    """
    Convert each PDF page to a JSON structure by:
      1. Rendering each page to a PIL.Image via PyMuPDF (fitz).  [oai_citation:3‡youtube.com](https://www.youtube.com/watch?v=9gDJ6PhvVck&utm_source=chatgpt.com)
      2. Running SmolDocling-256M on each image to generate DocTags markup.  [oai_citation:4‡github.com](https://github.com/docling-project/docling?utm_source=chatgpt.com)
      3. Parsing the DocTags into a JSON object using Docling’s API.
      4. Aggregating results into {"num_pages": N, "pages": [{"page_number": i, "page_json": {...}}, ...]}.
    """

    def __init__(self,
                 pdf_path: str,
                 device: Optional[str] = None):
        """
        Args:
          pdf_path: Path to the input PDF (hardcoded below).
          device:   "cuda" or "cpu"; if None, auto-detects GPU if available.
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        self.pdf_path = pdf_path

        # Determine PyTorch device: GPU if available, else CPU.
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load SmolDocling-256M processor & model from Hugging Face.  [oai_citation:5‡github.com](https://github.com/docling-project/docling?utm_source=chatgpt.com)
        self.model_id = "ds4sd/SmolDocling-256M-preview"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            _attn_implementation="flash_attention_2" if self.device.type == "cuda" else "eager"
        )
        self.model.eval()
        self.model.to(self.device)

        # Placeholder: list of PIL.Images, one per PDF page.  [oai_citation:6‡youtube.com](https://www.youtube.com/watch?v=9gDJ6PhvVck&utm_source=chatgpt.com)
        self.page_images: List[Image.Image] = []

        # Placeholder: final results dict after extraction.
        self.results: Dict[str, Any] = {}

    def _convert_pdf_to_images(self) -> None:
        """
        Open the PDF with fitz and render each page to a PIL.Image in memory.
        Uses matrix=fitz.Matrix(2,2) for ~300 DPI equivalent.  [oai_citation:7‡youtube.com](https://www.youtube.com/watch?v=9gDJ6PhvVck&utm_source=chatgpt.com)
        """
        pdf_document = fitz.open(self.pdf_path)  # Load PDF  [oai_citation:8‡youtube.com](https://www.youtube.com/watch?v=9gDJ6PhvVck&utm_source=chatgpt.com)
        self.page_images = []

        for page_idx in range(pdf_document.page_count):
            page = pdf_document.load_page(page_idx)  # 0-based index  [oai_citation:9‡youtube.com](https://www.youtube.com/watch?v=9gDJ6PhvVck&utm_source=chatgpt.com)
            # 2× zoom in both dimensions → ~300 DPI  [oai_citation:10‡github.com](https://github.com/docling-project/docling?utm_source=chatgpt.com) [oai_citation:11‡youtube.com](https://www.youtube.com/watch?v=9gDJ6PhvVck&utm_source=chatgpt.com)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")  # Get PNG bytes (no disk I/O)  [oai_citation:12‡python.langchain.com](https://python.langchain.com/docs/integrations/document_loaders/docling/?utm_source=chatgpt.com)
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # Convert to PIL  [oai_citation:13‡python.langchain.com](https://python.langchain.com/docs/integrations/document_loaders/docling/?utm_source=chatgpt.com)
            self.page_images.append(pil_image)

        pdf_document.close()  # Release PDF resources  [oai_citation:14‡youtube.com](https://www.youtube.com/watch?v=9gDJ6PhvVck&utm_source=chatgpt.com)

    def _run_smoldocling_on_image(self, image: Image.Image) -> str:
        """
        Run SmolDocling-256M on a single PIL.Image to obtain DocTags output.
        We use a chat prompt: "Convert this page into DocTags format."  [oai_citation:15‡github.com](https://github.com/docling-project/docling?utm_source=chatgpt.com)
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this page into DocTags format."}
                ]
            }
        ]
        # Wrap image + instruction into SmolDocling’s chat template.
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[-1]

        # Generate up to 8192 tokens (to capture long DocTags).  [oai_citation:16‡github.com](https://github.com/docling-project/docling?utm_source=chatgpt.com)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=8192,
            )
        generated_ids = generated_ids[:, prompt_len:]
        # Decode and return a DocTags string (skip special tokens).
        doc_tags = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return doc_tags.strip()

    def _parse_doctags_to_json(self, doc_tags: str) -> Dict[str, Any]:
        """
        Parse a raw DocTags string into a JSON-serializable dictionary:
          1. Create a DocTagsDocument from the raw DocTags + a dummy image.  [oai_citation:34‡medium.com](https://medium.com/%40speaktoharisudhan/smoldocling-a-compact-vision-language-model-c54795474faf?utm_source=chatgpt.com)
          2. Call DoclingDocument.load_from_doctags(...) to build the model.  [oai_citation:35‡medium.com](https://medium.com/%40speaktoharisudhan/smoldocling-a-compact-vision-language-model-c54795474faf?utm_source=chatgpt.com)
          3. Call export_to_dict() to obtain a Python dict.
        """
        # Step 1: Build a DocTagsDocument (image pair list length = number of pages; we pass [None] if image is not needed).  [oai_citation:36‡medium.com](https://medium.com/%40speaktoharisudhan/smoldocling-a-compact-vision-language-model-c54795474faf?utm_source=chatgpt.com)
        tags_doc = DocTagsDocument.from_doctags_and_image_pairs([doc_tags], [None])
        # Step 2: Load into DoclingDocument.  [oai_citation:37‡medium.com](https://medium.com/%40speaktoharisudhan/smoldocling-a-compact-vision-language-model-c54795474faf?utm_source=chatgpt.com)
        doc = DoclingDocument.load_from_doctags(tags_doc, document_name="Document")
        # Step 3: Export the structured document to a dict.
        return doc.export_to_dict()

    def extract(self) -> Dict[str, Any]:
        """
        Full pipeline:
          1. PDF → PIL.Images via fitz.  [oai_citation:17‡youtube.com](https://www.youtube.com/watch?v=9gDJ6PhvVck&utm_source=chatgpt.com)
          2. SmolDocling on each PIL.Image → DocTags.  [oai_citation:18‡github.com](https://github.com/docling-project/docling?utm_source=chatgpt.com)
          3. Parse each DocTags string → JSON via Docling.
          4. Aggregate into {'num_pages': N, 'pages': [{'page_number': i, 'page_json': {...}}, ...]}.

        Returns:
          results_dict: { 'num_pages': int, 'pages': [ { 'page_number': int, 'page_json': dict }, ... ] }
        """
        # Step 1: Convert PDF → images.  [oai_citation:19‡youtube.com](https://www.youtube.com/watch?v=9gDJ6PhvVck&utm_source=chatgpt.com)
        self._convert_pdf_to_images()
        num_pages = len(self.page_images)

        # Step 2 & 3: For each page, run SmolDocling → DocTags → parse to JSON.  [oai_citation:20‡github.com](https://github.com/docling-project/docling?utm_source=chatgpt.com)
        pages_output: List[Dict[str, Any]] = []
        for idx, pil_img in enumerate(self.page_images, start=1):
            print(f"[SmolDocling] Processing page {idx}/{num_pages} ...")
            doc_tags = self._run_smoldocling_on_image(pil_img)
            page_json = self._parse_doctags_to_json(doc_tags)
            pages_output.append({"page_number": idx, "page_json": page_json})

        # Step 4: Build final results dict.
        self.results = {"num_pages": num_pages, "pages": pages_output}
        return self.results

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize self.results to a pretty-printed JSON string.
        Requires extract() to have been called.
        """
        if not self.results:
            raise RuntimeError("No results available. Call .extract() first.")
        return json.dumps(self.results, ensure_ascii=False, indent=indent)

    def save_to_file(self, output_path: str, indent: int = 2) -> None:
        """
        Write the JSON string (from to_json) to disk at output_path.
        """
        json_str = self.to_json(indent=indent)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)


# ──────────────────────────────────────────────────────────────────────────────
# Hardcoded execution block (no CLI arguments needed)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ─── Hardcoded file paths ───────────────────────────────────────
    INPUT_PDF_PATH = "/Users/saketm10/Projects/extraction_by_config/dataset/home_pds_0322.pdf"
    OUTPUT_JSON_PATH = "../output/others/output_file.json"
    # ────────────────────────────────────────────────────────────────────

    extractor = PDFSmolDoclingExtractor(
        pdf_path=INPUT_PDF_PATH,
        device=None  # Auto-detect "cuda" or "cpu"
    )

    # Run the extraction pipeline and get results.
    results_dict = extractor.extract()
    print(f"[Done] Extracted {results_dict['num_pages']} pages → JSON")

    # Save results to disk.
    extractor.save_to_file(OUTPUT_JSON_PATH)
    print(f"[Saved] JSON output written to: {OUTPUT_JSON_PATH}")
