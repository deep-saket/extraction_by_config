# Extraction by Configuration

A step-by-step guide on using the **Extraction by Configuration** toolkit to extract structured data from PDF documents using a simple JSON configuration.

---

## 1. Overview

**Extraction by Configuration** lets you specify *what* to extract from any PDF by providing a lightweight JSON array of "extraction items." Each item tells the system:

1. **Which field** (`field_name`) to extract.
2. A **description** or hint about that field.
3. **Where** to look (explicit `probable_pages` or automatic retrieval).
4. **How** to extract it—either as a single key-value or a list of bullet points.
5. Whether the field spans **multiple pages** (`multipage_value`).

The toolkit will:
- Convert each PDF page into an image.
- Generate vector embeddings for each page.
- Identify relevant pages if none are specified.
- Prompt a Vision-Language Model (VLM) to return exactly the field’s JSON.
- Validate and combine multi-page fragments if needed.
- Output a final JSON file containing all extracted fields.

---

## 2. Installation

1. **Clone the repo** and enter the project root:
   ```bash
   git clone https://github.com/your-repo/extraction-by-configuration.git
   cd extraction-by-configuration
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # (Windows: .venv\Scripts\activate)
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation** by running a minimal test (optional):
   ```bash
   python scratch/test_extraction_item.py
   ```
   You should see a message confirming that a sample `ExtractionItem` parses correctly.

---

## 3. Configuration (`de_config/extraction_fields.json`)

Create a JSON file (e.g., `de_config/extraction_fields.json`) containing an array of extraction rules. Each rule has this structure:

```jsonc
[
  {
    "field_name": "BorrowerName",
    "description": "Name of the borrower",
    "probable_pages": [1],
    "type": "key-value",
    "multipage_value": false,
    "multiline_value": false
  },
  {
    "field_name": "benefits_list",
    "description": "List of benefits',
    "probable_pages": [2],
    "type": "bullet-points",
    "multipage_value": false,
    "multiline_value": false
  }
]
```

- **`field_name`** (string): Unique identifier for the output field.
- **`description`** (string): Natural-language hint to help the VLM find the right content.
- **`probable_pages`** (array of ints, optional): Exact pages to scan (1-indexed). If omitted or empty, the system will automatically retrieve the top-3 relevant pages using embeddings.
- **`type`** (literal string): Either `"key-value"` or `"bullet-points"`.  
   - `"key-value"` → a single string value per field.  
   - `"bullet-points"` → a list of bullet entries.
- **`multipage_value`** (bool): If `true`, the field may span multiple pages. The final output will include a `multipage_detail` array for debugging.
- **`multiline_value`** (bool): Reserved for future use (currently ignored).

Add as many items as needed. Save this file under `de_config/`.

---

## 4. Running Extraction

Use the `local_test.py` or instantiate `Parser` directly. Example via CLI:

```bash
python src/local_test.py   --config de_config/extraction_fields.json   --input input/my_document.pdf   --output output/my_document_extracted.json
```

If you prefer Python code:

```python
import json
from src.Parser import Parser

# 1. Load extraction configuration
with open("de_config/extraction_fields.json") as f:
    extraction_config = json.load(f)

# 2. Instantiate Parser
parser = Parser(extraction_config, config_path="config/settings.yml")

# 3. Perform document extraction
output_model = parser.perform_de(
    pdf_path="input/my_document.pdf",
    output_json_path="output/my_document_extracted.json"
)

# 4. (Optional) Access a simple {field_name: value} map
flat = output_model.dict_by_field()
print(flat)
```

- **`config_path`** defaults to `config/settings.yml`, which should specify model‐loading options (e.g., local vs. API).
- **`device`** defaults to `mps` on macOS; you can pass `torch.device("cpu")` or `torch.device("cuda")` if desired.

---

## 5. Output Format

The final JSON (e.g., `output/my_document_extracted.json`) is an array of objects. Each object is either:

### 5.1 Key-Value Entry

```json
{
  "field_name": "BorrowerName",
  "value": "John Sample Doe",
  "post_processing_value": null,
  "page_number": 1,
  "key": "Name of the borrower",
  "multipage_detail": null
}
```

- **`field_name`**: Matches your config.  
- **`value`**: The extracted text. If `multipage_value=true`, this is the concatenation of all fragments.  
- **`post_processing_value`**: Always `null` unless you run extra cleanup.  
- **`page_number`**: If single-page extraction, the page number. If multi-page, the starting page.  
- **`key`**: The `description` from your config.  
- **`multipage_detail`**:  
  - If `multipage_value=false`, this is `null`.  
  - If `true`, an array of objects, each:

    ```json
    {
      "value": "partial text from page 2",
      "post_processing_value": null,
      "page_number": 2
    }
    ```

    …allowing you to debug exactly which page contributed which fragment.

### 5.2 Bullet-Points Entry

```json
{
  "field_name": "benefits_list",
  "points": [
    { "value": "10% discount on renewals",  "post_processing_value": null, "page_number": 2, "point_number": 1 },
    { "value": "Free roadside assistance",   "post_processing_value": null, "page_number": 2, "point_number": 2 },
    { "value": "Priority customer support",   "post_processing_value": null, "page_number": 3, "point_number": 3 }
  ],
  "key": "List of benefits"
}
```

- **`points`**: An array of bullet objects. Each bullet has:  
  - **`value`**: The extracted bullet text.  
  - **`post_processing_value`**: Always `null` unless you clean it further.  
  - **`page_number`**: The page on which that bullet was found.  
  - **`point_number`**: A running index across all pages (starts at 1).

- **`key`**: The `description` from your config.

---

## 6. Post-Processing Tips

1. **NER Cleanup (Optional)**  
   For `key-value` fields (names, dates, addresses), you might run a simple NER or regex to confirm/clean the extracted string before using it downstream.

2. **Discard Low-Confidence Fragments**  
   If you add a confidence score to each fragment (via a future VLM prompt), you can filter out fragments with low confidence before concatenation.

3. **Normalization**  
   - Trim excessive whitespace.  
   - Standardize date formats (e.g., `YYYY-MM-DD`).  
   - Remove trailing punctuation.

---

## 7. Troubleshooting

- **VLM outputs full conversation instead of JSON**  
  - Ensure you use `PromptBuilder` verbatim. It asks for “exactly ONE JSON object” with no extra text.  
  - If stray text still appears (e.g. markdown fences), `DirtyJsonParser` will strip fences and parse the first `{…}` block.

- **`ExtractionItems` validation fails**  
  - Make sure `"type"` is exactly `"key-value"` or `"bullet-points"` (including hyphen).  
  - Verify you have all required fields (`field_name`, `description`, `type`).

- **Page retrieval seems off**  
  - Check that your ColPali model is loading correctly in `config/settings.yml`.  
  - Inspect `ExtractionState.get_embeddings()` to ensure embeddings were generated.

- **Performance Comments**  
  - Converting a large PDF (100+ pages) to images and embeddings can take several seconds per page.  
  - Consider downsampling or cropping pages if you only need a small region.

---

## 8. Extending Extraction

1. **Add a new `type`**  
   - Define a Pydantic schema in `output_models.py` (e.g., `TableOutput`).  
   - Add a branch in `PromptBuilder.build(...)` with a new JSON schema.  
   - In `Parser._process_all_items`, call a new helper to parse table‐style JSON from VLM.

2. **Use OCR text instead of images**  
   - Modify `PDFProcessor` to run Tesseract (or layout‐aware OCR) and feed text to the VLM instead of images.

3. **Chain multi-field extraction**  
   - If one field’s value influences another (e.g. “extract employee list, then extract emails for each”), you can post‐process one JSON output and feed it back into the VLM.

---

## 9. License & Acknowledgments

Licensed under your chosen open‐source license, 2025.  
Built with:
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF→image
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for Qwen
- [ColPali](https://github.com/vidore/colpali_engine) for multimodal retrieval
- [Pydantic](https://github.com/pydantic/pydantic) for robust schemas
- [dirtyjson](https://pypi.org/project/dirtyjson) for tolerant JSON parsing

