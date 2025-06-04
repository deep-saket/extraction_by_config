
# Summary Extraction Guide

This guide explains how to configure and run **summary extraction** within the DE (Document Extraction) pipeline. It covers:

1. Overview of supported summary types  
2. Pydantic configuration model fields for summarization  
3. Prompt template structure  
4. High-level parsing logic (ParseSummary)  
5. Output schema for summaries  
6. Example configurations  
7. Troubleshooting tips  

---

## 1. Supported Summary Types

Our DE system supports four summary “scopes”:

- **whole**  
  Summarize the entire document end-to-end.  
  *Config keys:*  
  ```yaml
  type: "summarization"
  summary_scope: "whole"
  ```  

- **section**  
  Summarize a named section (e.g. “How to make a claim”).  
  *Config keys:*  
  ```yaml
  type: "summarization"
  summary_scope: "section"
  section_name: "<exact heading text>"
  search_keys: ["<heading phrase 1>", "<heading phrase 2>"]
  ```  

- **pages**  
  Summarize a fixed set of pages (e.g. pages 10–12).  
  *Config keys:*  
  ```yaml
  type: "summarization"
  summary_scope: "pages"
  probable_pages: [10, 11, 12]
  ```  

- **extracted_fields**  
  Summarize previously extracted fields (e.g. “CoverageDetails” + “ExcessSummary”).  
  *Config keys:*  
  ```yaml
  type: "summarization"
  summary_scope: "extracted_fields"
  extra_rules:
    fields_to_summarize:
      - "CoverageDetails"
      - "ExcessSummary"
  ```  

---

## 2. Pydantic Configuration Model

Each summary item must validate against the `ExtractionItem` schema. Key fields:

- `field_name` (str): Unique name of the summary output.  
- `description` (str): Brief description or hint for the summary.  
- `type` ("summarization"): Marks this item as a summary.  
- `summary_scope` (Literal): One of `"whole"`, `"section"`, `"pages"`, `"extracted_fields"`.  
- `section_name` (str, optional): Required if `summary_scope=="section"`.  
- `probable_pages` (List[int], optional): Required if `summary_scope=="pages"`.  
- `extra_rules.fields_to_summarize` (List[str], optional): Required if `summary_scope=="extracted_fields"`.  
- `search_keys` (List[str], optional): One or more short phrases to guide embedding‐based page retrieval.  

A simplified Pydantic snippet:

```python
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal, Dict, Any

class ExtractionItem(BaseModel):
    field_name: str
    description: str
    type: Literal["key-value", "bullet-points", "summarization"]
    summary_scope: Optional[Literal["whole", "section", "pages", "extracted_fields"]] = None
    section_name: Optional[str] = None
    probable_pages: Optional[List[int]] = None
    extra_rules: Dict[str, Any] = {}
    search_keys: Optional[List[str]] = []

    @model_validator(mode="after")
    def check_summary(cls, values):
        if values["type"] == "summarization":
            scope = values.get("summary_scope")
            if scope == "section" and not values.get("section_name"):
                raise ValueError("section_name required for scope 'section'")
            if scope == "pages" and not values.get("probable_pages"):
                raise ValueError("probable_pages required for scope 'pages'")
            if scope == "extracted_fields" and not values["extra_rules"].get("fields_to_summarize"):
                raise ValueError("fields_to_summarize required in extra_rules")
        return values
```  

---

## 3. Prompt Template Structure

Load a `config/prompts.yml` containing:

```yaml
system: |
  You are a document-extraction agent. Produce exactly one JSON object matching the schema:
  {schema}

postfix: |
  Output only the JSON object; no extra words or explanation.

user:
  summarization: |
    Now: summarize "{field_name}" (description: {description}).
    {scope_instruction}
    {search_keys_instruction}
    Previous summary (if any): "{prev_summary}"
    {postfix}

instructions:
  scope_whole: |
    • Summarize entire document end-to-end.
    • If no content, return {"summary": ""}.

  scope_section: |
    • Summarize only the section titled "{section_name}". Locate relevant pages via search_keys or fallback.
    • If no matching section, return {"summary": ""}.

  scope_pages: |
    • Summarize only pages: {probable_pages}.
    • Concatenate their text/images in order.
    • If pages missing, return {"summary": ""}.

  scope_fields: |
    • Summarize previously extracted fields: {fields_to_summarize}.
    • Concatenate their values, ignoring missing ones.
    • If none exist, return {"summary": ""}.

  search_keys: |
    • Use these embedding search terms: {search_keys}.
    • If exact phrase not found, attempt semantic match.
    • If no strong match, fallback to field_name + description.
```

Rendering logic:

1. Choose `system` + insert JSON schema for `SummaryGeneration`.  
2. Choose `user.summarization`.  
3. Substitute `{scope_instruction}` with one of:
   - `instructions.scope_whole`
   - `instructions.scope_section` (fill `{section_name}`)
   - `instructions.scope_pages` (fill `{probable_pages}`)
   - `instructions.scope_fields` (fill `{fields_to_summarize}`)
4. If `search_keys` exists, substitute `{search_keys_instruction}` with `instructions.search_keys` (fill `{search_keys}`), else leave blank.

---

## 4. High-Level Parsing Logic (ParseSummary)

1. **Determine target pages or fields**  
   - If `summary_scope="whole"`, set pages = [1..last_page].  
   - If `"section"`, use `search_keys` to find relevant pages via `PageFinder`.  
   - If `"pages"`, use `probable_pages` directly.  
   - If `"extracted_fields"`, skip pages; gather extracted values instead.

2. **For each page (or text chunk)**  
   - Build a VLM prompt including:
     - The chosen JSON schema.
     - Scope‐specific instructions.
     - `prev_summary` (concatenate all previous fragments).
   - Call VLM (e.g. `QwenV25Infer.generate(...)`) to produce exactly one JSON fragment:
     ```json
     {
       "summary": "<text>",
       "continue_next_page": <true|false>
     }
     ```
   - Parse and validate against `SummaryGeneration` Pydantic model.

3. **Concatenate fragments**  
   - Continue to next page while `"continue_next_page": true`.  
   - Stop when `"continue_next_page": false` or out of pages.

4. **Wrap into final `SummaryOutput`**  
   ```python
   SummaryOutput(
     field_name=...,
     summary="<full concatenated summary>",
     page_range=[start_page, end_page],      # if scope in ["whole","section","pages"]
     related_fields=[...]                    # if scope == "extracted_fields"
   )
   ```
   Validate and return.

---

## 5. Output Schema for Summaries

In `extraction_io/output_models.py`:

```python
from pydantic import BaseModel, Field, RootModel, model_validator
from typing import List, Optional, Union

class SummaryOutput(BaseModel):
    field_name: str = Field(..., description="Identifier for this summary")
    summary: str = Field(..., description="Final concatenated summary text")
    page_range: Optional[List[int]] = Field(None, description="[start, end] for pages")
    related_fields: Optional[List[str]] = Field(None, description="Fields summarized")

    @model_validator(mode="after")
    def check_nonempty(cls, values):
        if not values["summary"]:
            raise ValueError("Summary cannot be empty.")
        return values

class ExtractionOutput(RootModel[Union[KeyValueOutput, BulletPointsOutput, SummaryOutput]]):
    root: Union[KeyValueOutput, BulletPointsOutput, SummaryOutput]

class ExtractionOutputs(RootModel[List[ExtractionOutput]]):
    root: List[ExtractionOutput]
```

---

## 6. Example Configurations

### 6.1. Whole-Document Summary

```yaml
extraction_items:
  - field_name: "FullDocSummary"
    description: "One-paragraph overview of entire PDF"
    type: "summarization"
    summary_scope: "whole"
    search_keys:
      - "Product Disclosure Statement"
```

### 6.2. Section Summary

```yaml
extraction_items:
  - field_name: "ClaimsProcedureSummary"
    description: "How to make a claim"
    type: "summarization"
    summary_scope: "section"
    section_name: "How to make a claim"
    search_keys:
      - "How to make a claim"
      - "Claims Procedure"
```

### 6.3. Fixed Pages Summary

```yaml
extraction_items:
  - field_name: "FinancialsSummary"
    description: "Summarize pages 30–32"
    type: "summarization"
    summary_scope: "pages"
    probable_pages: [30, 31, 32]
```

### 6.4. Extracted Fields Summary

```yaml
extraction_items:
  - field_name: "PolicyOverviewSummary"
    description: "Summary of key policy details"
    type: "summarization"
    summary_scope: "extracted_fields"
    extra_rules:
      fields_to_summarize:
        - "CoverageDetails"
        - "ExcessSummary"
```

---

## 7. Troubleshooting Tips

- **Non-JSON output**: Double-check that `{schema}` and `{postfix}` are correctly inserted so the VLM outputs a single JSON object.  
- **Long summaries truncated**: Increase `max_new_tokens` in your VLM call (e.g. `generate(..., max_new_tokens=256)`).  
- **No pages found**: Provide precise `search_keys` that match headings in the PDF.  
- **Summary stops prematurely**: Ensure the VLM sets `"continue_next_page": false` on the last page/fragment.  
- **Empty summary**: Verify pages or fields truly contain text; add debug logs in `ParseSummary.run(...)`.

---

_End of Summary Extraction Guide_
