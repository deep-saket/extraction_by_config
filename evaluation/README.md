# Extraction Evaluation Toolkit

Tools for comparing predicted document-extraction results against annotated
ground truth and exporting summary metrics to Excel.

## Expected file format

Each document is represented by a JSON file named `<document_id>.json` in both
the predictions and ground-truth directories. The loader accepts two shapes:

- **Field bundles (preferred):**
  ```json
  {
    "document_id": "smoke_doc",
    "fields": [
      {
        "field_name": "PolicyNumber",
        "type": "key-value",
        "extraction_output": { "value": "PN-123" }
      }
    ]
  }
  ```
- **Legacy lists:**
  ```json
  [
    {
      "field_name": "PolicyNumber",
      "extraction_item": { "type": "key-value" },
      "extraction_output": { "value": "PN-123" }
    }
  ]
  ```

For each field the evaluator reads:

- `field_name` (required)
- `type` (optional; falls back to `extraction_item.type`)
- `extraction_output.value` (preferred) or `value` / `text` / `values`

Any additional keys are ignored.

## Generating a report

Install the dependencies once (pandas + openpyxl):

```bash
pip install pandas openpyxl
```

Run the evaluator:

```bash
python3 -m evaluation.evaluate \
  --pred-dir evaluation/examples/predictions \
  --gt-dir evaluation/examples/ground_truth \
  --output evaluation/report.xlsx
```

The Excel file contains four sheets:

1. `field_results` – one row per field/document with match status
2. `field_accuracy` – accuracy aggregated per field name and type
3. `type_accuracy` – accuracy aggregated per field type (e.g. key-value, table)
4. `mismatches` – subset of rows where prediction != ground truth, including
   missing or unexpected fields

## Example data

`evaluation/examples/` bundles a tiny smoke pair (ground truth vs predictions)
derived from the training smoke dataset. Use it to validate the workflow end to
end or as a template for your own exports.

## Integrating with annotation pipelines

1. Export ground-truth annotations per PDF into `ground_truth/<doc_id>.json`.
2. Save model predictions with matching filenames under `predictions/`.
3. Run `evaluation.evaluate` and inspect the Excel report for overall accuracy,
   per-field performance (e.g. "PolicyNumber"), per-type accuracy, and detailed
   mismatches.
4. Share the Excel file with annotators or stakeholders for review.
