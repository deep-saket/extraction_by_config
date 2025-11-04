# Smoke Dataset (Page Grouping)

Single-page bundle generated from `dataset/dummy_statement.pdf` using
`train.tools.build_dataset --grouping page`. Demonstrates how multiple fields on
one page are stored in a `fields[]` array.

- PDF: `../smoke_dataset/dummy_statement.pdf`
- Config: `../../de_config/dummy_statement.json`
- Outputs stub: `../smoke_dataset/outputs.json`
- Samples: `samples/*.json`
- Images: `samples/images/*.png`

Usage:

```bash
python3 -m train.tools.build_dataset \
  --pdf dataset/dummy_statement.pdf \
  --config de_config/dummy_statement.json \
  --outputs tmp/dummy_statement_outputs.json \
  --out-dir train/smoke_dataset_page \
  --doc-id smoke_doc \
  --grouping page
```
