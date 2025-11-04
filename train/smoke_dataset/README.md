# Smoke Dataset (Field Grouping)

Tiny two-record slice generated from `dataset/dummy_statement.pdf` using
`train.tools.build_dataset --grouping field`. The source PDF is bundled here for
easy reference. Useful for CLI smoke tests.

- PDF: `dummy_statement.pdf`
- Config: `../../de_config/dummy_statement.json`
- Outputs stub: `outputs.json`
- Samples: `samples/*.json`
- Images: `samples/images/*.png`

Usage:

```bash
python3 -m train.tools.build_dataset \
  --pdf dataset/dummy_statement.pdf \
  --config de_config/dummy_statement.json \
  --outputs tmp/dummy_statement_outputs.json \
  --out-dir train/smoke_dataset \
  --doc-id smoke_doc \
  --grouping field
```
