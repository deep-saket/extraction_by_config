# Qwen2.5-VL Extraction Training (Standalone)

This package fine-tunes a vision-language model so that, given a single-page document image and an `extraction_item` payload, it produces the corresponding `extraction_output` JSON directly—no instruction prompts required.

## Directory layout

```
train/
  config/example.yml      # Editable template for training/inference parameters
  dataloader.py           # Builds DataLoader objects around single-page samples
  data_loader/
    single_page_dataset.py  # Dataset + collator
  model/
    qwen_extraction_model.py # Wrapper around Qwen2.5-VL checkpoints
  rl/
    grpo_agent.py           # Supervised + GRPO objectives
  train.py                  # Main training loop (supervised or GRPO)
  infer1.py                 # Single-sample inference
  infer_dir.py              # Batch inference over a directory
  eval.py                   # Evaluate a checkpoint on the held-out split
  utils/                    # Config loader, logging, seeding utilities
```

## Preparing data

Each dataset sample is a JSON file with paths relative to the file itself:

```json
{
  "page_image": "page_001.png",
  "extraction_item": { ... ExtractionItem JSON ... },
  "extraction_output": { ... ExtractionOutput JSON ... },
  "last_page_value": { ... optional payload for multipage context ... }
}
```

Place all samples in a directory (e.g. `/data/extraction_samples`). Images can live alongside the JSON or in subfolders; the JSON should reference them via a relative path (`page_image` above).

> **Page-level bundles.** The dataset builder can also emit *one JSON per page* via `--grouping page`. The resulting files contain `fields: [...]` arrays. `ExtractionSampleDataset` automatically flattens these bundles into individual training records, so you can freely mix field-level and page-level samples in the same directory.

### Converting existing labelled documents

If you already have:
- the source PDF,
- its extraction config (list of `ExtractionItem` dictionaries),
- the ground-truth extraction outputs (list of `ExtractionOutput` dictionaries),

you can generate training samples automatically:

```bash
python3 -m train.tools.build_dataset \
  --pdf path/to/document.pdf \
  --config path/to/extraction_items.json \
  --outputs path/to/extraction_outputs.json \
  --out-dir /data/extraction_samples/document_001 \
  --doc-id document_001 \
  --grouping page  # or "field" for legacy behaviour
```

The script renders per-page PNGs (under `images/`) and writes ready-to-train JSON files (under `samples/`).
For multi-page bullet/checkbox/table fields it emits one sample per page and threads the previous pages’ content via `last_page_value`.

### Bootstrapping from open datasets (LLM-powered)

To harvest PDFs from an open-source dataset and have ChatGPT generate draft configs/outputs automatically, use the pipeline in `train/pipelines/open_dataset_pipeline.py`. Example:

```bash
python3 -m train.pipelines.open_dataset_pipeline \
  --dataset-name rjfamily/DocDownsampled \
  --split "train[:5]" \
  --pdf-column pdf \
  --id-column document_id \
  --max-docs 5 \
  --image-dir ./auto_dataset/images \
  --dataset-dir ./auto_dataset \
  --model gpt-4.1-mini
```

Set `OPENAI_API_KEY` (or pass `--api-base`/`--model` for compatible endpoints). The script downloads PDFs via 🤗 Datasets, renders PNGs, calls the LLM with the page images, and writes the resulting configs/outputs plus ready-to-train samples under `dataset-dir`.

## Configuration

Copy `train/config/example.yml` and adjust paths/hyperparameters:

```bash
cp train/config/example.yml my_config.yml
```

Key sections:

- `data.processor_name`: Hugging Face processor matching your Qwen checkpoint.
- `data.dataset.root_dir`: Directory containing the JSON samples (field-level or page-level).
- `model.pretrained_name`: Qwen checkpoint to fine-tune and deployment device/dtype knobs.
- `training.mode`: `"supervised"` for cross-entropy warm-up; switch to `"grpo"` to enable policy optimisation.
- `training.loss`: Controls the supervised loss head (`type`, `reduction` strategy: `token`, `sequence`, or `sum`, and `label_smoothing`).
- `rl`: Reinforcement-specific knobs such as `kl_beta`, `reward_scale`, and `normalize_advantages`.
- `logging`: Directory for TensorBoard-style event logs.

See `train/PLAN.md` for an end-to-end schedule that links data prep, supervised warm-up, and GRPO fine-tuning.

## Running training

```bash
python3 -m train.train --config my_config.yml
```

The loop will write checkpoints into `training.checkpoint_dir`. Start with `mode: supervised` for a few epochs, then resume training with `mode: grpo` and the same checkpoint to continue with policy optimisation.

## Evaluation

```bash
python3 -m train.eval --config my_config.yml --checkpoint path/to/checkpoint.pt
```

This reports supervised-loss on the validation/test split.

## Inference

- Single sample:

  ```bash
  python3 -m train.infer1 --config my_config.yml --checkpoint path/to/checkpoint.pt --sample sample.json
  ```

- Entire directory:

  ```bash
  python3 -m train.infer_dir --config my_config.yml --checkpoint path/to/checkpoint.pt --input samples/ --output predictions/
  ```

Both commands emit raw JSON strings representing the predicted `extraction_output`.

## Notes

- When using page-level grouping, keep the `images/` directory adjacent to the JSON samples (e.g. `samples/images/...`).
- Multi-page scenarios should include persistent `last_page_value` context; the dataset builder handles this automatically when configs/output JSON include page metadata.
- Qwen checkpoints expect high-quality images; avoid resizing beyond their documented limits.
- GRPO training is compute intensive. On Apple M-series hardware keep batch size at 1 and monitor memory.
