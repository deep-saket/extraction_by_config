#!/usr/bin/env python3
"""
Utility to convert labelled document extraction results into training samples
for the Qwen extraction fine-tuning pipeline.

Each output sample is a JSON file containing the rendered page image and
ground-truth payloads. Two grouping modes are supported:

* field-level (default): one JSON per extraction field per page, with keys
  `document_id`, `field_name`, `page_index`, `page_image`,
  `extraction_item`, `extraction_output`, and optional `last_page_value`.
* page-level: one JSON per page, with keys `document_id`, `page_index`,
  `page_image`, and `fields` (list of field-level entries, each matching the
  structure above).
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("PyMuPDF is required. Install via `pip install pymupdf`.") from exc


GROUPING_FIELD = "field"
GROUPING_PAGE = "page"


def export_pdf_images(pdf_path: Path, image_dir: Path) -> Dict[int, Path]:
    """
    Render each PDF page to PNG and return mapping {page_number: image_path}.
    """
    pdf = fitz.open(str(pdf_path))
    image_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[int, Path] = {}

    for idx, page in enumerate(pdf, start=1):
        pix = page.get_pixmap()
        file_path = image_dir / f"{pdf_path.stem}_page_{idx:03d}.png"
        pix.save(str(file_path))
        mapping[idx] = file_path

    return mapping


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_pages(item: Dict, output: Dict) -> List[int]:
    """
    Infer relevant pages for a given field output.
    """
    item_type = item.get("type", "key-value")
    if item_type in {"key-value", "summary"}:
        page = output.get("page_number")
        if page:
            return [int(page)]
        page_range = output.get("page_range")
        if isinstance(page_range, (list, tuple)) and page_range:
            return [int(page_range[0])]
        probable = item.get("probable_pages") or []
        return [int(probable[0])] if probable else [1]

    if item_type in {"bullet-points", "checkbox"}:
        pages = sorted({int(entry.get("page_number")) for entry in output.get("value", []) if entry.get("page_number")})
        if pages:
            return pages
        probable = item.get("probable_pages") or []
        return [int(probable[0])] if probable else [1]

    if item_type == "table":
        pages = output.get("page_numbers")
        if isinstance(pages, list) and pages:
            return sorted({int(p) for p in pages})
        probable = item.get("probable_pages") or []
        return [int(probable[0])] if probable else [1]

    return [1]


def slice_output_by_page(item_type: str, output: Dict, target_page: int) -> Dict:
    """
    Produce a page-specific view of the extraction output.
    """
    sliced = deepcopy(output)

    if item_type in {"bullet-points", "checkbox"}:
        values = output.get("value") or []
        sliced["value"] = [deepcopy(entry) for entry in values if entry.get("page_number") == target_page]
    elif item_type == "table":
        multipage_detail = output.get("multipage_detail") or []
        detail_filtered = [deepcopy(entry) for entry in multipage_detail if entry.get("page_number") == target_page]
        if detail_filtered:
            sliced["multipage_detail"] = detail_filtered
        page_rows = []
        for row in output.get("value", []):
            page_num = getattr(row, "page_number", None) or (row.get("page_number") if isinstance(row, dict) else None)
            if page_num is None or page_num == target_page:
                page_rows.append(deepcopy(row))
        if page_rows:
            sliced["value"] = page_rows
        sliced["page_numbers"] = [target_page]
    else:
        # key-value / summary use intact payload
        pass

    return sliced


def merge_outputs(item_type: str, accumulated: Optional[Dict], fragment: Dict) -> Dict:
    """
    Combine previous pages' outputs with the current fragment to build last_page_value.
    """
    if accumulated is None:
        return deepcopy(fragment)

    merged = deepcopy(accumulated)
    if item_type in {"bullet-points", "checkbox"}:
        prev = merged.get("value") or []
        curr = fragment.get("value") or []
        merged["value"] = prev + deepcopy(curr)
    elif item_type == "table":
        prev_detail = merged.get("multipage_detail") or []
        curr_detail = fragment.get("multipage_detail") or []
        merged["multipage_detail"] = prev_detail + deepcopy(curr_detail)

        prev_rows = merged.get("value") or []
        curr_rows = fragment.get("value") or []
        merged["value"] = prev_rows + deepcopy(curr_rows)

        prev_pages = set(merged.get("page_numbers") or [])
        prev_pages.update(fragment.get("page_numbers") or [])
        merged["page_numbers"] = sorted(prev_pages)
    else:
        # key-value / summary: return the original (single page)
        merged = deepcopy(fragment)

    return merged


def generate_samples(
    config_items: Iterable[Dict],
    extraction_outputs: Iterable[Dict],
    page_images: Dict[int, Path],
    document_id: str,
    dataset_root: Path,
    grouping: str = GROUPING_FIELD,
) -> List[Dict]:
    """
    Build dataset samples for all fields in one document.
    """
    grouping_mode = grouping.lower()
    if grouping_mode not in {GROUPING_FIELD, GROUPING_PAGE}:
        raise ValueError(f"Unsupported grouping mode: {grouping}")

    config_by_field = {item["field_name"]: item for item in config_items}
    outputs_by_field = {entry.get("field_name"): entry for entry in extraction_outputs}

    samples: List[Dict] = []
    page_payloads: Dict[int, Dict] = {}

    for field_name, item in config_by_field.items():
        output = outputs_by_field.get(field_name)
        if output is None:
            continue

        item_type = item.get("type", "key-value")
        pages = discover_pages(item, output)
        accumulated: Optional[Dict] = None

        for position, page in enumerate(pages):
            image_path = page_images.get(page)
            if image_path is None:
                raise ValueError(f"No rendered image for page {page} (field {field_name}).")

            fragment = slice_output_by_page(item_type, output, page)
            last_page_value = deepcopy(accumulated) if position > 0 and accumulated is not None else None

            if grouping_mode == GROUPING_FIELD:
                sample = {
                    "document_id": document_id,
                    "field_name": field_name,
                    "page_index": int(page),
                    "page_image": os.path.relpath(image_path, dataset_root),
                    "extraction_item": deepcopy(item),
                    "extraction_output": deepcopy(fragment),
                }

                if last_page_value is not None:
                    sample["last_page_value"] = last_page_value

                samples.append(sample)
            else:
                payload = page_payloads.setdefault(
                    int(page),
                    {
                        "document_id": document_id,
                        "page_index": int(page),
                        "page_image": os.path.relpath(image_path, dataset_root),
                        "fields": [],
                    },
                )

                field_entry = {
                    "field_name": field_name,
                    "extraction_item": deepcopy(item),
                    "extraction_output": deepcopy(fragment),
                }

                if last_page_value is not None:
                    field_entry["last_page_value"] = last_page_value

                payload["fields"].append(field_entry)

            accumulated = merge_outputs(item_type, accumulated, fragment)

    if grouping_mode == GROUPING_PAGE:
        for payload in page_payloads.values():
            payload["fields"].sort(key=lambda entry: entry["field_name"])
        samples = [page_payloads[page] for page in sorted(page_payloads.keys())]

    return samples


def save_samples(samples: List[Dict], output_dir: Path, document_id: str, grouping: str) -> None:
    """
    Write samples to disk using stable filenames.
    """
    grouping_mode = grouping.lower()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples, start=1):
        page = sample["page_index"]
        if grouping_mode == GROUPING_FIELD:
            field = sample["field_name"]
            file_path = samples_dir / f"{document_id}_{field}_p{page:03d}_{idx:04d}.json"
        elif grouping_mode == GROUPING_PAGE:
            file_path = samples_dir / f"{document_id}_page_{page:03d}_{idx:04d}.json"
        else:
            raise ValueError(f"Unsupported grouping mode: {grouping}")
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(sample, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build extraction training samples from labelled documents.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the source PDF.")
    parser.add_argument("--config", type=str, required=True, help="Path to the ExtractionItems config JSON.")
    parser.add_argument("--outputs", type=str, required=True, help="Path to the ExtractionOutputs JSON.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write images and samples.")
    parser.add_argument("--doc-id", type=str, default=None, help="Optional document identifier (defaults to PDF stem).")
    parser.add_argument(
        "--grouping",
        type=str,
        choices=[GROUPING_FIELD, GROUPING_PAGE],
        default=GROUPING_FIELD,
        help="Grouping strategy for generated samples: 'field' (default) or 'page'.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    outputs_path = Path(args.outputs).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not outputs_path.is_file():
        raise FileNotFoundError(f"Outputs not found: {outputs_path}")

    document_id = args.doc_id or pdf_path.stem
    image_dir = out_dir / "images"
    page_images = export_pdf_images(pdf_path, image_dir)

    config_items = load_json(config_path)
    outputs = load_json(outputs_path)

    samples = generate_samples(
        config_items,
        outputs,
        page_images,
        document_id,
        out_dir,
        grouping=args.grouping,
    )
    save_samples(samples, out_dir, document_id, grouping=args.grouping)

    print(f"Wrote {len(samples)} {args.grouping}-grouped samples to {out_dir}")


if __name__ == "__main__":
    main()
