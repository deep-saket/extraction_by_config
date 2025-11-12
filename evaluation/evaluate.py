#!/usr/bin/env python3
"""Evaluate predicted extraction outputs against ground-truth annotations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class FieldEntry:
    document_id: str
    field_name: str
    field_type: str
    value: object
    display_value: str


def load_documents(directory: Path) -> Dict[str, Dict[str, FieldEntry]]:
    documents: Dict[str, Dict[str, FieldEntry]] = {}

    for path in sorted(directory.glob("*.json")):
        if path.is_dir():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        doc_id = str(payload.get("document_id") or path.stem)
        entries = _extract_entries(doc_id, payload)
        documents[doc_id] = {entry.field_name: entry for entry in entries}

    return documents


def _extract_entries(document_id: str, payload: object) -> List[FieldEntry]:
    if isinstance(payload, dict):
        if "fields" in payload and isinstance(payload["fields"], list):
            entries = payload["fields"]
        elif "extraction_output" in payload:
            entries = [payload]
        else:
            entries = []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []

    results: List[FieldEntry] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        field_name = entry.get("field_name")
        if not field_name:
            extraction_item = entry.get("extraction_item") if isinstance(entry.get("extraction_item"), dict) else {}
            field_name = extraction_item.get("field_name")
        if not field_name:
            continue

        field_type = entry.get("type") or _guess_type(entry)
        value_obj = _extract_value(entry)
        display_value = _normalise_display_value(value_obj)

        results.append(
            FieldEntry(
                document_id=document_id,
                field_name=str(field_name),
                field_type=str(field_type or "unknown"),
                value=value_obj,
                display_value=display_value,
            )
        )

    return results


def _guess_type(entry: Dict) -> Optional[str]:
    extraction_item = entry.get("extraction_item")
    if isinstance(extraction_item, dict):
        return extraction_item.get("type")
    extraction_output = entry.get("extraction_output")
    if isinstance(extraction_output, dict):
        return extraction_output.get("type")
    return None


def _extract_value(entry: Dict) -> object:
    if "extraction_output" in entry and isinstance(entry["extraction_output"], dict):
        payload = entry["extraction_output"]
    else:
        payload = entry

    if isinstance(payload, dict):
        if "value" in payload:
            return payload["value"]
        if "text" in payload:
            return payload["text"]
        if "values" in payload:
            return payload["values"]
    return payload


def _normalise_display_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def compare_documents(
    gt_docs: Dict[str, Dict[str, FieldEntry]],
    pred_docs: Dict[str, Dict[str, FieldEntry]],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for doc_id, gt_fields in gt_docs.items():
        pred_fields = pred_docs.get(doc_id, {})
        for field_name, gt_entry in gt_fields.items():
            pred_entry = pred_fields.get(field_name)
            if pred_entry is None:
                records.append(
                    {
                        "document_id": doc_id,
                        "field_name": field_name,
                        "field_type": gt_entry.field_type,
                        "gt_value": gt_entry.display_value,
                        "pred_value": "",
                        "match": False,
                        "status": "missing_prediction",
                    }
                )
                continue

            match = gt_entry.display_value == pred_entry.display_value
            status = "matched" if match else "value_mismatch"
            records.append(
                {
                    "document_id": doc_id,
                    "field_name": field_name,
                    "field_type": gt_entry.field_type or pred_entry.field_type,
                    "gt_value": gt_entry.display_value,
                    "pred_value": pred_entry.display_value,
                    "match": match,
                    "status": status,
                }
            )

        # unexpected predictions
        for field_name, pred_entry in pred_fields.items():
            if field_name in gt_fields:
                continue
            records.append(
                {
                    "document_id": doc_id,
                    "field_name": field_name,
                    "field_type": pred_entry.field_type,
                    "gt_value": "",
                    "pred_value": pred_entry.display_value,
                    "match": False,
                    "status": "unexpected_prediction",
                }
            )

    # documents only present in predictions
    for doc_id, pred_fields in pred_docs.items():
        if doc_id in gt_docs:
            continue
        for field_name, pred_entry in pred_fields.items():
            records.append(
                {
                    "document_id": doc_id,
                    "field_name": field_name,
                    "field_type": pred_entry.field_type,
                    "gt_value": "",
                    "pred_value": pred_entry.display_value,
                    "match": False,
                    "status": "unexpected_document",
                }
            )

    return pd.DataFrame.from_records(records)


def build_field_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["field_name", "field_type", "total", "correct", "accuracy"])
    grouped = df.groupby(["field_name", "field_type"], dropna=False)
    summary = grouped["match"].agg(["count", "sum"]).reset_index()
    summary = summary.rename(columns={"count": "total", "sum": "correct"})
    summary["accuracy"] = summary.apply(
        lambda row: row["correct"] / row["total"] if row["total"] else 0.0,
        axis=1,
    )
    return summary.sort_values(["field_name", "field_type"]).reset_index(drop=True)


def build_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["field_type", "total", "correct", "accuracy"])
    grouped = df.groupby("field_type", dropna=False)
    summary = grouped["match"].agg(["count", "sum"]).reset_index()
    summary = summary.rename(columns={"count": "total", "sum": "correct"})
    summary["accuracy"] = summary.apply(
        lambda row: row["correct"] / row["total"] if row["total"] else 0.0,
        axis=1,
    )
    return summary.sort_values("field_type").reset_index(drop=True)


def write_report(field_df: pd.DataFrame, output_path: Path) -> None:
    field_summary = build_field_summary(field_df)
    type_summary = build_type_summary(field_df)
    mismatches = field_df[field_df["match"] == False]  # noqa: E712

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        field_df.to_excel(writer, index=False, sheet_name="field_results")
        field_summary.to_excel(writer, index=False, sheet_name="field_accuracy")
        type_summary.to_excel(writer, index=False, sheet_name="type_accuracy")
        mismatches.to_excel(writer, index=False, sheet_name="mismatches")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction predictions vs ground truth.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory containing prediction JSON files.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory containing ground-truth JSON files.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the Excel report.")
    args = parser.parse_args(argv)

    if not args.pred_dir.is_dir():
        raise NotADirectoryError(f"Prediction directory not found: {args.pred_dir}")
    if not args.gt_dir.is_dir():
        raise NotADirectoryError(f"Ground truth directory not found: {args.gt_dir}")

    gt_docs = load_documents(args.gt_dir)
    pred_docs = load_documents(args.pred_dir)

    field_df = compare_documents(gt_docs, pred_docs)
    write_report(field_df, args.output)

    overall_accuracy = field_df["match"].mean() if not field_df.empty else 0.0
    print(f"Wrote evaluation report to {args.output} (overall accuracy={overall_accuracy:.3f})")


if __name__ == "__main__":
    main()
