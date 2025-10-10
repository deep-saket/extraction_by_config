from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
from datasets import load_dataset
from openai import OpenAI


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _encode_image(path: Path) -> str:
    with path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


@dataclass
class DatasetSourceConfig:
    dataset_name: str
    split: str
    pdf_column: str
    id_column: str
    max_documents: int
    cache_dir: Optional[str]


class DatasetSource:
    """
    Fetch PDF documents from an open-source dataset hosted on Hugging Face.
    """

    def __init__(self, cfg: DatasetSourceConfig):
        self.cfg = cfg

    def stream(self) -> Iterable[Tuple[str, bytes]]:
        dataset = load_dataset(
            self.cfg.dataset_name,
            split=self.cfg.split,
            cache_dir=self.cfg.cache_dir,
        )
        for idx, sample in enumerate(dataset):
            if idx >= self.cfg.max_documents:
                break
            doc_id = str(sample[self.cfg.id_column])
            pdf_bytes = sample[self.cfg.pdf_column]
            if isinstance(pdf_bytes, dict) and "bytes" in pdf_bytes:
                pdf_bytes = pdf_bytes["bytes"]
            yield doc_id, pdf_bytes


@dataclass
class PDFRenderConfig:
    image_dir: str
    dpi: int
    max_pages: Optional[int]


class PDFRenderer:
    """
    Convert each PDF page to PNG images for downstream prompting.
    """

    def __init__(self, cfg: PDFRenderConfig):
        self.cfg = cfg
        self.root = Path(cfg.image_dir).expanduser().resolve()
        _ensure_dir(self.root)

    def render(self, doc_id: str, pdf_bytes: bytes) -> List[Path]:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        image_paths: List[Path] = []
        for page_index, page in enumerate(pdf, start=1):
            if self.cfg.max_pages and page_index > self.cfg.max_pages:
                break
            pix = page.get_pixmap(dpi=self.cfg.dpi)
            target = self.root / f"{doc_id}_page_{page_index:03d}.png"
            pix.save(str(target))
            image_paths.append(target)
        return image_paths


@dataclass
class LLMConfig:
    api_key: Optional[str]
    api_base: Optional[str]
    model: str
    temperature: float
    max_tokens: int


class LLMExtractor:
    """
    Uses OpenAI GPT endpoints (or compatible) to generate ExtractionItems and
    ExtractionOutputs from page images and descriptors.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY or pass via config.")
        self.client = OpenAI(api_key=api_key, base_url=cfg.api_base)

    def build_prompt(self, doc_id: str, image_paths: Sequence[Path]) -> List[Dict]:
        messages: List[Dict] = [
            {
                "role": "system",
                "content": (
                    "You are an expert document extraction assistant. Given a document page image, "
                    "produce a structured extraction config (ExtractionItems) and corresponding ExtractionOutputs. "
                    "Return JSON with two keys: `extraction_items` (array) and `extraction_outputs` (array). "
                    "Use the ExtractionItem schema from the training project (field_name, description, type, etc.). "
                    "For each field include the predicted type, probable_pages, multipage hints, and search keys. "
                    "For outputs, return the value structure that matches the expected ExtractionOutput schema."
                ),
            }
        ]

        for index, path in enumerate(image_paths, start=1):
            encoded = _encode_image(path)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Document ID: {doc_id} - Page {index}"},
                        {"type": "image_base64", "image_base64": encoded},
                    ],
                }
            )
        return messages

    def infer(self, doc_id: str, image_paths: Sequence[Path]) -> Dict:
        messages = self.build_prompt(doc_id, image_paths)
        response = self.client.responses.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_tokens,
            input=messages,
            response_format={"type": "json_object"},
        )
        text = response.output[0].content[0].text  # type: ignore[attr-defined]
        return json.loads(text)


@dataclass
class SampleWriterConfig:
    dataset_dir: str


class SampleWriter:
    """
    Persist generated configs, outputs, and training samples.
    """

    def __init__(self, cfg: SampleWriterConfig):
        self.cfg = cfg
        self.root = Path(cfg.dataset_dir).expanduser().resolve()
        self.samples_dir = self.root / "samples"
        self.raw_dir = self.root / "raw"
        _ensure_dir(self.samples_dir)
        _ensure_dir(self.raw_dir)

    def write(
        self,
        doc_id: str,
        image_paths: Sequence[Path],
        extraction_items: List[Dict],
        extraction_outputs: List[Dict],
    ) -> None:
        # Save raw artefacts
        raw_payload = {
            "document_id": doc_id,
            "images": [str(path) for path in image_paths],
            "extraction_items": extraction_items,
            "extraction_outputs": extraction_outputs,
        }
        raw_path = self.raw_dir / f"{doc_id}.json"
        with raw_path.open("w", encoding="utf-8") as handle:
            json.dump(raw_payload, handle, indent=2, ensure_ascii=False)

        # Produce page-level samples using existing builder helper
        from train.tools.build_dataset import generate_samples  # lazy import to avoid circular

        # Build mapping {page_index: path}
        page_mapping = {index + 1: path for index, path in enumerate(image_paths)}
        samples = generate_samples(
            extraction_items,
            extraction_outputs,
            page_mapping,
            doc_id,
            self.root,
        )

        for idx, sample in enumerate(samples, start=1):
            sample_path = self.samples_dir / f"{doc_id}_{idx:04d}.json"
            with sample_path.open("w", encoding="utf-8") as handle:
                json.dump(sample, handle, indent=2, ensure_ascii=False)


@dataclass
class PipelineConfig:
    dataset: DatasetSourceConfig
    renderer: PDFRenderConfig
    llm: LLMConfig
    writer: SampleWriterConfig


class DocumentExtractionPipeline:
    """
    End-to-end automation for sourcing open PDFs, generating extraction configs
    via ChatGPT (or compatible), and saving training samples.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.source = DatasetSource(cfg.dataset)
        self.renderer = PDFRenderer(cfg.renderer)
        self.llm = LLMExtractor(cfg.llm)
        self.writer = SampleWriter(cfg.writer)

    def run(self) -> None:
        for doc_id, pdf_bytes in self.source.stream():
            print(f"[Pipeline] Processing document {doc_id}")
            image_paths = self.renderer.render(doc_id, pdf_bytes)
            llm_result = self.llm.infer(doc_id, image_paths)
            items = llm_result.get("extraction_items") or []
            outputs = llm_result.get("extraction_outputs") or []
            if not items or not outputs:
                print(f"[Pipeline] Skipping {doc_id}: LLM returned empty payload.")
                continue
            self.writer.write(doc_id, image_paths, items, outputs)


def main() -> None:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Automate dataset creation from open-source PDFs.")
    parser.add_argument("--dataset-name", type=str, default="rjfamily/DocDownsampled")
    parser.add_argument("--split", type=str, default="train[:10]")
    parser.add_argument("--pdf-column", type=str, default="pdf")
    parser.add_argument("--id-column", type=str, default="document_id")
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--image-dir", type=str, default="./auto_dataset/images")
    parser.add_argument("--dataset-dir", type=str, default="./auto_dataset")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    pipeline = DocumentExtractionPipeline(
        PipelineConfig(
            dataset=DatasetSourceConfig(
                dataset_name=args.dataset_name,
                split=args.split,
                pdf_column=args.pdf_column,
                id_column=args.id_column,
                max_documents=args.max_docs,
                cache_dir=args.cache_dir,
            ),
            renderer=PDFRenderConfig(
                image_dir=args.image_dir,
                dpi=args.dpi,
                max_pages=args.max_pages,
            ),
            llm=LLMConfig(
                api_key=None,
                api_base=args.api_base,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ),
            writer=SampleWriterConfig(
                dataset_dir=args.dataset_dir,
            ),
        )
    )

    pipeline.run()


if __name__ == "__main__":  # pragma: no cover
    main()

