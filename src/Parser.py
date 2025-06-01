# src/parsers/parser.py

import json
from typing import Union, List
import torch

from vector_retrieve import PDFProcessor
from src.helper import PromptBuilder, VLMProcessor, PageFinder
from models import ModelManager
from common import ExtractionState
from extraction_io.ExtractionItems import ExtractionItems
from extraction_io.ExtractionOutputs import ExtractionOutput, ExtractionOutputs
from extraction_io.result_builders import KeyValueResultBuilder, BulletPointsResultBuilder


class Parser:
    def __init__(
        self,
        extraction_config: Union[List[dict], ExtractionItems],
        config_path: str = "config/settings.yml",
        device: torch.device = torch.device("mps")
    ):
        """
        Initializes the Parser. Validates and stores extraction_config via Pydantic.

        Args:
            extraction_config: Either a raw list-of-dicts or an ExtractionItems object.
            config_path:       Path to your YAML with model settings.
            device:            The torch.device to run inference on ("cuda"/"mps"/"cpu").
        """
        # 1) Validate extraction_config up front
        if isinstance(extraction_config, ExtractionItems):
            self.extraction_config = extraction_config
        else:
            self.extraction_config = ExtractionItems.model_validate(extraction_config)

        # 2) Initialize models
        ModelManager.load_config(config_path)
        ModelManager.initialize_models(device)

        # 3) Build helper components
        self.pdf_processor  = PDFProcessor(ModelManager.colpali_infer)
        self.prompt_builder = PromptBuilder()
        self.vlm_processor  = VLMProcessor(ModelManager.qwen_infer)
        self.page_finder    = PageFinder(self.pdf_processor)

    def perform_de(self, pdf_path: str, output_json_path: str) -> ExtractionOutputs:
        """
        Top‐level document extraction. Steps:
          1) Reset state
          2) Populate images & embeddings
          3) Process each config item into a validated output
          4) Write JSON
        """
        self._reset_state()
        self._populate_images_and_embeddings(pdf_path)
        self._process_all_items()
        return self._write_output(output_json_path)

    # ────────────────────────────────────────────────────────────────────────────
    # 1) Reset everything in ExtractionState so we start fresh
    # ────────────────────────────────────────────────────────────────────────────
    def _reset_state(self):
        ExtractionState.reset()

    # ────────────────────────────────────────────────────────────────────────────
    # 2) Use PDFProcessor to fill ExtractionState.images & .embeddings
    # ────────────────────────────────────────────────────────────────────────────
    def _populate_images_and_embeddings(self, pdf_path: str):
        # PDFProcessor.__call__(pdf_path) will (internally) reset state and set images/embeddings
        _ = self.pdf_processor(pdf_path)

    # ────────────────────────────────────────────────────────────────────────────
    # 3) Loop over each ExtractionItem in the config and produce entries
    # ────────────────────────────────────────────────────────────────────────────
    def _process_all_items(self):
        for item in self.extraction_config:
            # Extract parameters from the typed Pydantic model:
            fn      = item.field_name
            desc    = item.description
            extype  = item.type             # "key-value" or "bullet-points"
            multi   = item.multipage_value
            pages   = item.probable_pages or []

            if not pages:
                pages = self.page_finder(
                    ExtractionState.get_embeddings(),
                    fn,
                    desc
                )

            # Dispatch based on type
            if extype == "key-value":
                model_obj = self._process_key_value(fn, desc, pages, multi)
            else:  # "bullet-points"
                model_obj = self._process_bullet_points(fn, desc, pages)

            # Store the validated Pydantic model into ExtractionState.entries
            ExtractionState.add_entry(model_obj)

    # ────────────────────────────────────────────────────────────────────────────
    # 3.a) Handle a single key-value field: gather fragments + validate
    # ────────────────────────────────────────────────────────────────────────────
    def _process_key_value(
        self,
        field_name: str,
        description: str,
        pages: List[int],
        multipage: bool
    ) -> ExtractionOutput:
        """
        Returns a validated KeyValueOutput model.
        """
        fragments = self._gather_fragments(field_name, description, pages)
        # Delegate to the builder to create a Pydantic KeyValueOutput
        kv_model = KeyValueResultBuilder.build(
            field_name=field_name,
            fragments=fragments,
            key=description,
            multipage=multipage
        )
        # Wrap in ExtractionOutput so that uniform validation is next:
        return ExtractionOutput.model_validate(kv_model.model_dump())

    # ────────────────────────────────────────────────────────────────────────────
    # 3.b) Handle a single bullet-points field: gather bullets + validate
    # ────────────────────────────────────────────────────────────────────────────
    def _process_bullet_points(
        self,
        field_name: str,
        description: str,
        pages: List[int]
    ) -> ExtractionOutput:
        """
        Returns a validated BulletPointsOutput model.
        """
        bullets = self._gather_bullets(field_name, description, pages)
        # Delegate to the builder to create a Pydantic BulletPointsOutput
        bp_model = BulletPointsResultBuilder.build(
            field_name=field_name,
            bullets=bullets,
            key=description
        )
        return ExtractionOutput.model_validate(bp_model.model_dump())

    # ────────────────────────────────────────────────────────────────────────────
    # 3.a.i) Gather page‐level fragments for a key-value field
    # ────────────────────────────────────────────────────────────────────────────
    def _gather_fragments(
        self,
        field_name: str,
        description: str,
        pages: List[int]
    ) -> List[dict]:
        """
        Returns a list of dicts, each containing:
          {
            "value": str,
            "post_processing_value": Optional[str],
            "page_number": int
          }
        """
        fragments = []
        for pg in pages:
            for num, img_path in ExtractionState.get_images():
                if num == pg:
                    from PIL import Image
                    img = Image.open(img_path).convert("RGB")
                    prompt = self.prompt_builder(field_name, description, pg, "key-value")
                    raw_output = self.vlm_processor(img, prompt, typ="key-value")

                    if isinstance(raw_output, dict):
                        val  = raw_output.get("value", "")
                        post = raw_output.get("post_processing_value", None)
                    elif hasattr(raw_output, "value"):
                        val  = getattr(raw_output, "value", "")
                        post = getattr(raw_output, "post_processing_value", None)
                    else:
                        val  = str(raw_output)
                        post = None

                    fragments.append({
                        "value": val,
                        "post_processing_value": post,
                        "page_number": pg
                    })
                    break
        return fragments

    # ────────────────────────────────────────────────────────────────────────────
    # 3.b.i) Gather page‐level bullets for a bullet-points field
    # ────────────────────────────────────────────────────────────────────────────
    def _gather_bullets(
        self,
        field_name: str,
        description: str,
        pages: List[int]
    ) -> List[dict]:
        """
        Returns a flat list of dicts, each containing:
          {
            "value": str,
            "post_processing_value": None,
            "page_number": int,
            "point_number": int
          }
        """
        all_points = []
        point_idx = 1

        for pg in pages:
            for num, img_path in ExtractionState.get_images():
                if num == pg:
                    from PIL import Image
                    img = Image.open(img_path).convert("RGB")
                    prompt = self.prompt_builder(field_name, description, pg, "bullet-points")
                    raw_output = self.vlm_processor(img, prompt, typ="bullet-points")

                    # Normalize raw_output into a list of strings
                    candidates: List[str] = []
                    if isinstance(raw_output, dict) and "points" in raw_output:
                        candidates = raw_output["points"]
                    elif hasattr(raw_output, "points"):
                        candidates = getattr(raw_output, "points", [])
                    elif isinstance(raw_output, list):
                        candidates = [str(x) for x in raw_output]
                    elif isinstance(raw_output, str):
                        candidates = [
                            line.strip()
                            for line in raw_output.splitlines()
                            if line.strip()
                        ]

                    for b in candidates:
                        all_points.append({
                            "value": b,
                            "post_processing_value": None,
                            "page_number": pg,
                            "point_number": point_idx
                        })
                        point_idx += 1
                    break
        return all_points

    # ────────────────────────────────────────────────────────────────────────────
    # 4) Read all Pydantic entries and write final JSON
    # ────────────────────────────────────────────────────────────────────────────
    def _write_output(self, output_json_path: str) -> ExtractionOutputs:
        """
        Wraps all entries in ExtractionOutputs, validates, and writes JSON.
        """
        all_models = ExtractionState.get_entries()
        # Each is already a Pydantic model, so model_dump() → raw dict
        raw_list = [m.model_dump() for m in all_models]
        final_output = ExtractionOutputs.model_validate(raw_list)

        with open(output_json_path, "w") as f:
            f.write(final_output.model_dump_json(indent=2))

        return final_output