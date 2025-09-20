# src/parsers/parser.py

import importlib
from typing import Union, List, Type, Any
import torch
from dotenv import load_dotenv

from vector_retrieve import PDFProcessor
from models import ModelManager
from common import ExtractionState, BaseComponent
from extraction_io.ExtractionItems import ExtractionItems, ExtractionItem
from extraction_io.ExtractionOutputs import ExtractionOutput, ExtractionOutputs
from src.helper import PromptBuilder, VLMProcessor, PageFinder, ParentProcessor, LMProcessor
from config.loader import settings

load_dotenv()

class Parser(BaseComponent):
    """
    High‐level orchestrator for document extraction. Responsibilities:
      1) Validate and store the extraction configuration (ExtractionItems).
      2) Reset global state (images, embeddings, entries).
      3) Convert the PDF into images & embeddings via PDFProcessor.
      4) Instantiate helper components (PromptBuilder, VLMProcessor, PageFinder) once in __init__.
      5) For each ExtractionItem, find relevant pages, then:
           a. Dynamically instantiate the correct Parse* subclass.
           b. Call its run(pages) method to gather raw fragments or bullets.
           c. Feed those raw fragments/bullets into the appropriate ResultBuilder.
           d. Validate and store the final Pydantic model in state.
      6) After all items are processed, wrap state entries in ExtractionOutputs and write JSON.
    """

    def __init__(self, *args, **kwargs):

        # ---------------------------------------------------------------------
        # Pull parser settings from the already‐loaded settings dict
        # ---------------------------------------------------------------------
        parser_cfg = settings.get("parser", {}).get("args", {})
        self.device = torch.device(parser_cfg.get("device", "cpu"))
        self.models = parser_cfg.get("models", [])
        self.vlm_candidate = parser_cfg.get("vlm_candidate", "QwenV25Infer")
        self.embedding_candidate = parser_cfg.get("embedding_candidate", "ColPaliInfer")
        self.cb_candidate = parser_cfg.get("cb_candidate")


        super().__init__(parser_cfg)
        
        # 1) Load and initialize models
        ModelManager.initialize_models(self.device, model_classes=self.models)

        # 2) Instantiate PDFProcessor with the shared ColPaliInfer
        self.pdf_processor = PDFProcessor(getattr(ModelManager, self.embedding_candidate),
                                          getattr(ModelManager, self.cb_candidate))

        # 3) Dynamically import and instantiate helper components now that ModelManager is ready
        self.prompt_builder = PromptBuilder()
        self.vlm_processor = VLMProcessor(getattr(ModelManager, self.vlm_candidate))
        self.page_finder = PageFinder(self.pdf_processor)
        self.parent_processor = ParentProcessor()
        self.lm_processor = LMProcessor(getattr(ModelManager, self.vlm_candidate))

    def _validate_extraction_items(self, extraction_items: Union[List[dict], ExtractionItems]) -> ExtractionItems:
        """
        Validate extraction items and convert to ExtractionItems if needed.

        Args:
            extraction_items: Raw extraction items data or ExtractionItems instance

        Returns:
            ExtractionItems: Validated extraction items
        """
        if not isinstance(extraction_items, ExtractionItems):
            extraction_items = ExtractionItems.model_validate(extraction_items)
            
        return extraction_items

    def perform_de(
            self,
            pdf_path: str,
            extraction_items: Union[List[dict], ExtractionItems],
            output_json_path: str
    ) -> ExtractionOutputs:
        """
        Main entrypoint for document extraction.

        Steps:
          1) Validate `extraction_items` as a Pydantic ExtractionItems.
          2) Reset ExtractionState (clears images, embeddings, entries).
          3) Store extraction_items in global state.
          4) Populate images & embeddings by calling PDFProcessor(pdf_path).
          5) Process each item via _process_all_items().
          6) Wrap all entries in ExtractionOutputs and write JSON to output_json_path.
          7) Return the ExtractionOutputs Pydantic object.
        """
        # 1) Validate `extraction_items` 
        extraction_items = self._validate_extraction_items(extraction_items)

        self.logger.info("Resetting parser state...")
        self._reset_state()

        self.logger.info("Storing extraction_items in global state...")
        self._set_extraction_items_in_state(extraction_items)

        self.logger.info("Converting PDF → images & embeddings...")
        self._populate_images_and_embeddings(pdf_path)

        self.logger.info("Processing all extraction items...")
        self._process_all_items()

        self.logger.info("Writing final JSON output...")
        return self._write_output(output_json_path)
    
    

    def _reset_state(self):
        """
        Clear out any previously stored images, embeddings, and Pydantic entries.
        Called at the beginning of each perform_de() run so that every document starts fresh.
        """
        ExtractionState.reset()

    def _set_extraction_items_in_state(self, extraction_items: ExtractionItems):
        """
        Store the user’s list of ExtractionItem objects in global state so that all
        downstream methods can refer to ExtractionState.get_extraction_items().
        """
        ExtractionState.set_extraction_items(extraction_items)

    def _populate_images_and_embeddings(self, pdf_path: str):
        """
        Invoke PDFProcessor.__call__(pdf_path), which:
          a. Converts each page of the PDF into an image and stores (page_num, path) in state.
          b. Generates image embeddings via ColPaliInfer and stores (page_num, tensor) in state.
        """
        _ = self.pdf_processor(pdf_path)

    def _process_all_items(self):
        """
        For each ExtractionItem in the user’s config:
          1) Determine extype (e.g., "key-value", "table").
          2) Determine the list of pages: use `probable_pages` or call PageFinder on embeddings.
          3) Dynamically instantiate the correct Parse* class via _get_parser_for_type().
          4) Parsers can access parent info from ExtractionState if needed.
          5) Feed the result into the corresponding ResultBuilder.
          6) Validate final Pydantic model via ExtractionOutput.model_validate() and store it in state.
        """
        for idx, item in enumerate(ExtractionState.get_extraction_items()):
            self.logger.info("=" * 80)
            self.logger.info(f"[Parser] Processing item #{idx}: field_name = '{item.field_name}'")

            ExtractionState.update_curr_extraction_item(idx)

            pages = item.probable_pages or []
            if not pages:
                # If no explicit probable_pages, use PageFinder to get top‐k pages
                pages = self.page_finder()

            extype = item.type
            parser_response_model = self._get_parser_generation_model(item)
            parser_instance = self._get_parser_for_type(item, parser_response_model)

            # Call parser (parsers access parent info via ExtractionState if necessary)
            raw_data = parser_instance(pages)

            # Apply any generic parent_processor transformations
            raw_data = self.parent_processor(raw_data)

            cls_suffix = "".join(part.capitalize() for part in extype.split("-"))
            builder_class_name = f"{cls_suffix}ResultBuilder"
            builder_module = importlib.import_module("extraction_io.result_builders")
            builder_cls: Type = getattr(builder_module, builder_class_name)
            kwargs = {
                "field_name": item.field_name,
                "key": item.description,
                "fragments": raw_data,
                "multipage": item.multipage_value
            }
            result = builder_cls.build(**kwargs)
            model_obj = ExtractionOutput.model_validate(result.model_dump())
            ExtractionState.add_response(model_obj)
            root = model_obj.root
            self.logger.info(f"[Parser] Extracted value for '{item.field_name}': {root.value!r}")

    def _get_parser_for_type(self, item: ExtractionItem, parser_response_model: Any):
        """
        Dynamically load and instantiate:
          - src.parsers.parse_key_value.ParseKeyValue
          - src.parsers.parse_bullet_points.ParseBulletPoints
          - Or fallback to ParseBase (will raise if not implemented)

        Naming convention:
          extype = "key-value" → cls_suffix="KeyValue" → class_name="ParseKeyValue"
          module_name = "src.parsers.parse_key_value"
        """
        extype = item.type
        cls_suffix = "".join(part.capitalize() for part in extype.split("-"))
        class_name = f"Parse{cls_suffix}"
        module_name = f"src.parsers"

        try:
            module = importlib.import_module(module_name)
            parser_cls: Type = getattr(module, class_name)
        except (ModuleNotFoundError, AttributeError):
            # Fallback to ParseBase (which will error if no _process_page is defined)
            self.logger.exception(f"Not able to instantiate parser for type '{extype}'. Using ParseBase instead.")
            from src.parsers.ParseBase import ParseBase
            parser_cls = ParseBase

        return parser_cls(item, self.vlm_processor, self.prompt_builder, parser_response_model)

    def _get_parser_generation_model(self, item: ExtractionItem):
        # Choose JSON schema from the appropriate Pydantic model
        extype = item.type
        cls_suffix = "".join(part.capitalize() for part in extype.split("-"))
        class_name = f"{cls_suffix}Generation"
        module_name = f"extraction_io.generation_utils"

        module = importlib.import_module(module_name)
        model_cls: Type = getattr(module, class_name)

        return model_cls

    def _write_output(self, output_json_path: str) -> ExtractionOutputs:
        """
        After all items have been processed and validated, gather them from ExtractionState:
          a. Each entry is a Pydantic ExtractionOutput.
          b. Call .model_dump() on each to get a raw dict.
          c. Wrap the list of dicts in ExtractionOutputs.model_validate().
          d. Write the pretty-printed JSON to output_json_path.
          e. Return the ExtractionOutputs instance.
        """
        all_models = ExtractionState.get_responses()
        raw_list = [m.model_dump() for m in all_models]
        final_output = ExtractionOutputs.model_validate(raw_list)

        with open(output_json_path, "w") as f:
            f.write(final_output.model_dump_json(indent=2))

        return final_output