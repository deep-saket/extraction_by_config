import importlib
from common import CallableComponent, ExtractionState
from src.visual_extract.parent_processors.ExtractionItemsBase import ExtractionItemsBase


class ParentProcessor(CallableComponent):
    def __call__(self, raw_data, *args, **kwargs):
        """
        Look up the processor name in extraction_item.extra['parent_processor'],
        import that processor class from this module, instantiate it, and
        delegate the call to obtain raw data.
        """
        extraction_item = ExtractionState.get_current_extraction_item()
        processor_name = extraction_item.extra.get("parent_processor", None)
        if not processor_name:
            # Fallback to base parent processor if parents exist
            if extraction_item.parent:
                return ExtractionItemsBase()(raw_data, *args, **kwargs)
            return raw_data

        # If parent_processor explicitly set but no parents, skip
        if not extraction_item.parent:
            return raw_data

        if not kwargs.get("skip_parents", False):
            for parent_name in extraction_item.parent:
                ExtractionState.get_response_by_field_name(parent_name)

        # Dynamically import module under visual_extract.parent_processors
        module = importlib.import_module("src.visual_extract.parent_processors")
        try:
            processor_cls = getattr(module, processor_name)
        except AttributeError:
            self.logger.exception(f"Processor '{processor_name}' not found in src.visual_extract.parent_processors")
            raise ImportError(f"Processor '{processor_name}' not found in src.visual_extract.parent_processors")

        processor_instance = processor_cls()
        return processor_instance(raw_data, *args, **kwargs)
