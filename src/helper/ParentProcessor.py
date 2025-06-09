import importlib
from common import CallableComponent, ExtractionState


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
            return raw_data

        ## get parent items
        if not extraction_item.parent:
            return raw_data

        parent_fields = []
        for parent_name in extraction_item.parent:
            parent_fields.append(ExtractionState.get_response_by_field_name(parent_name))

        # Dynamically import this module
        module = importlib.import_module("src.parent_processors")
        try:
            processor_cls = getattr(module, processor_name)
        except AttributeError:
            self.logger.exception(f"Processor '{processor_name}' not found in src.parent_processor")
            raise ImportError(f"Processor '{processor_name}' not found in src.parent_processor")

        # Instantiate and invoke
        processor_instance = processor_cls()
        raw_data = processor_instance(raw_data, *args, **kwargs)
        return raw_data