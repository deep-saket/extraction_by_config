import importlib
from typing import Type
from extraction_io.ExtractionOutputs import ExtractionOutput
from common import ExtractionState, CallableComponent


class ResultBuilderFactory(CallableComponent):
    """
    Dynamically resolves and executes the appropriate ResultBuilder class
    for a given extraction item and raw fragments.
    """

    def __init__(self, builder_module: str = "extraction_io.result_builders", suffix: str = "ResultBuilder"):
        """
        Args:
            builder_module (str): Module path where all ResultBuilder classes live.
            suffix (str): Suffix appended to the type-derived class name (e.g., KeyValue -> KeyValueResultBuilder).
        """
        super().__init__()
        self.builder_module = builder_module
        self.suffix = suffix

    def build_result(self, item, raw_data, interim=False) -> ExtractionOutput:
        """
        Build a validated ExtractionOutput for the given item.

        Args:
            item (ExtractionItem): Current extraction item being processed.
            raw_data (Any): Fragments produced by the parser.

        Returns:
            ExtractionOutput: Validated Pydantic model.
        """
        # 1) Build class suffix (e.g., "key-value" → "KeyValueResultBuilder")
        cls_suffix = "".join(part.capitalize() for part in item.type.split("-"))
        builder_class_name = f"{cls_suffix}{self.suffix}"

        # 2) Import dynamically
        builder_module = importlib.import_module(self.builder_module)
        builder_cls: Type = getattr(builder_module, builder_class_name)

        # 3) Assemble kwargs
        kwargs = {
            "field_name": item.field_name,
            "key": item.description,
            "fragments": raw_data,
            "multipage": item.multipage_value,
        }

        # 4) Run builder and validate with Pydantic
        result = builder_cls.build(**kwargs)
        model_obj = ExtractionOutput.model_validate(result.model_dump())

        # 5) Store in global state
        if interim:
            model_obj._interim = True
        ExtractionState.add_response(model_obj)
        return model_obj

    def __call__(self, item, raw_data, interim=False):
        return self.build_result(item, raw_data, interim=interim)