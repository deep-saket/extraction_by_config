from common import CallableComponent
import json
from pydantic import ValidationError
from common import DirtyJsonParser, ExtractionState


class LMProcessor(CallableComponent):
    """
    Handles extraction tasks using a Language Model (LM) inference engine.
    """

    def __init__(self, lm_infer):
        super().__init__()
        self.lm_infer = lm_infer


    def extract(self, prompt, generation_model, **kwargs):
        """
        Runs the LM inference with the given prompt, then parses the JSON and validates.

        Arguments:
            prompt | str - extraction prompt asking for JSON only.
            typ | str - type of extraction. One of ["key-value", "bullet-points"]

        Returns:
            A Pydantic model instance (KVGeneration or BulletPointsGeneration) on success.

        Raises:
            RuntimeError if the LM output cannot be parsed or validated.
        """
        item = kwargs.get('item') if kwargs.get('item') else ExtractionState.get_current_extraction_item()

        self.logger.info("Running LM inference...")
        raw_output = self.lm_infer.infer_lang(prompt)
        self.logger.info(f"Finished LM inference. {raw_output}")

        try:
        # Attempt to parse as JSON string
            parsed = DirtyJsonParser.parse(raw_output)
        except json.JSONDecodeError as e:
            self.logger.exception(f"VLM did not return valid JSON: {raw_output}")
            raise RuntimeError(f"VLM did not return valid JSON: {raw_output!r}") from e

        # Validate against the appropriate generation model
        try:
            parsed = generation_model.model_validate(parsed)
            if item:
                parsed.field_name = item.field_name
            return parsed
        except ValidationError as e:
            self.logger.exception(f"LM JSON failed schema validation: {e}")
            return None

    def __call__(self, prompt, generation_model, *args, **kwargs):
        return self.extract(prompt, generation_model, **kwargs)
