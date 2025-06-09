from common import CallableComponent
import json
from pydantic import ValidationError
from common import DirtyJsonParser


class VLMProcessor(CallableComponent):
    """
    Handles extraction tasks using a Vision-Language Model (VLM) inference engine.
    """
    def __init__(self, vlm_infer):
        super().__init__()
        self.vlm_infer = vlm_infer

    def extract(self, image_data, prompt, generation_model):
        """
        Runs the VLM inference on image_data with the given prompt, then parses the JSON and validates.

        Arguments:
            image_data | PIL.Image.Image - image where extraction to be performed.
            prompt | str - extraction prompt asking for JSON only.
            typ | str - type of extraction. One of ["key-value", "bullet-points"]

        Returns:
            A Pydantic model instance (KVGeneration or BulletPointsGeneration) on success.

        Raises:
            RuntimeError if the VLM output cannot be parsed or validated.
        """
        self.logger.info("Running VLM inference on image_data...")
        raw_output = self.vlm_infer.infer(image_data, prompt)
        self.logger.info("Finished VLM inference on image_data.")

        try:
            # Attempt to parse as JSON string
            parsed = DirtyJsonParser.parse(raw_output)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"VLM did not return valid JSON: {raw_output!r}") from e

        # Validate against the appropriate generation model
        try:
            return generation_model.model_validate(parsed)
        except ValidationError as e:
            raise RuntimeError(f"VLM JSON failed schema validation: {e}") from e

    def __call__(self, image_data, prompt, generation_model, *args, **kwargs):
        return self.extract(image_data, prompt, generation_model)
