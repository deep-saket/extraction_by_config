from common import CallableComponent, ExtractionState
import json
from pydantic import ValidationError
from common import DirtyJsonParser


class VLMProcessor(CallableComponent):
    """
    Handles extraction tasks using a Vision-Language Model (VLM) inference engine.
    """
    def __init__(self, vlm_infer, lm_processor=None):
        super().__init__()
        self.vlm_infer = vlm_infer
        self.lm_processor = lm_processor

    def extract(self, image_data, prompt, generation_model, **kwargs):
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
        item = kwargs.get('item') if kwargs.get('item') else ExtractionState.get_current_extraction_item()

        parsed = None
        last_raw_output = None
        last_validation_error = None
        had_parse_error = False
        had_validation_error = False
        for i in range(2):
            self.logger.info("Running VLM inference on image_data...")
            raw_output = self.vlm_infer.infer(image_data, prompt)
            last_raw_output = raw_output
            self.logger.info("Finished VLM inference on image_data.")

            try:
                # Attempt to parse as JSON string
                parsed = DirtyJsonParser.parse(raw_output)

            except json.JSONDecodeError as e:
                # raise RuntimeError(f"VLM did not return valid JSON: {raw_output!r}") from e
                self.logger.warning(f"VLM output is not valid JSON (attempt {i+1}/2): {e}. Retrying...")
                had_parse_error = True
                prompt += "\n\nPlease respond with valid JSON only."
                continue

            # Validate against the appropriate generation model
            try:
                parsed = generation_model.model_validate(parsed)
                if item:
                    parsed.field_name = item.field_name
                return parsed
            except ValidationError as e:
                last_validation_error = e
                self.logger.warning(f"VLM output failed schema validation (attempt {i+1}/2): {e}. Retrying...")
                had_validation_error = True
                prompt += "\n\nPlease respond with valid JSON only respecting the schema."
                parsed = None

        # If we reach here, 2 attempts failed.
        # Only try LM repair if the failure mode was JSON parsing (not just schema validation).
        if (not had_parse_error) and (not had_validation_error):
            # Do NOT invoke LM repair for pure schema validation issues
            return parsed

        # Try LM repair using text-only inference.
        try:
            schema = generation_model.model_json_schema()
        except Exception:
            schema = {}
        try:
            schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
        except Exception:
            schema_json = str(schema)

        repair_instructions = (
            "You are a strict JSON repair assistant. Given a target JSON schema and a previous model response, "
            "produce a new JSON that strictly validates against the schema.\n"
            "Rules:\n"
            "- Return ONLY JSON. No prose.\n"
            "- Include all required fields. Use empty strings/lists where appropriate if missing.\n"
            "- Do not add extra fields not in the schema.\n"
            "- Preserve extracted content where possible, mapping to the closest schema field.\n"
            "- The 'field_name' will be set by the system; you don't need to include it if not in schema.\n"
        )
        prev_output = last_raw_output if last_raw_output is not None else ''
        lm_prompt = (
            f"{repair_instructions}\n"
            f"Target JSON Schema (Pydantic v2):\n{schema_json}\n\n"
            "Previous model response (may be invalid):\n"
            f"{prev_output}\n\n"
            "Return the corrected JSON now:"
        )

        try:
            self.logger.info("Attempting schema repair with LM (text-only)...")
            lm_processor = kwargs.get('lm_processor') or self.lm_processor
            if lm_processor is not None:
                repaired = lm_processor(lm_prompt, generation_model, item=item)
            else:
                lm_raw = self.vlm_infer.infer_lang(lm_prompt)
                # Optional: log truncated raw
                self.logger.debug(f"LM repair raw output (truncated): {str(lm_raw)[:500]}")
                repaired = DirtyJsonParser.parse(lm_raw)
                repaired = generation_model.model_validate(repaired)
                if item:
                    repaired.field_name = item.field_name
            self.logger.info("Schema repair succeeded via LM.")
            return repaired
        except Exception as e:
            self.logger.error(
                "Schema repair via LM failed: %s. Last VLM validation error: %s", e, last_validation_error
            )
            return parsed


    def __call__(self, image_data, prompt, generation_model, *args, **kwargs):
        return self.extract(image_data, prompt, generation_model, **kwargs)
