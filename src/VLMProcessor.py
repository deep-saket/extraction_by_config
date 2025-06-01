
from common import CallableComponent


class VLMProcessor(CallableComponent):
    """
    Handles extraction tasks using a Vision-Language Model (VLM) inference engine.
    """
    def __init__(self, vlm_infer):
        super().__init__()
        self.vlm_infer = vlm_infer

    def extract(self, image_data, prompt, typ):
        """
        Runs the VLM inference on image_data with the given prompt.

        Arguments:
            image_data | PIL.Image.Image - image where extraction to be performed.
            prompt | str - extraction prompt.
            typ | str - type of extraction. One of ["key-value", "bullet-points"]
        """
        self.logger.info("Running VLM inference on image_data...")
        result = self.vlm_infer.infer(image_data, prompt)
        self.logger.info("Finished VLM inference on image_data...")
        return result

    def __call__(self, image_data, prompt, typ, *args, **kwargs):
        return self.extract(image_data, prompt, typ)