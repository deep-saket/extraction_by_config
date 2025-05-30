from common import CallableComponent


class PromptBuilder(CallableComponent):
    """
    Builds standardized prompts for extraction tasks.
    """
    def __init__(self):
        super(PromptBuilder, self).__init__()

    def build(self, field_name, description, page_num, extraction_type):
        return f"Extract the {field_name} ({description}) from page {page_num} as {extraction_type}."

    def __call__(self, field_name, description, page_num, extraction_type):
        return self.build(field_name, description, page_num, extraction_type)