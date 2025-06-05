from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union

class CheckboxGeneration(BaseModel):
    """
    Unified schema for checkbox extraction (single‐ or multi‐value) with a page‐continuation flag.
    Exactly one of `selected_option` or `selected_options` must be set,
    and `continue_next_page` must always be present.
    """
    field_name: str = Field(
        ...,
        description="The logical field name, e.g. 'OccupancyStatus' or 'FeaturesSelected'."
    )
    selected_option: Optional[str] = Field(
        None,
        description="If exactly one checkbox is expected, that single selected value (\"\" if none)."
    )
    selected_options: Optional[List[str]] = Field(
        None,
        description="If multiple selections are allowed, the list of all selected values (empty list if none)."
    )
    continue_next_page: bool = Field(
        ...,
        description="True if there is another page to process for this field; otherwise false."
    )

    @model_validator(mode="after")
    def check_exclusive(cls, model: "CheckboxGeneration") -> "CheckboxGeneration":
        """
        Enforce that exactly one of `selected_option` or `selected_options` is provided,
        and ensure `continue_next_page` is always a boolean.
        """
        one = model.selected_option is not None
        two = model.selected_options is not None

        if one and two:
            raise ValueError(
                "CheckboxGeneration: cannot set both selected_option and selected_options."
            )
        if not one and not two:
            # Even if “no selection,” the model should explicitly set:
            # - single_value → selected_option = ""
            # - multi_value  → selected_options = []
            raise ValueError(
                "CheckboxGeneration: one of selected_option or selected_options must be provided (even if empty)."
            )

        # continue_next_page must be boolean; Pydantic will enforce its type,
        # so no extra runtime check is needed here beyond trusting Pydantic.

        return model