from extraction_io.ExtractionOutputs import (
    ExtractionOutputs,
    KeyValueOutput,
    BulletPointsOutput,
)

raw_json = [
    {
        "field_name": "name_of_applicant",
        "value": "John Doe",
        "post_processing_value": "DOE, JOHN",
        "page_number": 3,
        "key": "Name of Applicant"
    },
    {
        "field_name": "terms_and_conditions",
        "value": "These are the terms and … full contract …",
        "post_processing_value": "These Are The Terms And … Full Contract …",
        "page_number": 5,
        "key": "T&C",
        "multipage_detail": [
            {
                "value": "These are the terms and …",
                "post_processing_value": "These Are The Terms And …",
                "page_number": 5
            },
            {
                "value": "... full contract for user agreement",
                "post_processing_value": "... Full Contract For User Agreement",
                "page_number": 6
            }
        ]
    },
    {
        "field_name": "benefits_list",
        "points": [
            {
                "value": "10% discount on renewal",
                "post_processing_value": "Discount: 10%",
                "page_number": 2,
                "point_number": 1
            },
            {
                "value": "Free roadside assistance",
                "post_processing_value": "Roadside assistance included",
                "page_number": 2,
                "point_number": 2
            },
            {
                "value": "Accidental damage cover",
                "post_processing_value": "Covers accidental damage",
                "page_number": 2,
                "point_number": 3
            }
        ],
        "key": "Benefits"
    },
    {
        "field_name": "policy_clauses",
        "points": [
            {
                "value": "Clause 1: The insurer shall …",
                "post_processing_value": "Insurer shall pay in case of …",
                "page_number": 10,
                "point_number": 1
            },
            {
                "value": "Clause 2: The policy cannot be …",
                "post_processing_value": "Policy invalid if …",
                "page_number": 10,
                "point_number": 2
            },
            {
                "value": "Clause 3: This agreement shall …",
                "post_processing_value": "Agreement begins on date …",
                "page_number": 11,
                "point_number": 3
            },
            {
                "value": "Clause 4: The beneficiary must …",
                "post_processing_value": "Beneficiary must submit docs …",
                "page_number": 11,
                "point_number": 4
            }
        ],
        "key": "Policy Clauses"
    }
]

# Parse/validate
output = ExtractionOutputs.parse_obj(raw_json)

# Access a flat { field_name: value_or_list } dict
flat_map = output.dict_by_field()
# ⇒ {
#    "name_of_applicant": "John Doe",
#    "terms_and_conditions": "These are the terms and … full contract …",
#    "benefits_list": ["10% discount on renewal", "Free roadside assistance", "Accidental damage cover"],
#    "policy_clauses": ["Clause 1: …", "Clause 2: …", "Clause 3: …", "Clause 4: …"]
# }

# Render JSON back out
json_str = output.json(indent=2)