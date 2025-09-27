from extraction_io.result_builders.TableResultBuilder import TableResultBuilder
from extraction_io.ExtractionOutputs import TableOutput

# Simulate two-page fragments: page 1 simple rows, page 2 TableGeneration-style
fragments = [
    {
        "page_number": 1,
        "columns": ["Date", "Desc", "Amount"],
        "rows": [
            ["2025-09-01", "Opening Balance", "0.00"],
            {"cells": ["2025-09-02", "Coffee", "-3.50"]},
        ],
    },
    {
        "page_number": 2,
        # TableGeneration-style rows
        "rows": [
            {
                "row_number": 3,
                "cols": [
                    {"col_number": 1, "value": "2025-09-03"},
                    {"col_number": 2, "value": "Groceries"},
                    {"col_number": 3, "value": "-54.21"},
                ],
            },
            {
                "row_number": 4,
                "cols": [
                    {"col": 1, "value": "2025-09-04"},
                    {"col": 2, "value": "Salary"},
                    {"col": 3, "value": "+2500.00"},
                ],
            },
        ],
    },
]

out: TableOutput = TableResultBuilder.build(
    field_name="AccountTracnsactions", fragments=fragments, key="AccountTracnsactions", multipage=True
)

print("Columns:", out.columns)
print("Pages:", out.page_numbers)
print("Rows:")
for r in out.value:
    print(r.row, r.page_number, [(c.col, c.col_name, c.value) for c in r.cells])

