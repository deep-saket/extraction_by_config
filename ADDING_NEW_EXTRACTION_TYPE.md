# Checklist: Adding a New Extraction Type

This checklist ensures you add a new extraction type (e.g., table, bullet-points, checkbox) to the codebase in a robust, maintainable, and consistent way.

---

## 1. Define the Extraction Type in Config
- [ ] Add the new type (e.g., "table") to the `type` field in ExtractionItem (extraction_io/ExtractionItems.py).
- [ ] Add any type-specific config fields (e.g., table_config for tables).
- [ ] Update model validation to enforce required flags (e.g., multi_line_value=True for tables).

## 2. Prompt Template
- [ ] Add a dedicated prompt for the new type in `config/files/prompts.yml` under the `user:` section.
- [ ] Ensure the prompt covers all relevant instructions, including location, disambiguation, and output format.

## 3. Output Schema
- [ ] Add a new output model in `extraction_io/ExtractionOutputs.py` (e.g., TableOutput, TableRowFragment).
- [ ] Support multi-page and multi-line output as needed (e.g., per-row page tracking).
- [ ] Add model validation (e.g., non-empty, correct format).

## 4. Parser Implementation
- [ ] Create a new parser class (e.g., ParseTable) in `src/parsers/ParseTable.py`, inheriting from ParserBase.
- [ ] Implement the `run` method, supporting parent_outputs and multi-page logic.
- [ ] Ensure the parser uses the correct prompt and schema.

## 5. Result Builder
- [ ] Add a result builder (e.g., TableResultBuilder) in `extraction_io/result_builders/`.
- [ ] Implement a static/class `build` method to format and validate output.

## 6. Generation Utils (Optional but Recommended)
- [ ] Add a generation utility (e.g., TableGeneration) in `extraction_io/generation_utils/` for schema/prompt generation.

## 7. Integration in Main Pipeline
- [ ] Ensure dynamic loading in Parser.py supports the new type (no hardcoded type checks).
- [ ] Update any parent/child logic to support the new type.

## 8. Test Coverage
- [ ] Add tests for the new type, including multi-page, multi-line, and parent/child scenarios.
- [ ] Validate output against the schema.

## 9. Documentation
- [ ] Update README and this checklist as needed.
- [ ] Add usage/config examples for the new type.

---

**Tip:** Use the bullet-points or table type as a reference for best practices.

