# Contact / Email / Signature Extraction Enhancements

## Goal
Push richer extraction coverage for contact panels, email addresses, and signature blocks while keeping everything within the existing `entity-block` / `key-value` abstraction. The aim is to steer the model via scopes and hints instead of proliferating new extraction types.

---

## 1. Config & Validation
- Continue using `entity-block` (and, where appropriate, `key-value`) as the extraction type.
- Extend `scope` options to cover:
  - `contact`, `person`, `organization`, `support`, `billing`
  - `email`
  - `signature_block`
- Optional hints on `ExtractionItem`:
  - `contact_fields: List[str]` – list of sub-fields expected when the scope is contact-oriented.
  - `entity_type: List[str]` – semantic categories to bias block selection (e.g., `["company", "person"]`).
  - `region_hint: List[str]` – restrict search to allowed regions (`header`, `footer`, `left`, `right`, `center`, `top_left`, `top_right`, `bottom_left`, `bottom_right`).
  - `anchor_phrases: List[str]`, `include_bbox: bool` – as already supported.
- Validation rules:
  - `contact_fields` allowed only for `entity-block` when scope is one of the contact scopes.
  - `entity_type`, `region_hint`, `include_bbox` allowed only for `entity-block`.
  - `region_hint` values must come from the supported set; error otherwise.

---

## 2. Prompting
- Update `config/files/prompts.yml`:
  - Provide dedicated instruction fragments for each new scope (`contact`, `person`, `organization`, `support`, `billing`, `email`, `signature_block`).
  - Add renderers for `entity_type`, `region_hint` (with explanations + ordered priority), and `contact_fields`.
  - Ensure instructions clarify how to handle multiple `region_hint` entries (search in listed order) and what each hint represents.
- Enhance `PromptBuilder`:
  - Handle new list-based instructions (`entity_type`, `contact_fields`) in the generic list-processing loop.
  - Continue sourcing overrides from either the item or `item.extra`.

---

## 3. Generation & Output Schemas
- Reuse the `EntityBlockGeneration` schema, but extend it with optional structured sub-fields:
  - `fields: List[EntityField]` where each field has `key`, `value`, `confidence`.
- Mirror the change in `EntityBlockOutput` / `EntityBlockFragment` so structured sub-fields and bounding boxes are preserved end-to-end.
- Continue leveraging `KeyValueGeneration` for cases where a single normalised value (e.g., email) is sufficient.

---

## 4. Parser & Result Builder Updates
- `ParseEntityBlock`:
  - Collect `fields` from the model response when present and pass them downstream.
  - Preserve multipage continuation metadata (`continue_next_page`).
- `EntityBlockResultBuilder`:
  - Normalise and aggregate sub-field lists across fragments (deduplicate by key, prefer first non-empty value).
  - Attach the consolidated field list to the top-level output.
- No new parser/result builder classes required—rely on the enhanced entity-block pipeline.

---

## 5. Email & Signature Considerations
- **Email scope**
  - Add instruction fragments emphasising valid email patterns and optional domain restrictions.
  - Allow `contact_fields` to include keys like `"primary_email"`, `"billing_email"` when desired.
  - For simple single-value emails, the `key-value` type with `scope="email"` remains acceptable.
- **Signature scope**
  - Instruction fragments should call out signer name/title/date and encourage attention to cues such as “Signed by”.
  - Structured fields (e.g., `name`, `title`, `sign_date`) can ride on the same `fields` mechanism if specified in `contact_fields`.

---

## 6. Testing
- Unit tests to cover:
  - Validation behaviour for `entity_type`, `region_hint`, and `contact_fields`.
  - Prompt rendering snapshots for each new scope and region-hint combination.
  - Parser + result builder integration for scenarios with/without structured fields.
  - Dict flattening to ensure structured fields appear under `dict_by_field()`.
- Synthetic fixtures should include:
  - A contact masthead with structured sub-fields.
  - An email-only snippet requiring normalisation.
  - A signature block spanning multiple pages.

---

## 7. Documentation
- Update README / onboarding docs to describe new scopes and hints.
- Provide YAML config examples demonstrating:
  - Contact block extraction via `entity-block` + `scope: contact`.
  - Email detection with `scope: email`.
  - Signature capture with `scope: signature_block`, including structured field expectations.
- Highlight the allowed `region_hint` values and `entity_type` usage in the configuration guide.
