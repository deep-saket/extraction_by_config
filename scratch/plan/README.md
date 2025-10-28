# Implementation Roadmap

This document aggregates the planned enhancements discussed so far, starting with the introduction of an `entity-block` extraction type and extending through the upcoming contact/email/signature features and scope refinements.

---

## 1. `entity-block` Extraction Type

### 1.1 Goals
- Capture prominent visual blocks (e.g., company names that appear as headers).
- Support hints for location (header, footer) and provide optional bounding boxes.

### 1.2 Config & Validation
- Extend `ExtractionItem.type` to include `"entity-block"`.
- Optional hints:
  - `anchor_phrases: List[str]`
  - `include_bbox: bool`
  - `region_hint: List[str]` (e.g., `["header","left"]`)
  - `entity_type: List[str]`
  - Ensure validator enforces:
  - `include_bbox`, `region_hint`, `entity_type`, `contact_fields` only for entity-block.
  - `anchor_phrases` accepted for `entity-block` (and optionally `key-value`).
  - `multipage_value` auto-enabled for visual blocks if appropriate.

### 1.3 Prompting
- Add `user.entity_block` entry in `config/files/prompts.yml`.
- Extend `instructions_detail` with the new boolean/list flags.
- Instruction fragments:
  - Anchor phrase guidance.
  - Region hint explanation (ordered priority, allowed values).
  - Entity type focus (company vs person etc.).
  - Bounding box request when `include_bbox` is true.
  - Contact-oriented instructions keyed off the new scopes.

### 1.4 Generation Schema
- Create `EntityBlockGeneration` Pydantic model, including:
  - `field_name`, `value`, `page_number`, `continue_next_page`.
  - Optional `bbox` (`x1,y1,x2,y2`) and `confidence`.

### 1.5 Output Models
- Introduce `EntityBlockOutput` with fragments preserving page provenance, bounding boxes, and optional structured sub-fields.
- Update `ExtractionOutput` union and `dict_by_field()` to flatten entity-block outputs (e.g., `{value, page_number, bbox, fields}`).

### 1.6 Parser & Result Builder
- Implement `ParseEntityBlock` (similar to `ParseKeyValue` but expecting the new schema).
- Add `EntityBlockResultBuilder` to consolidate fragments and instantiate `EntityBlockOutput`.
- Ensure factories (`ResultBuilderFactory`, parser imports) resolve the new type.

### 1.7 PromptBuilder Enhancements
- Allow boolean/list instructions to pull override values from `item.extra`.
- Provide join/format helpers for `anchor_phrases`, `region_hint`, `entity_type`, and `contact_fields`.

### 1.8 Sample Config
```yaml
- field_name: company_name
  description: Main supplier name shown in the invoice masthead
  type: entity-block
  probable_pages: [1]
  anchor_phrases:
    - invoice
    - supplier
  entity_type: ["company"]
  contact_fields: []
  extra:
    region_hint: ["header"]
    include_bbox: true
```

### 1.9 Testing & Docs
- Unit tests for validation, prompt assembly, parser/builder integration (including contact scopes and structured fields).
- Sample output JSON showing `EntityBlockOutput` with bounding boxes and optional sub-fields.
- README/usage docs updated accordingly.

---

## 2. Scope Extensions for Key-Value / Entity Blocks

### 2.1 New Scope Values
- `address_block`, `currency`, `identifier`, `number`, `date`.
- `email`, `signature_block`, `contact`, `person`, `organization`, `support`, `billing`.

### 2.2 Prompting
- Add scope-specific instruction fragments describing expected formats or heuristics (e.g., currency symbols, ISO date normalization).
- For `email`: emphasise email pattern validation, optional domain restrictions.
- For `signature_block`: instruct the model to capture signer name/title/date and look near signature lines or footer regions.
- Ensure `PromptBuilder` injects the fragments when `item.scope` equals one of the new options (including contact-related scopes).

### 2.3 Validation
- Restrict these scope values to compatible types (`key-value`, `entity-block`).

### 2.4 Documentation
- Update config examples explaining when to pick each scope.

---

## 3. Contact-Oriented Scopes

- Treat contact extraction as an `entity-block` scope (`contact`, `person`, `organization`, `support`, `billing`).
- Allow optional `contact_fields` to request structured sub-fields (name, phone, email, etc.).
- Extend `EntityBlockGeneration/Output` to support these sub-fields alongside the primary block value.
- Ensure prompts highlight how to populate each field and how to behave when data is missing.
- Update tests to cover contact scope behaviour and structured field aggregation.

---

## 4. Anchor Phrase & Region Hint Usage

Example configuration notes to keep in scope while implementing:

```yaml
- field_name: company_name
  description: Main supplier name shown in the invoice masthead
  type: entity-block
  probable_pages: [1]
  anchor_phrases:
    - invoice
    - supplier
  extra:
    region_hint: ["header"]
    include_bbox: true
```

- Ensure `anchor_phrases` feed into page retrieval and prompt instructions.
- `region_hint` accepts the allowed positional hints (`header`, `footer`, `left`, `right`, `center`, `top_left`, `top_right`, `bottom_left`, `bottom_right`). When multiple values are supplied, they are processed in order.
- `entity_type` can be used to emphasise target categories (company, person, address, etc.).
- `include_bbox` toggles bounding-box requests in prompts and output models.

---

## 5. Next Steps
1. Review these plans and sequence the implementation (entity-block first, then scope support/contact features).
2. Confirm prompt and schema designs before coding.
3. After approval, update `ADDING_NEW_EXTRACTION_TYPE.md`, README, and example configs alongside code changes.
