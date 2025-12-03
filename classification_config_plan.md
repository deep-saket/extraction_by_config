# Classification Config Router Plan

## Goal
- Given a document (pdf/image) return the corresponding `de_config` filename.
- Store a new `classification_config` that lists candidate `de_config` files plus lightweight metadata to guide routing.

## Inputs / Outputs
- Input: document bytes/path + optional doc id.
- Output: selected `de_config` filename + confidence + (optional) rationale/log.

## Classification Config Schema (JSON)
- Stored as JSON (not YAML), e.g., `classification_config/*.json`.
- Top-level:
  - `candidates`: array of objects:
    - `file` (string): `de_config` filename to route to.
    - `text_hints` (optional list[string]): extra keywords/phrases to boost; these are additive on top of auto-derived signatures.
    - `structure` (optional dict): validation hints to down-rank mismatches, e.g.,
      - `page_count_range`: [min, max]
      - `has_tables`: bool
      - `must_contain_sections`: optional list[string] of expected headings (used only as soft checks via ColPali text queries)
      - `header_hints`: optional list[string] of header phrases expected across pages.
      - `footer_hints`: optional list[string] of footer phrases expected across pages.
      - `landing_page`: optional object to guide using a semantic “start page”:
        - `page_index` (int, 1-based): suggested landing/start page when first page is metadata.
        - `text_hints` (list[string]): phrases likely on the landing page.
        - `header_hints` / `footer_hints`: optional header/footer phrases specific to the landing page.
        - `consider_only`: bool; if true, classify using landing page only.
  - `thresholds`:
    - `min_score` (float): minimum score to accept a match; otherwise return unknown.
    - `top2_margin` (float): optional margin to decide if a tie-breaker is needed.
- `text_signatures` are **not** stored; they are auto-constructed from each candidate’s `de_config` by aggregating `field_name`, `description`, and `search_keys` from `ExtractionItem`s. `text_hints` is the only place for manual overrides/priority cues.

## Signals to Extract from Documents
- Stick to text-based cues; avoid template/anchor matching to reduce confusion.
- No OCR available: use minimal VLM/ColPali calls to extract text-like signals.
  - Page count (pdf metadata), table presence if cheap.
  - ColPali: score text queries against page images to approximate keyword hits without OCR.
  - Optional single VLM call on first page to extract a short list of key phrases if needed for tie-breaks.
- Reuse existing `PDFProcessor` to obtain page images/metadata for scoring.

## Algorithms
### 1) Naive heuristic (text queries via ColPali)
- For each candidate: use its `text_signatures` as queries against the document pages via ColPali and sum scores; combine with light structural hints (page count range, table presence flag).
- Pick top score if above threshold; else return unknown.

### 2) Hierarchical
- Stage 1: coarse routing using aggregated text signatures from each config’s ExtractionItems (e.g., most common `search_keys`/field names) to shortlist candidates with ColPali.
- Stage 2: run the naive ColPali query scorer on the shortlisted candidates.
- Optional Stage 3: tie-break with a single VLM call on first page to extract a few keywords only if the top-2 are close.

### 3) Faster / fewer-inference
- Optional fast path: precompute ColPali embeddings per config (outside the JSON) using their `text_signatures` over representative pages.
- At runtime, compute a ColPali embedding for the document’s landing page and do nearest-neighbor match against cached embeddings. Threshold to allow unknown.
- Cache doc embeddings if reused; VLM is only used as an optional tie-breaker when distances are too close.

## Wrapper Flow (runtime)
- Load `classification_config`.
- Extract doc signals once (page count, table hint, ColPali query scores for text signatures).
- Run configured strategy (default: hierarchical with fallback to fast embedding if provided).
- Return best `de_config` filename + confidence; log scores and thresholds for debugging.

## Algorithm Detail (using the JSON schema)
- Preload: For each candidate, auto-build `text_signatures` from its `de_config` (field_name + description + search_keys), append any candidate `text_hints`. Precompute per-candidate query list.
- Page selection:
  - Determine landing page index: default 1; override with `structure.landing_page.page_index` if present.
  - If `structure.landing_page.consider_only` is true, restrict classification to that page; otherwise score both landing page and first page (if different) with decay on non-landing pages.
- Scoring per candidate:
  - ColPali query scoring: run `text_signatures` (plus `landing_page.text_hints` if provided) against chosen pages; sum/average scores.
  - Header/footer hints: if present, run separate ColPali queries targeting header/footer strips; add weighted bonus if found.
  - Structure checks: apply soft penalties/bonuses for page_count_range, has_tables, must_contain_sections.
  - Normalize to produce a candidate score.
- Selection:
  - Take top-1 by score; require `min_score` from thresholds.
  - If `top2_margin` is set and top-2 are within margin, optionally trigger a single-page VLM keyword extraction on landing page to break the tie.
  - If below `min_score`, return unknown.

## Next Steps
- Confirm final schema and placement for `classification_config` files (e.g., `classification_config/*.json`).
- Inventory current `de_config` files and draft metadata (text signatures from ExtractionItems, structure cues where available).
- Implement wrapper class with pluggable strategies + logging.
- Add smoke tests with sample docs to verify routing and unknown handling.

## Implementation Plan (code)
- Add a `classification_io` package mirroring `extraction_io` style:
  - `classification_io/ClassificationItems.py`: Pydantic models for `ClassificationCandidate` (file, optional text_hints, optional structure with header/footer/landing_page/page_count/has_tables/must_contain_sections) and `ClassificationConfig` (candidates + thresholds).
  - `classification_io/ClassificationState.py`: class-level state (pages/images, embeddings if cached, current candidate/context) similar to `ExtractionState`.
  - `classification_io/__init__.py`: exports models/loaders/state.
- Add a `visual_classify` package for runtime logic:
  - `visual_classify/Classifier.py`: loader for classification config JSON, builds text signatures from `de_config` via `ExtractionItems`, merges `text_hints`, runs routing strategies (naive/hierarchical/fast).
  - `visual_classify/__init__.py` and helpers consistent with repo conventions.
- Integrate `PDFProcessor` to generate page images/metadata once per document for ColPali scoring and structural checks.
- Implement signature builder: parse `de_config/*.json` using existing `ExtractionItems` loader; aggregate `field_name`, `description`, `search_keys`; append candidate `text_hints`.
- Implement scoring utilities:
  - ColPali query scoring on landing/first page with landing-page logic and optional landing-only mode.
  - Header/footer hint scoring and structure-based soft penalties/bonuses.
  - Threshold/top2-margin handling with optional single-page VLM tie-breaker.
- Add smoke tests or a small script to exercise classification across a couple of existing `de_config` files with mock pages.
