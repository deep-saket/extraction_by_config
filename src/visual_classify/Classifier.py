import json
import os
from typing import List, Optional, Tuple, Dict
import torch

from common import BaseComponent
from classification_io import ClassificationConfig, ClassificationCandidate, ClassificationState
from extraction_io.ExtractionItems import ExtractionItems
from vector_retrieve import PDFProcessor
from models import ModelManager
from config.loader import settings


class Classifier(BaseComponent):
    """
    Routes a document to the best matching de_config using ColPali text queries.
    """

    def __init__(self):
        classify_cfg = settings.get("parser", {}).get("args", {})
        self.device = torch.device(classify_cfg.get("device", "cpu"))
        self.embedding_candidate = classify_cfg.get("embedding_candidate", "ColPaliInfer")

        super().__init__(classify_cfg)

        ModelManager.initialize_models(self.device, model_classes=[self.embedding_candidate])
        self.colpali_infer = getattr(ModelManager, self.embedding_candidate)
        self.pdf_processor = PDFProcessor(self.colpali_infer)

    def load_config(self, config_path: str) -> ClassificationConfig:
        """
        Load and validate the classification config JSON file.
        """
        with open(config_path, "r") as f:
            data = json.load(f)
        cfg = ClassificationConfig.model_validate(data)
        ClassificationState.set_config(cfg)
        return cfg

    def classify(self, pdf_path: str, classification_config_path: str) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Entry point: load config, prepare embeddings, score each candidate, and select the best.
        """
        cfg = self.load_config(classification_config_path)
        ClassificationState.reset()
        ClassificationState.set_config(cfg)

        images, embeddings = self._prepare_embeddings(pdf_path)
        ClassificationState.set_images(images)
        ClassificationState.set_embeddings(embeddings)

        scores = {}
        for candidate in cfg.candidates:
            score = self._score_candidate(candidate, images, embeddings)
            scores[candidate.file] = score
            ClassificationState.set_score(candidate.file, score)

        best_file = None
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            best_file, best_score = sorted_scores[0]
            if best_score < cfg.thresholds.min_score:
                best_file = None
            elif len(sorted_scores) > 1 and (best_score - sorted_scores[1][1]) < cfg.thresholds.top2_margin:
                # Tie-breaker placeholder; keep best_file as-is for now.
                pass
        return best_file, scores

    def _prepare_embeddings(self, pdf_path: str) -> Tuple[List[Tuple[int, str]], List[Tuple[int, torch.Tensor]]]:
        """
        Convert PDF pages to images and embeddings using PDFProcessor utilities.
        """
        images = self.pdf_processor.pdf_to_images(pdf_path)
        embeddings = self.pdf_processor.generate_embeddings(images)
        return images, embeddings

    def _resolve_de_config_path(self, file_name: str) -> str:
        """
        Resolve the de_config path for a candidate.
        """
        if os.path.isabs(file_name) and os.path.exists(file_name):
            return file_name
        if os.path.exists(file_name):
            return file_name
        fallback = os.path.join("de_config", file_name)
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(f"Could not resolve de_config path for '{file_name}'")

    def _build_text_signatures(self, candidate: ClassificationCandidate) -> List[str]:
        """
        Build text signatures from the referenced de_config plus optional text_hints.
        """
        de_config_path = self._resolve_de_config_path(candidate.file)
        with open(de_config_path, "r") as f:
            items_raw = json.load(f)
        items = ExtractionItems.model_validate(items_raw)
        phrases: List[str] = []
        for item in items:
            phrases.append(item.field_name)
            if item.description:
                phrases.append(item.description)
            if item.search_keys:
                phrases.extend(item.search_keys)
        phrases = [p for p in phrases if p]
        phrases.extend(candidate.text_hints or [])
        # De-duplicate while preserving order
        seen = set()
        deduped = []
        for p in phrases:
            if p not in seen:
                deduped.append(p)
                seen.add(p)
        return deduped

    def _score_candidate(
        self,
        candidate: ClassificationCandidate,
        images: List[Tuple[int, str]],
        embeddings: List[Tuple[int, torch.Tensor]],
    ) -> float:
        """
        Score a single candidate using ColPali queries plus optional structure hints.
        """
        signatures = self._build_text_signatures(candidate)
        structure = candidate.structure
        landing_idx = 1
        landing_only = False
        landing_text_hints: List[str] = []
        if structure and structure.landing_page:
            lp_index = structure.landing_page.page_index or 1
            landing_idx = max(1, lp_index)
            landing_only = structure.landing_page.consider_only
            landing_text_hints = structure.landing_page.text_hints or []

        allowed_pages = {landing_idx} if landing_only else {1, landing_idx}
        allowed_embeddings = [(p, emb) for (p, emb) in embeddings if p in allowed_pages]
        if not allowed_embeddings:
            allowed_embeddings = embeddings

        base_score = self._score_queries(signatures, allowed_embeddings)

        landing_bonus = 0.0
        if landing_text_hints:
            landing_embeddings = [(p, e) for (p, e) in embeddings if p == landing_idx]
            landing_bonus = 0.2 * self._score_queries(landing_text_hints, landing_embeddings or allowed_embeddings)

        header_footer_bonus = 0.0
        if structure:
            header_footer_queries = (structure.header_hints or []) + (structure.footer_hints or [])
            if header_footer_queries:
                header_footer_bonus = 0.1 * self._score_queries(header_footer_queries, embeddings)
            if structure.must_contain_sections:
                base_score += 0.1 * self._score_queries(structure.must_contain_sections, embeddings)

        structure_bonus = 0.0
        if structure and structure.page_count_range:
            page_count = len(images)
            min_page, max_page = structure.page_count_range[0], structure.page_count_range[1]
            if page_count < min_page or page_count > max_page:
                structure_bonus -= 0.05
            else:
                structure_bonus += 0.05

        total_score = base_score + landing_bonus + header_footer_bonus + structure_bonus
        return float(total_score)

    def _score_queries(
        self,
        queries: List[str],
        embeddings: List[Tuple[int, torch.Tensor]]
    ) -> float:
        """
        Average the max similarity for each query over provided embeddings.
        """
        if not queries or not embeddings:
            return 0.0
        page_embeddings = torch.stack([emb for _, emb in embeddings], dim=0)

        # Normalize shapes to match ColPali score_multi_vector expectations:
        #   queries: (B, N, D), passages: (C, S, D)
        if page_embeddings.dim() == 2:
            # (num_pages, dim) -> (num_pages, 1, dim)
            page_embeddings = page_embeddings.unsqueeze(1)
        elif page_embeddings.dim() == 3:
            # already (num_pages, seq_len, dim)
            pass
        elif page_embeddings.dim() == 4:
            # squeeze potential batch dimension: (num_pages, 1, seq_len, dim) -> (num_pages, seq_len, dim)
            page_embeddings = page_embeddings.squeeze(1)
        else:
            raise ValueError(f"Unexpected page_embeddings shape: {page_embeddings.shape}")

        page_embeddings = page_embeddings.to(self.device)

        scores = []
        for query in queries:
            query_emb = self.colpali_infer.get_text_embedding(query)
            if query_emb.dim() == 2 and query_emb.shape[0] == 1:
                query_emb = query_emb.squeeze(0).unsqueeze(0)
            if query_emb.dim() == 2:
                # (1, dim) -> (1, 1, dim)
                query_emb = query_emb.unsqueeze(0)
            query_emb = query_emb.to(self.device)
            sim = self.colpali_infer.processor.score_multi_vector(query_emb, page_embeddings)
            sim = sim.squeeze(0)
            scores.append(float(sim.max().detach().cpu()))
        return sum(scores) / len(scores)
