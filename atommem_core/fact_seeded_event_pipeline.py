# test_preextracted_pipeline_fact_seeded_event_level.py
# Pre-extracted facts pipeline with fact-seeded one-stage event attribution.

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List

import config
from src.fact_storage import FactStorageManager
from src.llm_interface import LLMInterface
from src.profile_manager import ProfileManager
from src.query_response import QueryResponder
from src.utils import cosine_similarity, jaccard_similarity
from atommem_core.preextracted_pipeline import PreExtractedFactsPipelineTester
from atommem_core.event_level_pipeline import (
    EventLevelEventManager,
    EventLevelPreExtractedFactsPipelineTester,
    SingleRoundQueryResponder,
)

def normalize_keyword_list(
    keywords: List[Any],
    people: List[str] = None,
    max_keywords: int = 5,
) -> List[str]:
    """Lightly canonicalize LLM-provided keywords without hand-built lexical rules."""
    _ = people
    normalized: List[str] = []
    seen = set()

    for keyword in keywords or []:
        item = normalize_single_keyword(keyword)
        if not item:
            continue

        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)

        if len(normalized) >= max_keywords:
            break

    return normalized


def normalize_single_keyword(keyword: Any) -> str:
    if not isinstance(keyword, str):
        return ""

    return re.sub(r"\s+", " ", keyword.strip().strip("'\"()[]{}.,;:，。；：")).lower()


class FactSeededEventLevelEventManager(EventLevelEventManager):
    """
    Event manager using fact-seeded event-level attribution.

    Flow:
    1. Retrieve top-k facts by the configured score.
    2. Convert those facts into candidates:
       - facts with event_ids contribute their existing events, deduplicated
       - facts without event_ids become singleton-fact candidates
    3. Let the LLM select existing events and/or singleton facts.
    """

    def _retrieve_top_event_candidates(
        self,
        new_fact: Dict[str, Any],
        all_facts: List[Dict[str, Any]],
        all_events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        top_facts = self._retrieve_top_similar_facts(new_fact, all_facts)
        if not top_facts:
            return []

        event_lookup = {event.get("event_id"): event for event in all_events}
        event_id_to_trigger_scores: Dict[str, List[float]] = {}
        singleton_candidates: List[Dict[str, Any]] = []

        for fact in top_facts:
            fact_event_ids = fact.get("event_ids") or []
            if fact_event_ids:
                for event_id in fact_event_ids:
                    if event_id not in event_lookup:
                        continue
                    event_id_to_trigger_scores.setdefault(event_id, []).append(
                        float(fact.get("score", 0.0))
                    )
            else:
                singleton_candidate = self._build_singleton_fact_candidate(fact, new_fact)
                if singleton_candidate:
                    singleton_candidate["score"] = fact.get("score", singleton_candidate["score"])
                    singleton_candidates.append(singleton_candidate)

        event_candidates: List[Dict[str, Any]] = []
        for event_id, trigger_scores in event_id_to_trigger_scores.items():
            event = event_lookup.get(event_id)
            if not event:
                continue

            candidate = self._build_existing_event_candidate(event, new_fact)
            if not candidate:
                continue

            candidate["event_summary_score"] = candidate["score"]
            candidate["score"] = max(trigger_scores)
            event_candidates.append(candidate)

        candidates = event_candidates + singleton_candidates
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    def _llm_generate_event_info(
        self,
        facts: List[Dict[str, Any]],
        existing_summary: str = None,
        existing_keywords: List[str] = None,
        is_incremental_update: bool = False,
    ) -> Dict[str, Any]:
        event_info = super()._llm_generate_event_info(
            facts,
            existing_summary=existing_summary,
            existing_keywords=existing_keywords,
            is_incremental_update=is_incremental_update,
        )
        people = []
        for fact in facts:
            people.extend(fact.get("people", []) or [])
        event_info["keywords"] = normalize_keyword_list(
            event_info.get("keywords", []),
            people=people,
        )
        return event_info

    def _retrieve_top_similar_facts(
        self,
        new_fact: Dict[str, Any],
        all_facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        new_people = set(new_fact.get("people", []) or [])

        for fact in all_facts:
            if fact.get("fact_id") == new_fact.get("fact_id"):
                continue
            fact_people = set(fact.get("people", []) or [])
            if not (new_people & fact_people):
                continue

            emb_sim = cosine_similarity(fact.get("embedding", []), new_fact.get("embedding", []))
            kw_sim = jaccard_similarity(fact.get("keywords", []), new_fact.get("keywords", []))
            score = (
                config.EVENT_ATTRIBUTION_EMBEDDING_WEIGHT * emb_sim
                + config.EVENT_ATTRIBUTION_KEYWORD_WEIGHT * kw_sim
            )

            scored_fact = dict(fact)
            scored_fact["score"] = score
            candidates.append(scored_fact)

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:config.EVENT_ATTRIBUTION_TOP_K]


class FactSeededEventLevelFactStorageManager(FactStorageManager):
    """Fact storage manager wired to fact-seeded event-level attribution."""

    def __init__(self, conversation_id: str):
        super().__init__(conversation_id)
        self.event_manager = FactSeededEventLevelEventManager()
        self.profile_manager = KeywordNormalizedProfileManager()


class KeywordNormalizedProfileManager(ProfileManager):
    """Profile manager that normalizes profile keywords after LLM extraction."""

    def _llm_extract_profiles(
        self,
        person: str,
        facts: List[Dict[str, Any]],
        existing_profiles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        profiles = super()._llm_extract_profiles(person, facts, existing_profiles)
        for profile in profiles:
            profile["keywords"] = normalize_keyword_list(
                profile.get("keywords", []),
                people=[person],
            )
        return profiles


class KeywordNormalizedSingleRoundQueryResponder(SingleRoundQueryResponder):
    """Single-round QA responder that normalizes query-intent keywords."""

    def _extract_query_intent(self, query: str) -> Dict[str, Any]:
        query_info = super()._extract_query_intent(query)
        query_info["keywords"] = normalize_keyword_list(
            query_info.get("keywords", []),
            people=query_info.get("people", []),
        )
        return query_info


class FactSeededEventLevelPreExtractedFactsPipelineTester(EventLevelPreExtractedFactsPipelineTester):
    """Pre-extracted pipeline tester using fact-seeded event-level attribution."""

    def __init__(
        self,
        conversation_id: str,
        output_dir: str = "runs/atommem",
    ):
        PreExtractedFactsPipelineTester.__init__(self, conversation_id, output_dir)
        self.fact_storage = FactSeededEventLevelFactStorageManager(conversation_id)
        self.query_responder = KeywordNormalizedSingleRoundQueryResponder(conversation_id, llm=self.llm)
        print("-> Event attribution mode: fact-seeded one-stage event-level retrieval")
        print("-> QA retrieval mode: single round with evidence summary")

    def extract_fact_metadata(
        self,
        fact_text: str,
        dia_id: str,
        session_time: str,
        speaker: str,
        previous_context: str = "",
    ) -> Dict[str, Any]:
        metadata = super().extract_fact_metadata(
            fact_text,
            dia_id,
            session_time,
            speaker,
            previous_context,
        )
        metadata["keywords"] = normalize_keyword_list(
            metadata.get("keywords", []),
            people=metadata.get("people", []),
        )
        return metadata
