# Fact-seeded event-level pipeline with incremental temporal profile updates.

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from src.fact_storage import FactStorageManager
from src.llm_interface import LLMInterface
from src.utils import generate_profile_id
from atommem_core.preextracted_pipeline import PreExtractedFactsPipeline
from atommem_core.fact_seeded_event_pipeline import (
    FactSeededEventLevelEventManager,
    KeywordNormalizedProfileManager,
    normalize_keyword_list,
)
from atommem_core.temporal_profile_version_chain import (
    TemporalProfileBuilder,
    TemporalProfileQueryResponder,
    compare_dates,
    normalize_date_value,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_OUTPUT_DIR = "runs/atommem"


def extract_profile_content_time(content: Any) -> str:
    """Best-effort fallback for profile statements like 'liked coffee in 2020'."""
    if not isinstance(content, str):
        return ""

    text = content.strip()
    date_match = re.search(r"\b(19\d{2}|20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", text)
    if date_match:
        return date_match.group(0)

    month_match = re.search(r"\b(19\d{2}|20\d{2})-(0[1-9]|1[0-2])\b", text)
    if month_match:
        return month_match.group(0)

    year_match = re.search(r"\b(?:in|during|since|before|after|by|until)\s+(19\d{2}|20\d{2})\b", text, re.I)
    if year_match:
        return year_match.group(1)

    return ""


class IncrementalTemporalProfileProcessor(TemporalProfileBuilder):
    """Apply the temporal profile decision flow to one newly extracted profile at a time."""

    def __init__(
        self,
        llm: LLMInterface,
        output_dir: str,
        conversation_id: str,
        top_k: int = 8,
        min_llm_score: float = 0.55,
    ):
        super().__init__(llm=llm, top_k=top_k, min_llm_score=min_llm_score)
        self.output_dir = output_dir
        self.conversation_id = conversation_id

    def _prepare_candidate(
        self,
        profile: Dict[str, Any],
        fact_lookup: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        item = super()._prepare_candidate(profile, fact_lookup)
        explicit_valid_from = normalize_date_value(profile.get("valid_from", ""))
        content_time = extract_profile_content_time(item.get("content", ""))
        if not explicit_valid_from and content_time:
            item["valid_from"] = content_time
        return item

    def process_candidate(
        self,
        candidate: Dict[str, Any],
        temporal_profiles: List[Dict[str, Any]],
        fact_lookup: Dict[str, Dict[str, Any]],
    ) -> bool:
        prepared = self._prepare_candidate(candidate, fact_lookup)
        if not prepared.get("content"):
            return False

        candidates = self._retrieve_candidates(prepared, temporal_profiles)
        top_score = candidates[0]["score"] if candidates else None

        if not candidates or top_score < self.min_llm_score:
            temporal_profiles.append(prepared)
            self.stats["created"] += 1
            return True

        direct_decision = self._direct_decision(prepared, candidates)
        if direct_decision:
            applied = self._apply_decision(prepared, temporal_profiles, direct_decision)
            return applied

        decision = self._llm_decide(prepared, candidates)
        applied = self._apply_decision(prepared, temporal_profiles, decision)
        if not applied:
            temporal_profiles.append(prepared)
            self.stats["fallback_new"] += 1
            return True

        return True

    def normalize_profiles(self, profiles: List[Dict[str, Any]]) -> None:
        self._normalize_timelines(profiles)


class IncrementalTemporalProfileManager(KeywordNormalizedProfileManager):
    """Extract candidate profiles in batches, then merge them incrementally into timelines."""

    def __init__(self, output_dir: str, conversation_id: str, top_k: int = 8, min_llm_score: float = 0.55):
        super().__init__()
        self.output_dir = output_dir
        self.conversation_id = conversation_id
        self.temporal_processor = IncrementalTemporalProfileProcessor(
            llm=self.llm,
            output_dir=output_dir,
            conversation_id=conversation_id,
            top_k=top_k,
            min_llm_score=min_llm_score,
        )
        self._reserved_profile_ids: List[Dict[str, Any]] = []

    def batch_extract_profiles(
        self,
        pending_facts: List[Dict[str, Any]],
        all_facts: List[Dict[str, Any]],
        all_profiles: List[Dict[str, Any]],
    ):
        if not pending_facts:
            return

        fact_lookup = {fact.get("fact_id"): fact for fact in all_facts if fact.get("fact_id")}

        facts_by_person: Dict[str, List[Dict[str, Any]]] = {}
        for fact in pending_facts:
            for person in fact.get("people", []):
                facts_by_person.setdefault(person, []).append(fact)

        extracted_candidates: List[Dict[str, Any]] = []
        for person, facts in facts_by_person.items():
            existing_profiles = [p for p in all_profiles if p.get("person") == person]
            new_profiles_data = self._llm_extract_profiles(person, facts, existing_profiles)
            for profile_data in new_profiles_data:
                candidate = self._build_candidate_profile(profile_data, person, all_profiles)
                if candidate:
                    extracted_candidates.append(candidate)

        for candidate in extracted_candidates:
            self.temporal_processor.process_candidate(
                candidate=candidate,
                temporal_profiles=all_profiles,
                fact_lookup=fact_lookup,
            )

        self.temporal_processor.normalize_profiles(all_profiles)

        for fact in pending_facts:
            fact["needs_profile_extraction"] = False

    def force_extract_all_pending(self, all_facts: List[Dict[str, Any]], all_profiles: List[Dict[str, Any]]):
        pending_facts = [f for f in all_facts if f.get("needs_profile_extraction", False)]
        if pending_facts:
            self.batch_extract_profiles(pending_facts, all_facts, all_profiles)

    def _build_candidate_profile(
        self,
        profile_data: Dict[str, Any],
        person: str,
        all_profiles: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(profile_data, dict):
            return None
        content = profile_data.get("content", "").strip()
        if not content:
            return None

        profile_id = self._next_profile_id(all_profiles)
        keywords = normalize_keyword_list(profile_data.get("keywords", []), people=[person])
        return {
            "profile_id": profile_id,
            "person": person,
            "content": content,
            "embedding": self.embedding_model.encode(content),
            "keywords": keywords,
            "evidence": profile_data.get("evidence", []),
            "valid_from": normalize_date_value(profile_data.get("valid_from", "")),
            "history": list(profile_data.get("history", []) or []),
        }

    def _next_profile_id(self, all_profiles: List[Dict[str, Any]]) -> str:
        profile_id = generate_profile_id(list(all_profiles) + self._reserved_profile_ids)
        self._reserved_profile_ids.append({"profile_id": profile_id})
        return profile_id

class IncrementalTemporalProfileFactStorageManager(FactStorageManager):
    """Fact storage manager wired to fact-seeded events and incremental temporal profiles."""

    def __init__(
        self,
        conversation_id: str,
        output_dir: str,
        profile_top_k: int = 8,
        min_profile_llm_score: float = 0.55,
    ):
        super().__init__(conversation_id)
        self.event_manager = FactSeededEventLevelEventManager()
        self.profile_manager = IncrementalTemporalProfileManager(
            output_dir=output_dir,
            conversation_id=conversation_id,
            top_k=profile_top_k,
            min_llm_score=min_profile_llm_score,
        )


class IncrementalTemporalProfileQueryResponder(TemporalProfileQueryResponder):
    """Temporal profile QA responder that handles query-intent time lists."""

    def _apply_profile_time_filter(
        self,
        profiles: List[Dict[str, Any]],
        query_time: Any,
    ) -> List[Dict[str, Any]]:
        query_times = self._normalize_query_times(query_time)
        if not query_times:
            return [self._current_profile_view(profile) for profile in profiles]

        selected = []
        for profile in profiles:
            current_valid_from = profile.get("valid_from", "")
            use_current = False
            for qt in query_times:
                cmp = compare_dates(qt, current_valid_from)
                if cmp is None or cmp >= 0:
                    use_current = True
                    break

            if use_current:
                selected.append(self._current_profile_view(profile))
                continue

            for qt in query_times:
                historical = self._history_view_for_time(profile, qt)
                if historical:
                    selected.append(historical)
                    break

        return selected

    @staticmethod
    def _normalize_query_times(query_time: Any) -> List[str]:
        raw_items = query_time if isinstance(query_time, list) else [query_time]
        result = []
        seen = set()
        for item in raw_items:
            value = normalize_date_value(item)
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result


class IncrementalTemporalProfilePipeline(PreExtractedFactsPipeline):
    """Full pre-extracted pipeline with incremental temporal profile construction."""

    def __init__(
        self,
        conversation_id: str,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        profile_top_k: int = 8,
        min_profile_llm_score: float = 0.55,
    ):
        PreExtractedFactsPipeline.__init__(self, conversation_id, output_dir)
        self.fact_storage = IncrementalTemporalProfileFactStorageManager(
            conversation_id=conversation_id,
            output_dir=output_dir,
            profile_top_k=profile_top_k,
            min_profile_llm_score=min_profile_llm_score,
        )
        self.query_responder = IncrementalTemporalProfileQueryResponder(conversation_id, llm=self.llm)

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

    def process_all_facts(self, dialogue_results: List[Dict[str, Any]], start_session: str = None):
        result = super().process_all_facts(dialogue_results, start_session=start_session)
        self._finalize_temporal_profiles()
        self.save_memory_snapshot()
        return result

    def _finalize_temporal_profiles(self) -> None:
        profiles = self.fact_storage.storage.load_profiles()
        self.fact_storage.profile_manager.temporal_processor.normalize_profiles(profiles)
        self.fact_storage.storage.save_profiles(profiles)

    def save_memory_snapshot(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        facts = self.fact_storage.storage.load_facts()
        events = self.fact_storage.storage.load_events()
        profiles = self.fact_storage.storage.load_profiles()
        for name, data in [("facts", facts), ("events", events), ("profiles", profiles)]:
            path = os.path.join(self.output_dir, f"{name}_{self.conversation_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def get_aggregate_llm_statistics(self) -> Dict[str, Any]:
        llm_instances = {
            "pipeline_llm": self.llm,
            "fact_storage_llm": self.fact_storage.llm,
            "event_manager_llm": self.fact_storage.event_manager.llm,
            "profile_manager_llm": self.fact_storage.profile_manager.llm,
        }
        totals = {
            "calls": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        for llm in llm_instances.values():
            stats = llm.get_statistics()
            totals["calls"] += stats.get("calls", 0)
            totals["total_tokens"] += stats.get("total_tokens", 0)
            totals["prompt_tokens"] += stats.get("prompt_tokens", 0)
            totals["completion_tokens"] += stats.get("completion_tokens", 0)
        totals["total_tokens_k"] = round(totals["total_tokens"] / 1000, 2)
        return totals




