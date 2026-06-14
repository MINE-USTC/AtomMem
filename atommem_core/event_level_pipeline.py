# Event-level memory construction.

import json
import time
from typing import Any, Dict, List, Tuple

import config
from src.event_manager import EventManager
from src.fact_storage import FactStorageManager
from src.query_response import QueryResponder
from src.utils import cosine_similarity, jaccard_similarity
from atommem_core.preextracted_pipeline import PreExtractedFactsPipeline


class EventLevelEventManager(EventManager):
    """Event manager using one-stage attribution over events and singleton facts."""

    def process_event_attribution(
        self,
        new_fact: Dict[str, Any],
        all_facts: List[Dict[str, Any]],
        all_events: List[Dict[str, Any]],
        use_batch_filter: bool = False,
    ) -> List[str]:
        """
        Attribute a new fact directly to top-k event-level candidates.

        Candidate types:
        - existing event: selecting it means adding the new fact to that event.
        - singleton fact: a fact with no event_ids; selecting it creates a separate
          new event containing only the new fact and that singleton fact.

        Multiple selections are allowed. Selecting multiple singleton facts creates
        multiple new events, one per singleton fact.
        """
        top_candidates = self._retrieve_top_event_candidates(new_fact, all_facts, all_events)
        if not top_candidates:
            return []

        selected_event_ids, selected_singleton_fact_ids, _reason = (
            self._llm_judge_event_level_attribution(new_fact, top_candidates)
        )

        attributed_event_ids: List[str] = []

        # Existing event selections: add the new fact and update each selected event.
        for event_id in selected_event_ids:
            event = next((e for e in all_events if e.get("event_id") == event_id), None)
            if not event:
                continue

            if "fact_ids" not in event:
                event["fact_ids"] = []

            if new_fact["fact_id"] not in event["fact_ids"]:
                event["fact_ids"].append(new_fact["fact_id"])
                self._update_event_info(event, new_fact, all_facts)

            attributed_event_ids.append(event_id)

        # Singleton fact selections: create one separate event for each selected fact.
        for fact_id in selected_singleton_fact_ids:
            singleton_fact = next((f for f in all_facts if f.get("fact_id") == fact_id), None)
            if not singleton_fact:
                continue
            if singleton_fact.get("fact_id") == new_fact.get("fact_id"):
                continue
            if singleton_fact.get("event_ids"):
                continue

            new_event = self._create_new_event(new_fact, [singleton_fact], all_events)
            if "event_ids" not in singleton_fact:
                singleton_fact["event_ids"] = []
            singleton_fact["event_ids"].append(new_event["event_id"])
            all_events.append(new_event)
            attributed_event_ids.append(new_event["event_id"])

        return list(dict.fromkeys(attributed_event_ids))

    def _retrieve_top_event_candidates(
        self,
        new_fact: Dict[str, Any],
        all_facts: List[Dict[str, Any]],
        all_events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k mixed candidates: existing events + singleton facts."""
        candidates: List[Dict[str, Any]] = []

        for event in all_events:
            candidate = self._build_existing_event_candidate(event, new_fact)
            if candidate:
                candidates.append(candidate)

        for fact in all_facts:
            if fact.get("fact_id") == new_fact.get("fact_id"):
                continue
            if fact.get("event_ids"):
                continue
            candidate = self._build_singleton_fact_candidate(fact, new_fact)
            if candidate:
                candidates.append(candidate)

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:config.EVENT_ATTRIBUTION_TOP_K]

    def _build_existing_event_candidate(
        self,
        event: Dict[str, Any],
        new_fact: Dict[str, Any],
    ) -> Dict[str, Any]:
        event_id = event.get("event_id")
        if not event_id:
            return {}

        summary = event.get("summary", "")
        embedding = event.get("embedding")
        if not embedding and summary:
            embedding = self.embedding_model.encode(summary)
            event["embedding"] = embedding

        score = self._score_candidate(new_fact, embedding or [], event.get("keywords", []))
        return {
            "candidate_id": event_id,
            "candidate_type": "event",
            "score": score,
            "summary": summary,
            "people": event.get("people", []),
            "keywords": event.get("keywords", []),
            "time": event.get("time", ["", ""]),
            "fact_ids": event.get("fact_ids", []),
        }

    def _build_singleton_fact_candidate(
        self,
        fact: Dict[str, Any],
        new_fact: Dict[str, Any],
    ) -> Dict[str, Any]:
        fact_id = fact.get("fact_id")
        if not fact_id:
            return {}

        score = self._score_candidate(new_fact, fact.get("embedding", []), fact.get("keywords", []))
        return {
            "candidate_id": fact_id,
            "candidate_type": "singleton_fact",
            "score": score,
            "fact": fact.get("fact", ""),
            "people": fact.get("people", []),
            "keywords": fact.get("keywords", []),
            "time": fact.get("time", ["", ""]),
        }

    def _score_candidate(
        self,
        new_fact: Dict[str, Any],
        candidate_embedding: List[float],
        candidate_keywords: List[str],
    ) -> float:
        emb_sim = cosine_similarity(candidate_embedding, new_fact.get("embedding", []))
        kw_sim = jaccard_similarity(candidate_keywords, new_fact.get("keywords", []))
        return (
            config.EVENT_ATTRIBUTION_EMBEDDING_WEIGHT * emb_sim
            + config.EVENT_ATTRIBUTION_KEYWORD_WEIGHT * kw_sim
        )

    def _llm_judge_event_level_attribution(
        self,
        new_fact: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str], str]:
        system_prompt = """You are an AI assistant specialized in event attribution.

Task: Determine which top-k event-level candidates describe the same event as the new fact.

Candidate types:
- "event": an existing event. Selecting it means the new fact should be added to that existing event.
- "singleton_fact": an independent fact that currently belongs to no event. Selecting it means creating a separate new event containing the new fact and that singleton fact.

Rules:
1. You may select multiple existing events.
2. You may select multiple singleton facts. Each selected singleton fact will create a separate new event with the new fact.
3. If you select both existing events and singleton facts, the new fact will both join the selected existing events and create separate events with each selected singleton fact.
4. Do not force attribution. If none are the same event, select nothing.
5. Focus on same occurrence, same subject-object relationship, or a directly shared event context. Consider the provided fact text and event summary carefully; keyword overlap alone is not enough.

Output JSON:
{
  "selected_event_ids": ["E1", "E3"] or [],
  "selected_singleton_fact_ids": ["F2", "F9"] or [],
  "reason": "Brief explanation"
}
"""

        prompt_candidates = []
        for candidate in candidates:
            item = {
                "candidate_id": candidate["candidate_id"],
                "candidate_type": candidate["candidate_type"],
                "keywords": candidate.get("keywords", []),
                "summary": (
                    candidate.get("summary", "")
                    if candidate["candidate_type"] == "event"
                    else candidate.get("fact", "")
                ),
            }
            prompt_candidates.append(item)

        user_prompt = f"""
New Fact:
- {new_fact.get("fact_id")}: {new_fact.get("fact", "")}

Top-{len(candidates)} Event-Level Candidates:
{json.dumps(prompt_candidates, indent=2, ensure_ascii=False)}

Select all candidates that describe the same event as the new fact.
"""

        response = self.llm.call_with_retry(
            system_prompt,
            user_prompt,
            response_format="json",
            call_type="event_attribution",
        )

        if "error" in response:
            print(f"  Warning: LLM error in event-level attribution: {response.get('error')}")
            return [], [], "LLM error"

        candidate_event_ids = {
            candidate["candidate_id"]
            for candidate in candidates
            if candidate["candidate_type"] == "event"
        }
        candidate_singleton_fact_ids = {
            candidate["candidate_id"]
            for candidate in candidates
            if candidate["candidate_type"] == "singleton_fact"
        }

        selected_event_ids = [
            item
            for item in self._normalize_id_list(response.get("selected_event_ids", []))
            if item in candidate_event_ids
        ]
        selected_singleton_fact_ids = [
            item
            for item in self._normalize_id_list(
                response.get("selected_singleton_fact_ids", response.get("selected_fact_ids", []))
            )
            if item in candidate_singleton_fact_ids
        ]
        reason = response.get("reason", "")

        return (
            list(dict.fromkeys(selected_event_ids)),
            list(dict.fromkeys(selected_singleton_fact_ids)),
            reason,
        )

    @staticmethod
    def _normalize_id_list(value: Any) -> List[str]:
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            return []

        normalized = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                candidate_id = (
                    item.get("candidate_id")
                    or item.get("event_id")
                    or item.get("fact_id")
                    or item.get("id")
                )
                if candidate_id:
                    normalized.append(candidate_id)
        return normalized


class EventLevelFactStorageManager(FactStorageManager):
    """Fact storage manager wired to the one-stage event-level attribution flow."""

    def __init__(self, conversation_id: str):
        super().__init__(conversation_id)
        self.event_manager = EventLevelEventManager()


class SingleRoundQueryResponder(QueryResponder):
    """Query responder that answers directly from one retrieval round."""

    def answer_query(self, query: str, category: Any = None) -> Dict[str, Any]:
        """Answer a query with exactly one retrieval round."""
        latency: Dict[str, float] = {}

        _t = time.time()
        query_info = self._extract_query_intent(query)
        latency["intent_extraction_ms"] = (time.time() - _t) * 1000

        _t = time.time()
        query_info["embedding"] = self.embedding_model.encode(query)
        latency["embedding_ms"] = (time.time() - _t) * 1000

        all_facts = self.storage.load_facts()
        all_events = self.storage.load_events()
        all_profiles = self.storage.load_profiles()

        _t = time.time()
        retrieval_result = self.retriever.retrieve_for_query(
            query_info,
            all_facts,
            all_events,
            all_profiles,
        )
        latency["retrieval_ms"] = (time.time() - _t) * 1000

        _t = time.time()
        answer = self._generate_answer(
            query,
            retrieval_result["facts"],
            retrieval_result["profiles"],
            retrieval_result["event_contexts"],
            query_info=query_info,
            category=category,
        )
        latency["answer_generation_ms"] = (time.time() - _t) * 1000

        evidence = [
            fact.get("dia_id")
            for fact in retrieval_result["facts"]
            if fact.get("dia_id")
        ]

        return {
            "query": query,
            "answer": answer,
            "retrieved_evidence": evidence,
            "retrieved_facts": retrieval_result["facts"],
            "retrieved_profiles": retrieval_result["profiles"],
            "event_contexts": retrieval_result["event_contexts"],
            "retrieval_rounds": 1,
            "query_info": {
                "need_specific": query_info.get("need_specific", False),
                "need_attribute": query_info.get("need_attribute", False),
                "people": query_info.get("people", []),
                "keywords": query_info.get("keywords", []),
                "time": query_info.get("time"),
            },
            "latency_breakdown": latency,
        }


class EventLevelPreExtractedFactsPipeline(PreExtractedFactsPipeline):
    """Pre-extracted pipeline using event-level attribution."""

    def __init__(self, conversation_id: str, output_dir: str = "runs/atommem"):
        super().__init__(conversation_id, output_dir)
        self.fact_storage = EventLevelFactStorageManager(conversation_id)
        self.query_responder = SingleRoundQueryResponder(conversation_id, llm=self.llm)

    def get_aggregate_llm_statistics(self) -> Dict[str, Any]:
        """Aggregate token statistics from every LLM-owning module."""
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


