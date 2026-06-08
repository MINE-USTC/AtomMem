# test_temporal_profile_version_chain.py
# Sidecar experiment for temporal profile version chains.

import argparse
import copy
import json
import os
import re
import sys
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import config
from src.embedding import EmbeddingModel
from src.llm_interface import LLMInterface
from src.utils import cosine_similarity, jaccard_similarity
from atommem_core.fact_seeded_event_pipeline import (
    FactSeededEventLevelPreExtractedFactsPipelineTester,
    KeywordNormalizedSingleRoundQueryResponder,
    normalize_keyword_list,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


TEMPORAL_UPDATE_PROMPT = os.path.join(config.PROMPTS_DIR, "profile_temporal_update_prompt.txt")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def profile_sort_key(profile: Dict[str, Any]) -> Tuple[int, str]:
    profile_id = profile.get("profile_id", "")
    match = re.match(r"^P(\d+)$", profile_id)
    return (int(match.group(1)) if match else 10**9, profile_id)


def fact_sort_key(fact_id: str) -> Tuple[int, str]:
    match = re.match(r"^F(\d+)$", fact_id or "")
    return (int(match.group(1)) if match else 10**9, fact_id or "")


def normalize_date_value(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    if re.fullmatch(r"\d{4}-\d{2}", value):
        return value
    if re.fullmatch(r"\d{4}", value):
        return value
    return ""


def date_start(value: Any) -> Optional[date]:
    value = normalize_date_value(value)
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            y, m, d = [int(x) for x in value.split("-")]
            return date(y, m, d)
        if re.fullmatch(r"\d{4}-\d{2}", value):
            y, m = [int(x) for x in value.split("-")]
            return date(y, m, 1)
        if re.fullmatch(r"\d{4}", value):
            return date(int(value), 1, 1)
    except ValueError:
        return None
    return None


def compare_dates(a: Any, b: Any) -> Optional[int]:
    da = date_start(a)
    db = date_start(b)
    if da is None or db is None:
        return None
    if da < db:
        return -1
    if da > db:
        return 1
    return 0


def time_in_interval(query_time: Any, valid_from: Any, valid_to: Any) -> bool:
    qt = date_start(query_time)
    vf = date_start(valid_from)
    vt = date_start(valid_to)
    if qt is None or vf is None:
        return False
    if qt < vf:
        return False
    if vt is not None and qt >= vt:
        return False
    return True


def unique_list(items: List[Any]) -> List[Any]:
    result = []
    seen = set()
    for item in items or []:
        if not item:
            continue
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def union_keywords(a: List[Any], b: List[Any], people: List[str]) -> List[str]:
    return normalize_keyword_list(list(a or []) + list(b or []), people=people)


def evidence_time(profile: Dict[str, Any], fact_lookup: Dict[str, Dict[str, Any]]) -> str:
    times = []
    for fact_id in sorted(profile.get("evidence", []) or [], key=fact_sort_key):
        fact = fact_lookup.get(fact_id)
        if not fact:
            continue
        time_info = fact.get("time") or []
        event_time = time_info[0] if len(time_info) > 0 else ""
        interaction_time = time_info[1] if len(time_info) > 1 else ""
        chosen = normalize_date_value(event_time) or normalize_date_value(interaction_time)
        if chosen:
            times.append(chosen)
    if not times:
        return ""
    times.sort(key=lambda t: date_start(t) or date.max)
    return times[0]


def clone_current_to_history(profile: Dict[str, Any], valid_to: str) -> Optional[Dict[str, Any]]:
    content = profile.get("content", "").strip()
    if not content:
        return None
    if not valid_to:
        return None
    valid_from = profile.get("valid_from", "")
    if valid_from and valid_to and compare_dates(valid_from, valid_to) == 0:
        return None
    history = profile.setdefault("history", [])
    version_id = f"{profile.get('profile_id')}_v{len(history) + 1}"
    return {
        "version_id": version_id,
        "content": content,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "evidence": list(profile.get("evidence", []) or []),
    }


class TemporalProfileBuilder:
    def __init__(self, llm: LLMInterface, top_k: int = 8, min_llm_score: float = 0.55):
        self.llm = llm
        self.top_k = top_k
        self.min_llm_score = min_llm_score
        self.embedding_model = EmbeddingModel()
        with open(TEMPORAL_UPDATE_PROMPT, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
        self.stats = {
            "input_profiles": 0,
            "output_profiles": 0,
            "llm_calls_attempted": 0,
            "created": 0,
            "confirm": 0,
            "update_current": 0,
            "update_history": 0,
            "fallback_new": 0,
        }

    def build(
        self,
        additive_profiles: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        limit_profiles: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        fact_lookup = {fact.get("fact_id"): fact for fact in facts if fact.get("fact_id")}
        temporal_profiles: List[Dict[str, Any]] = []
        source_profiles = sorted(additive_profiles, key=profile_sort_key)
        if limit_profiles:
            source_profiles = source_profiles[:limit_profiles]
        self.stats["input_profiles"] = len(source_profiles)

        for index, candidate in enumerate(source_profiles, 1):
            prepared = self._prepare_candidate(candidate, fact_lookup)
            if not prepared.get("content"):
                continue

            candidates = self._retrieve_candidates(prepared, temporal_profiles)
            if not candidates or candidates[0]["score"] < self.min_llm_score:
                temporal_profiles.append(prepared)
                self.stats["created"] += 1
                continue

            direct_decision = self._direct_decision(prepared, candidates)
            if direct_decision:
                self._apply_decision(prepared, temporal_profiles, direct_decision)
                continue

            print(
                f"[profile {index}/{len(source_profiles)}] "
                f"{prepared.get('profile_id')} top={candidates[0]['profile_id']} "
                f"score={candidates[0]['score']:.3f}"
            )
            decision = self._llm_decide(prepared, candidates)
            applied = self._apply_decision(prepared, temporal_profiles, decision)
            if not applied:
                temporal_profiles.append(prepared)
                self.stats["fallback_new"] += 1

        self._normalize_timelines(temporal_profiles)
        self.stats["output_profiles"] = len(temporal_profiles)
        return temporal_profiles

    def _prepare_candidate(
        self,
        profile: Dict[str, Any],
        fact_lookup: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        item = copy.deepcopy(profile)
        content = item.get("content", "").strip()
        item["content"] = content
        item["person"] = item.get("person", "")
        item["keywords"] = normalize_keyword_list(
            item.get("keywords", []),
            people=[item.get("person", "")],
        )
        item["evidence"] = unique_list(item.get("evidence", []))
        item["valid_from"] = normalize_date_value(item.get("valid_from", "")) or evidence_time(item, fact_lookup)
        item["history"] = list(item.get("history", []) or [])
        if not item.get("embedding") and content:
            item["embedding"] = self.embedding_model.encode(content)
        return item

    def _retrieve_candidates(
        self,
        candidate: Dict[str, Any],
        temporal_profiles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        scored = []
        for profile in temporal_profiles:
            if profile.get("person") != candidate.get("person"):
                continue
            emb_sim = cosine_similarity(candidate.get("embedding", []), profile.get("embedding", []))
            kw_sim = jaccard_similarity(candidate.get("keywords", []), profile.get("keywords", []))
            score = config.EMBEDDING_WEIGHT_ALPHA * emb_sim + config.KEYWORD_WEIGHT_BETA * kw_sim
            row = {
                "profile_id": profile.get("profile_id"),
                "score": score,
                "embedding_similarity": emb_sim,
                "keyword_jaccard": kw_sim,
                "profile": profile,
            }
            scored.append(row)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: self.top_k]

    def _direct_decision(
        self,
        candidate: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        top = candidates[0] if candidates else None
        if not top:
            return None
        profile = top["profile"]
        same_text = self._normalized_text(candidate.get("content")) == self._normalized_text(profile.get("content"))
        if not same_text:
            return None

        action = "confirm"
        current_valid_from = profile.get("valid_from", "")
        candidate_valid_from = candidate.get("valid_from", "")
        if current_valid_from and candidate_valid_from and compare_dates(candidate_valid_from, current_valid_from) == -1:
            action = "update_history"

        self.stats["deterministic_confirm"] = self.stats.get("deterministic_confirm", 0) + 1
        return {
            "decisions": [{
                "profile_id": profile.get("profile_id"),
                "action": action,
                "updated_content": candidate.get("content", ""),
                "target_version_id": "",
                "evidence": candidate.get("evidence", []),
            }],
            "new_profile": False,
            "new_content": "",
        }

    def _llm_decide(self, candidate: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.stats["llm_calls_attempted"] += 1
        user_prompt = self._build_user_prompt(candidate, candidates)
        response = self.llm.call_with_retry(
            self.system_prompt,
            user_prompt,
            response_format="json",
            call_type="profile_temporal_update",
        )
        if "error" in response or "raw_content" in response:
            return self._fallback_decision(candidate, candidates)
        return response

    def _build_user_prompt(self, candidate: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
        candidate_block = {
            "person": candidate.get("person"),
            "content": candidate.get("content"),
            "keywords": candidate.get("keywords", []),
            "valid_from": candidate.get("valid_from", ""),
            "evidence": candidate.get("evidence", []),
        }
        rows = []
        for row in candidates:
            profile = row["profile"]
            compact_history = []
            for h in profile.get("history", [])[-3:]:
                compact_history.append({
                    "version_id": h.get("version_id", ""),
                    "content": h.get("content", ""),
                    "valid_from": h.get("valid_from", ""),
                    "valid_to": h.get("valid_to", ""),
                })
            rows.append({
                "profile_id": profile.get("profile_id"),
                "score": round(row["score"], 3),
                "content": profile.get("content"),
                "keywords": profile.get("keywords", []),
                "valid_from": profile.get("valid_from", ""),
                "history": compact_history,
            })
        return "Candidate:\n" + json.dumps(candidate_block, ensure_ascii=False) + "\n\nProfiles:\n" + json.dumps(rows, ensure_ascii=False)

    def _fallback_decision(self, candidate: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        top = candidates[0] if candidates else None
        if not top or top["score"] < 0.88:
            return {"decisions": [], "new_profile": True, "new_content": candidate.get("content", "")}
        profile = top["profile"]
        action = "confirm"
        if self._normalized_text(candidate.get("content")) != self._normalized_text(profile.get("content")):
            action = "update_current"
        return {
            "decisions": [{
                "profile_id": profile.get("profile_id"),
                "action": action,
                "updated_content": candidate.get("content", ""),
                "target_version_id": "",
                "evidence": candidate.get("evidence", []),
            }],
            "new_profile": False,
            "new_content": "",
        }

    @staticmethod
    def _normalized_text(text: Any) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()

    def _apply_decision(
        self,
        candidate: Dict[str, Any],
        temporal_profiles: List[Dict[str, Any]],
        decision: Dict[str, Any],
    ) -> bool:
        if not isinstance(decision, dict):
            return False

        decisions = decision.get("decisions", [])
        if not isinstance(decisions, list):
            decisions = []

        applied_any = False
        for item in decisions:
            if not isinstance(item, dict):
                continue
            profile = self._find_profile(temporal_profiles, item.get("profile_id"))
            if not profile:
                continue
            action = item.get("action")
            if action == "confirm":
                self._apply_confirm(profile, candidate, item)
            elif action == "update_current":
                self._apply_update_current(profile, candidate, item)
            elif action == "update_history":
                self._apply_update_history(profile, candidate, item)
            else:
                continue
            self.stats[action] += 1
            applied_any = True

        if decision.get("new_profile") is True:
            new_item = copy.deepcopy(candidate)
            if isinstance(decision.get("new_content"), str) and decision["new_content"].strip():
                new_item["content"] = decision["new_content"].strip()
                new_item["embedding"] = self.embedding_model.encode(new_item["content"])
            temporal_profiles.append(new_item)
            self.stats["created"] += 1
            applied_any = True

        return applied_any

    @staticmethod
    def _find_profile(profiles: List[Dict[str, Any]], profile_id: Any) -> Optional[Dict[str, Any]]:
        for profile in profiles:
            if profile.get("profile_id") == profile_id:
                return profile
        return None

    def _apply_confirm(self, profile: Dict[str, Any], candidate: Dict[str, Any], item: Dict[str, Any]) -> None:
        target_version_id = item.get("target_version_id")
        evidence = unique_list(item.get("evidence", []) or candidate.get("evidence", []))
        if target_version_id:
            history = self._find_history(profile, target_version_id)
            if history:
                history["evidence"] = unique_list(list(history.get("evidence", [])) + evidence)
                return
        profile["evidence"] = unique_list(list(profile.get("evidence", [])) + evidence)

    def _apply_update_current(self, profile: Dict[str, Any], candidate: Dict[str, Any], item: Dict[str, Any]) -> None:
        current_valid_from = normalize_date_value(profile.get("valid_from", ""))
        effective_time = normalize_date_value(candidate.get("valid_from", "")) or current_valid_from
        if current_valid_from and effective_time and compare_dates(effective_time, current_valid_from) == -1:
            self._apply_update_history(profile, candidate, item)
            return

        history_item = clone_current_to_history(profile, effective_time)
        if history_item:
            profile.setdefault("history", []).append(history_item)

        updated_content = item.get("updated_content") or candidate.get("content", "")
        profile["content"] = updated_content.strip()
        profile["valid_from"] = effective_time or current_valid_from
        profile["keywords"] = union_keywords(profile.get("keywords", []), candidate.get("keywords", []), [profile.get("person", "")])
        profile["evidence"] = unique_list(list(item.get("evidence", []) or candidate.get("evidence", [])))
        profile["embedding"] = self.embedding_model.encode(profile["content"])

    def _apply_update_history(self, profile: Dict[str, Any], candidate: Dict[str, Any], item: Dict[str, Any]) -> None:
        effective_time = normalize_date_value(candidate.get("valid_from", ""))
        evidence = unique_list(item.get("evidence", []) or candidate.get("evidence", []))
        updated_content = (item.get("updated_content") or candidate.get("content", "")).strip()

        if not effective_time:
            profile["evidence"] = unique_list(list(profile.get("evidence", [])) + evidence)
            return

        target = self._find_history(profile, item.get("target_version_id"))
        if not target:
            target = self._find_history_by_time(profile, effective_time)

        if target:
            target["content"] = updated_content or target.get("content", "")
            target["evidence"] = unique_list(list(target.get("evidence", [])) + evidence)
            return

        history = profile.setdefault("history", [])
        next_valid_from = self._next_valid_from(profile, effective_time)
        version = {
            "version_id": f"{profile.get('profile_id')}_v{len(history) + 1}",
            "content": updated_content,
            "valid_from": effective_time,
            "valid_to": next_valid_from,
            "evidence": evidence,
        }
        history.append(version)
        history.sort(key=lambda h: date_start(h.get("valid_from")) or date.max)
        self._renumber_history(profile)

    @staticmethod
    def _find_history(profile: Dict[str, Any], version_id: Any) -> Optional[Dict[str, Any]]:
        if not version_id:
            return None
        for item in profile.get("history", []) or []:
            if item.get("version_id") == version_id:
                return item
        return None

    @staticmethod
    def _find_history_by_time(profile: Dict[str, Any], effective_time: str) -> Optional[Dict[str, Any]]:
        if not effective_time:
            return None
        for item in profile.get("history", []) or []:
            if time_in_interval(effective_time, item.get("valid_from"), item.get("valid_to")):
                return item
        return None

    @staticmethod
    def _next_valid_from(profile: Dict[str, Any], effective_time: str) -> str:
        candidates = []
        current_vf = profile.get("valid_from", "")
        if current_vf and compare_dates(effective_time, current_vf) == -1:
            candidates.append(current_vf)
        for item in profile.get("history", []) or []:
            vf = item.get("valid_from", "")
            if vf and compare_dates(effective_time, vf) == -1:
                candidates.append(vf)
        candidates.sort(key=lambda t: date_start(t) or date.max)
        return candidates[0] if candidates else current_vf

    @staticmethod
    def _renumber_history(profile: Dict[str, Any]) -> None:
        for idx, item in enumerate(profile.get("history", []) or [], 1):
            item["version_id"] = f"{profile.get('profile_id')}_v{idx}"

    def _normalize_timelines(self, profiles: List[Dict[str, Any]]) -> None:
        for profile in profiles:
            rows = []
            missing_evidence = []

            current_vf = normalize_date_value(profile.get("valid_from", ""))
            if current_vf:
                rows.append({
                    "content": profile.get("content", ""),
                    "valid_from": current_vf,
                    "evidence": list(profile.get("evidence", []) or []),
                    "order": 10**6,
                })
            else:
                missing_evidence.extend(profile.get("evidence", []) or [])

            for order, item in enumerate(profile.get("history", []) or []):
                vf = normalize_date_value(item.get("valid_from", ""))
                if not vf:
                    missing_evidence.extend(item.get("evidence", []) or [])
                    continue
                rows.append({
                    "content": item.get("content", ""),
                    "valid_from": vf,
                    "evidence": list(item.get("evidence", []) or []),
                    "order": order,
                })

            if not rows:
                profile["history"] = []
                profile["evidence"] = unique_list(list(profile.get("evidence", [])) + missing_evidence)
                continue

            rows.sort(key=lambda row: (date_start(row["valid_from"]) or date.max, row["order"]))
            grouped = []
            for row in rows:
                if grouped and grouped[-1]["valid_from"] == row["valid_from"]:
                    if row.get("content"):
                        grouped[-1]["content"] = row["content"]
                    grouped[-1]["evidence"] = unique_list(grouped[-1].get("evidence", []) + row.get("evidence", []))
                else:
                    grouped.append({
                        "content": row.get("content", ""),
                        "valid_from": row.get("valid_from", ""),
                        "evidence": unique_list(row.get("evidence", [])),
                    })

            current = grouped[-1]
            current_evidence = unique_list(current.get("evidence", []) + missing_evidence)
            profile["content"] = current.get("content", profile.get("content", ""))
            profile["valid_from"] = current.get("valid_from", profile.get("valid_from", ""))
            profile["evidence"] = current_evidence
            if profile.get("content"):
                profile["embedding"] = self.embedding_model.encode(profile["content"])

            history = []
            for idx, row in enumerate(grouped[:-1], 1):
                history.append({
                    "version_id": f"{profile.get('profile_id')}_v{idx}",
                    "content": row.get("content", ""),
                    "valid_from": row.get("valid_from", ""),
                    "valid_to": grouped[idx].get("valid_from", ""),
                    "evidence": row.get("evidence", []),
                })
            profile["history"] = history
        self.stats["timeline_normalized"] = True


class TemporalProfileQueryResponder(KeywordNormalizedSingleRoundQueryResponder):
    """Single-round QA with temporal profile version selection after Top-k retrieval."""

    def answer_query(self, query: str) -> Dict[str, Any]:
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
        retrieval_result["profiles"] = self._apply_profile_time_filter(
            retrieval_result["profiles"],
            query_info.get("time"),
        )
        latency["retrieval_ms"] = (time.time() - _t) * 1000

        _t = time.time()
        evidence_summary = self._summarize_retrieved_evidence(
            query,
            query_info,
            retrieval_result["facts"],
            retrieval_result["profiles"],
        )
        latency["evidence_summary_ms"] = (time.time() - _t) * 1000

        _t = time.time()
        answer = self._generate_answer(
            query,
            retrieval_result["facts"],
            retrieval_result["profiles"],
            retrieval_result["event_contexts"],
            query_info=query_info,
            evidence_summary=evidence_summary,
        )
        latency["answer_generation_ms"] = (time.time() - _t) * 1000

        evidence = [fact.get("dia_id") for fact in retrieval_result["facts"] if fact.get("dia_id")]
        return {
            "query": query,
            "answer": answer,
            "retrieved_evidence": evidence,
            "retrieved_facts": retrieval_result["facts"],
            "retrieved_profiles": retrieval_result["profiles"],
            "event_contexts": retrieval_result["event_contexts"],
            "evidence_summary": evidence_summary,
            "retrieval_rounds": 1,
            "no_useful_retrieval_rounds": 0,
            "stopped_for_insufficient_evidence": False,
            "query_info": {
                "need_specific": query_info.get("need_specific", False),
                "need_attribute": query_info.get("need_attribute", False),
                "people": query_info.get("people", []),
                "keywords": query_info.get("keywords", []),
                "time": query_info.get("time"),
            },
            "latency_breakdown": latency,
        }

    def _apply_profile_time_filter(
        self,
        profiles: List[Dict[str, Any]],
        query_time: Any,
    ) -> List[Dict[str, Any]]:
        if not query_time:
            return [self._current_profile_view(profile) for profile in profiles]

        selected = []
        for profile in profiles:
            current_valid_from = profile.get("valid_from", "")
            cmp = compare_dates(query_time, current_valid_from)
            if cmp is None or cmp >= 0:
                selected.append(self._current_profile_view(profile))
                continue

            historical = self._history_view_for_time(profile, query_time)
            if historical:
                selected.append(historical)
        return selected

    @staticmethod
    def _current_profile_view(profile: Dict[str, Any]) -> Dict[str, Any]:
        item = copy.deepcopy(profile)
        item.pop("history", None)
        return item

    @staticmethod
    def _history_view_for_time(profile: Dict[str, Any], query_time: Any) -> Optional[Dict[str, Any]]:
        for history in profile.get("history", []) or []:
            if time_in_interval(query_time, history.get("valid_from"), history.get("valid_to")):
                item = copy.deepcopy(profile)
                item["content"] = history.get("content", "")
                item["evidence"] = history.get("evidence", [])
                item["valid_from"] = history.get("valid_from", "")
                item["valid_to"] = history.get("valid_to", "")
                item["profile_version_id"] = history.get("version_id", "")
                item.pop("history", None)
                return item
        return None



