# src/keyword_fact_graph.py
# Keyword-based fact graph retrieval for multi-hop QA expansion.

from collections import defaultdict
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import config
from src.retrieval import LayeredRetriever


ROLE_PEOPLE = {"user", "assistant", "system"}


def normalize_keyword(value: Any) -> str:
    """Normalize a keyword for exact graph matching."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def normalize_people(values: Iterable[Any]) -> List[str]:
    """Normalize people and remove dialogue roles that are not real entities."""
    normalized: List[str] = []
    seen = set()
    for value in values or []:
        if not isinstance(value, str):
            continue
        item = " ".join(value.strip().lower().split())
        if not item or item in ROLE_PEOPLE:
            continue
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


class KeywordFactGraphIndex:
    """Lightweight fact-only index built from existing fact keywords/people."""

    def __init__(
        self,
        conversation_id: str = "",
        num_facts: int = 0,
        keyword_df: Optional[Dict[str, int]] = None,
        fact_keywords: Optional[Dict[str, List[str]]] = None,
        fact_people: Optional[Dict[str, List[str]]] = None,
        keyword_to_facts: Optional[Dict[str, List[str]]] = None,
    ):
        self.conversation_id = conversation_id
        self.num_facts = num_facts
        self.keyword_df = keyword_df or {}
        self.fact_keywords = fact_keywords or {}
        self.fact_people = fact_people or {}
        self.keyword_to_facts = keyword_to_facts or {}

    @classmethod
    def build(cls, facts: List[Dict[str, Any]], conversation_id: str = "") -> "KeywordFactGraphIndex":
        """Build an index from fact metadata."""
        fact_keywords: Dict[str, List[str]] = {}
        fact_people: Dict[str, List[str]] = {}
        keyword_to_facts: Dict[str, List[str]] = defaultdict(list)

        for fact in facts:
            fact_id = fact.get("fact_id")
            if not fact_id:
                continue

            keywords = []
            seen_keywords = set()
            for keyword in fact.get("keywords", []) or []:
                item = normalize_keyword(keyword)
                if not item or item in seen_keywords:
                    continue
                seen_keywords.add(item)
                keywords.append(item)

            people = normalize_people(fact.get("people", []))
            fact_keywords[fact_id] = keywords
            fact_people[fact_id] = people

            for keyword in keywords:
                keyword_to_facts[keyword].append(fact_id)

        keyword_df = {
            keyword: len(set(fact_ids))
            for keyword, fact_ids in keyword_to_facts.items()
        }

        return cls(
            conversation_id=conversation_id,
            num_facts=len(fact_keywords),
            keyword_df=keyword_df,
            fact_keywords=fact_keywords,
            fact_people=fact_people,
            keyword_to_facts={k: list(dict.fromkeys(v)) for k, v in keyword_to_facts.items()},
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeywordFactGraphIndex":
        """Restore an index from a JSON-serializable dictionary."""
        return cls(
            conversation_id=data.get("conversation_id", ""),
            num_facts=int(data.get("num_facts", 0) or 0),
            keyword_df=data.get("keyword_df", {}) or {},
            fact_keywords=data.get("fact_keywords", {}) or {},
            fact_people=data.get("fact_people", {}) or {},
            keyword_to_facts=data.get("keyword_to_facts", {}) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the index to a JSON-serializable dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "num_facts": self.num_facts,
            "keyword_df": self.keyword_df,
            "fact_keywords": self.fact_keywords,
            "fact_people": self.fact_people,
            "keyword_to_facts": self.keyword_to_facts,
        }

    def idf(self, keyword: str) -> float:
        """Return smoothed IDF for a normalized keyword."""
        df = self.keyword_df.get(keyword, 0)
        n = max(self.num_facts, 1)
        return math.log((n + 1) / (df + 1))

    def df_ratio(self, keyword: str) -> float:
        """Return document-frequency ratio for a normalized keyword."""
        if self.num_facts <= 0:
            return 0.0
        return self.keyword_df.get(keyword, 0) / self.num_facts

    def is_empty(self) -> bool:
        return not self.fact_keywords or self.num_facts <= 0


class KeywordFactGraphRetriever:
    """Query-time PPR retriever over the keyword fact graph index."""

    def __init__(self, graph_index: KeywordFactGraphIndex):
        self.index = graph_index
        self.last_debug: Dict[str, Any] = {}

    def retrieve(
        self,
        query_info: Dict[str, Any],
        seed_facts: List[Dict[str, Any]],
        all_facts: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Return graph-expanded facts plus debug information."""
        top_k = top_k or config.KEYWORD_GRAPH_RECALL_TOP_K
        debug: Dict[str, Any] = {
            "enabled": True,
            "query_keywords": [],
            "query_people": [],
            "seed_fact_ids": [],
            "local_nodes": 0,
            "local_edges": 0,
            "candidate_edges": 0,
            "rejected_by_person_gate": 0,
            "graph_recalled_facts": [],
        }

        if self.index.is_empty() or not all_facts:
            debug["enabled"] = False
            debug["reason"] = "empty_graph_index_or_facts"
            self.last_debug = debug
            return [], debug

        fact_lookup = {
            fact.get("fact_id"): fact
            for fact in all_facts
            if fact.get("fact_id")
        }
        query_keywords = [
            keyword
            for keyword in (normalize_keyword(k) for k in query_info.get("keywords", []) or [])
            if keyword
        ]
        query_keywords = list(dict.fromkeys(query_keywords))
        query_keyword_set = set(query_keywords)
        query_people = set(normalize_people(query_info.get("people", [])))
        debug["query_keywords"] = query_keywords
        debug["query_people"] = sorted(query_people)

        seed_scores = self._build_seed_scores(
            query_info=query_info,
            seed_facts=seed_facts,
            all_facts=all_facts,
            query_keywords=query_keywords,
            query_people=query_people,
        )
        if not seed_scores:
            debug["enabled"] = False
            debug["reason"] = "no_seed_scores"
            self.last_debug = debug
            return [], debug

        seed_fact_ids = set(fact.get("fact_id") for fact in seed_facts if fact.get("fact_id"))
        debug["seed_fact_ids"] = [
            fact_id for fact_id, _score in sorted(seed_scores.items(), key=lambda item: item[1], reverse=True)
        ][: config.KEYWORD_GRAPH_MAX_SEED_FACTS]

        local_ids = self._build_local_node_set(
            seed_scores=seed_scores,
            query_info=query_info,
            query_keywords=query_keywords,
            query_people=query_people,
            fact_lookup=fact_lookup,
        )
        if not local_ids:
            debug["enabled"] = False
            debug["reason"] = "empty_local_graph"
            self.last_debug = debug
            return [], debug

        adjacency, edge_debug = self._build_adjacency(local_ids, query_keyword_set)
        debug.update(edge_debug)
        debug["local_nodes"] = len(local_ids)

        ppr_scores = self._run_ppr(local_ids, adjacency, seed_scores)
        if not ppr_scores:
            debug["enabled"] = False
            debug["reason"] = "empty_ppr_scores"
            self.last_debug = debug
            return [], debug

        ranked = sorted(ppr_scores.items(), key=lambda item: item[1], reverse=True)
        graph_facts: List[Dict[str, Any]] = []
        for fact_id, score in ranked:
            if fact_id in seed_fact_ids:
                continue
            fact = fact_lookup.get(fact_id)
            if not fact:
                continue
            graph_fact = dict(fact)
            graph_fact["source"] = "keyword_graph"
            graph_fact["score"] = score
            graph_fact["graph_score"] = score
            graph_fact["graph_shared_keywords_from_seed"] = self._best_shared_keywords_from_seed(
                fact_id,
                debug["seed_fact_ids"],
                query_keyword_set,
            )
            graph_facts.append(graph_fact)
            if len(graph_facts) >= top_k:
                break

        debug["graph_recalled_facts"] = [
            {
                "fact_id": fact.get("fact_id"),
                "dia_id": fact.get("dia_id"),
                "graph_score": round(float(fact.get("graph_score", 0.0)), 6),
                "shared_keywords_from_seed": fact.get("graph_shared_keywords_from_seed", []),
                "fact": fact.get("fact", ""),
            }
            for fact in graph_facts
        ]
        self.last_debug = debug
        return graph_facts, debug

    def _build_seed_scores(
        self,
        query_info: Dict[str, Any],
        seed_facts: List[Dict[str, Any]],
        all_facts: List[Dict[str, Any]],
        query_keywords: List[str],
        query_people: Set[str],
    ) -> Dict[str, float]:
        """Build query-personalized PPR seed distribution."""
        existing_scores: Dict[str, float] = {}
        for fact in seed_facts[: config.KEYWORD_GRAPH_MAX_SEED_FACTS]:
            fact_id = fact.get("fact_id")
            if not fact_id:
                continue
            score = float(fact.get("score", 0.0) or 0.0)
            existing_scores[fact_id] = max(existing_scores.get(fact_id, 0.0), score)

        query_match_scores: Dict[str, float] = {}
        for fact in all_facts:
            fact_id = fact.get("fact_id")
            if not fact_id or not self._fact_matches_query_filters(fact, query_info, query_people):
                continue
            score = self._query_keyword_match_score(fact_id, set(query_keywords))
            if score > 0:
                query_match_scores[fact_id] = score

        query_match_scores = dict(
            sorted(query_match_scores.items(), key=lambda item: item[1], reverse=True)[
                : config.KEYWORD_GRAPH_QUERY_KEYWORD_HIT_TOP_K
            ]
        )

        existing_max = max(existing_scores.values(), default=0.0) or 1.0
        query_max = max(query_match_scores.values(), default=0.0) or 1.0
        fact_ids = set(existing_scores) | set(query_match_scores)

        seed_scores = {}
        for fact_id in fact_ids:
            existing_norm = existing_scores.get(fact_id, 0.0) / existing_max
            query_norm = query_match_scores.get(fact_id, 0.0) / query_max
            score = 0.6 * existing_norm + 0.4 * query_norm
            if score > 0:
                seed_scores[fact_id] = score

        total = sum(seed_scores.values())
        if total <= 0:
            return {}
        return {fact_id: score / total for fact_id, score in seed_scores.items()}

    def _build_local_node_set(
        self,
        seed_scores: Dict[str, float],
        query_info: Dict[str, Any],
        query_keywords: List[str],
        query_people: Set[str],
        fact_lookup: Dict[str, Dict[str, Any]],
    ) -> Set[str]:
        """Construct a bounded query-time local graph."""
        max_nodes = config.KEYWORD_GRAPH_MAX_LOCAL_NODES
        local_ids: Set[str] = set()

        ranked_seeds = [
            fact_id for fact_id, _score in sorted(seed_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        for fact_id in ranked_seeds:
            if fact_id in fact_lookup:
                local_ids.add(fact_id)
            if len(local_ids) >= max_nodes:
                return local_ids

        frontier = set(local_ids)
        query_keyword_set = set(query_keywords)
        for _hop in range(config.KEYWORD_GRAPH_MAX_HOPS):
            next_frontier: Set[str] = set()
            for fact_id in list(frontier):
                neighbors = self._top_neighbors(
                    fact_id=fact_id,
                    query_info=query_info,
                    query_keywords=query_keyword_set,
                    query_people=query_people,
                    fact_lookup=fact_lookup,
                )
                for neighbor_id, _weight in neighbors:
                    if neighbor_id in local_ids:
                        continue
                    local_ids.add(neighbor_id)
                    next_frontier.add(neighbor_id)
                    if len(local_ids) >= max_nodes:
                        return local_ids
            if not next_frontier:
                break
            frontier = next_frontier

        return local_ids

    def _top_neighbors(
        self,
        fact_id: str,
        query_info: Dict[str, Any],
        query_keywords: Set[str],
        query_people: Set[str],
        fact_lookup: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        """Return top weighted keyword neighbors for one fact."""
        neighbor_ids: Set[str] = set()
        for keyword in self.index.fact_keywords.get(fact_id, []) or []:
            if self._should_skip_keyword(keyword, query_keywords):
                continue
            for neighbor_id in self.index.keyword_to_facts.get(keyword, []) or []:
                if neighbor_id != fact_id:
                    neighbor_ids.add(neighbor_id)

        scored = []
        for neighbor_id in neighbor_ids:
            fact = fact_lookup.get(neighbor_id)
            if not fact or not self._fact_matches_query_filters(fact, query_info, query_people):
                continue
            if not self._passes_person_gate(fact_id, neighbor_id):
                continue
            weight = self._edge_weight(fact_id, neighbor_id, query_keywords)
            if weight > 0:
                scored.append((neighbor_id, weight))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: config.KEYWORD_GRAPH_MAX_NEIGHBORS_PER_FACT]

    def _build_adjacency(
        self,
        local_ids: Set[str],
        query_keywords: Set[str],
    ) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, Any]]:
        """Build weighted adjacency for the local graph."""
        ids = list(local_ids)
        adjacency: Dict[str, List[Tuple[str, float]]] = {fact_id: [] for fact_id in ids}
        candidate_edges = 0
        rejected_by_person_gate = 0

        for i, left_id in enumerate(ids):
            for right_id in ids[i + 1:]:
                left_keywords = set(self.index.fact_keywords.get(left_id, []))
                right_keywords = set(self.index.fact_keywords.get(right_id, []))
                if not (left_keywords & right_keywords):
                    continue
                candidate_edges += 1
                if not self._passes_person_gate(left_id, right_id):
                    rejected_by_person_gate += 1
                    continue
                weight = self._edge_weight(left_id, right_id, query_keywords)
                if weight <= 0:
                    continue
                adjacency[left_id].append((right_id, weight))
                adjacency[right_id].append((left_id, weight))

        for fact_id, neighbors in adjacency.items():
            neighbors.sort(key=lambda item: item[1], reverse=True)
            adjacency[fact_id] = neighbors[: config.KEYWORD_GRAPH_MAX_NEIGHBORS_PER_FACT]

        local_edges = sum(len(neighbors) for neighbors in adjacency.values())
        return adjacency, {
            "candidate_edges": candidate_edges,
            "rejected_by_person_gate": rejected_by_person_gate,
            "local_edges": local_edges,
        }

    def _run_ppr(
        self,
        local_ids: Set[str],
        adjacency: Dict[str, List[Tuple[str, float]]],
        seed_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Run random walk with restart on the local graph."""
        local_seed = {
            fact_id: score
            for fact_id, score in seed_scores.items()
            if fact_id in local_ids and score > 0
        }
        seed_total = sum(local_seed.values())
        if seed_total <= 0:
            return {}
        seed = {fact_id: score / seed_total for fact_id, score in local_seed.items()}
        scores = {fact_id: seed.get(fact_id, 0.0) for fact_id in local_ids}

        restart_prob = config.KEYWORD_GRAPH_RESTART_PROB
        for _iteration in range(config.KEYWORD_GRAPH_MAX_ITER):
            next_scores = {fact_id: restart_prob * seed.get(fact_id, 0.0) for fact_id in local_ids}
            sink_mass = 0.0

            for source_id in local_ids:
                source_score = scores.get(source_id, 0.0)
                neighbors = adjacency.get(source_id, [])
                edge_total = sum(weight for _target_id, weight in neighbors)
                if edge_total <= 0:
                    sink_mass += source_score
                    continue
                for target_id, weight in neighbors:
                    next_scores[target_id] = next_scores.get(target_id, 0.0) + (
                        (1 - restart_prob) * source_score * weight / edge_total
                    )

            if sink_mass:
                for fact_id, seed_score in seed.items():
                    next_scores[fact_id] = next_scores.get(fact_id, 0.0) + (
                        (1 - restart_prob) * sink_mass * seed_score
                    )

            delta = sum(abs(next_scores.get(fact_id, 0.0) - scores.get(fact_id, 0.0)) for fact_id in local_ids)
            scores = next_scores
            if delta < config.KEYWORD_GRAPH_TOL:
                break

        return scores

    def _edge_weight(self, left_id: str, right_id: str, query_keywords: Set[str]) -> float:
        """Weighted cosine-style shared keyword edge weight."""
        left_keywords = set(self.index.fact_keywords.get(left_id, []))
        right_keywords = set(self.index.fact_keywords.get(right_id, []))
        shared_keywords = left_keywords & right_keywords
        if not shared_keywords:
            return 0.0

        shared_weight = sum(self._keyword_weight(keyword, query_keywords) for keyword in shared_keywords)
        left_total = sum(self._keyword_weight(keyword, query_keywords) for keyword in left_keywords)
        right_total = sum(self._keyword_weight(keyword, query_keywords) for keyword in right_keywords)
        denominator = math.sqrt(left_total * right_total + config.KEYWORD_GRAPH_EDGE_EPSILON)
        if denominator <= 0:
            return 0.0
        return shared_weight / denominator

    def _keyword_weight(self, keyword: str, query_keywords: Set[str]) -> float:
        """Query-time keyword weight."""
        if not keyword:
            return 0.0
        idf = self.index.idf(keyword)
        if idf <= 0:
            return 0.0

        is_query_keyword = keyword in query_keywords
        query_boost = config.KEYWORD_GRAPH_QUERY_KEYWORD_BOOST if is_query_keyword else 1.0
        penalty = self._frequency_penalty(keyword, is_query_keyword)
        return idf * query_boost * penalty

    def _frequency_penalty(self, keyword: str, is_query_keyword: bool) -> float:
        """Continuous soft frequency penalty with a floor for query keywords."""
        df_ratio = self.index.df_ratio(keyword)
        if is_query_keyword:
            tau = config.KEYWORD_GRAPH_QUERY_PENALTY_TAU
            gamma = config.KEYWORD_GRAPH_QUERY_PENALTY_GAMMA
            base = (tau / max(df_ratio, tau)) ** gamma
            return max(config.KEYWORD_GRAPH_QUERY_PENALTY_FLOOR, base)

        tau = config.KEYWORD_GRAPH_NON_QUERY_PENALTY_TAU
        gamma = config.KEYWORD_GRAPH_NON_QUERY_PENALTY_GAMMA
        return (tau / max(df_ratio, tau)) ** gamma

    def _query_keyword_match_score(self, fact_id: str, query_keywords: Set[str]) -> float:
        """Weighted overlap between a fact and query keywords."""
        fact_keywords = set(self.index.fact_keywords.get(fact_id, []))
        return sum(self._keyword_weight(keyword, query_keywords) for keyword in fact_keywords & query_keywords)

    def _passes_person_gate(self, left_id: str, right_id: str) -> bool:
        """Reject edges only when both facts have non-empty disjoint people sets."""
        left_people = set(self.index.fact_people.get(left_id, []) or [])
        right_people = set(self.index.fact_people.get(right_id, []) or [])
        if left_people and right_people and not (left_people & right_people):
            return False
        return True

    def _fact_matches_query_filters(
        self,
        fact: Dict[str, Any],
        query_info: Dict[str, Any],
        query_people: Set[str],
    ) -> bool:
        """Apply light query filters for graph expansion."""
        fact_id = fact.get("fact_id")
        if not fact_id:
            return False

        fact_people = set(self.index.fact_people.get(fact_id, []) or normalize_people(fact.get("people", [])))
        if query_people and fact_people and not (query_people & fact_people):
            return False

        query_time = query_info.get("time")
        if query_time:
            fact_times = fact.get("time", ["", ""])
            matched = False
            for fact_time in fact_times:
                if not fact_time:
                    continue
                for query_time_item in query_time:
                    if fact_time.startswith(query_time_item):
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                return False

        return True

    def _should_skip_keyword(self, keyword: str, query_keywords: Set[str]) -> bool:
        """Optionally skip extremely high-frequency non-query keywords."""
        skip_ratio = config.KEYWORD_GRAPH_NON_QUERY_SKIP_DF_RATIO
        if skip_ratio is None:
            return False
        if keyword in query_keywords:
            return False
        return self.index.df_ratio(keyword) > skip_ratio

    def _best_shared_keywords_from_seed(
        self,
        fact_id: str,
        seed_fact_ids: List[str],
        query_keywords: Set[str],
    ) -> List[str]:
        """Return shared keywords from the strongest seed connection for debugging."""
        best_keywords: List[str] = []
        best_weight = 0.0
        fact_keywords = set(self.index.fact_keywords.get(fact_id, []))
        for seed_id in seed_fact_ids:
            if seed_id == fact_id:
                continue
            shared = fact_keywords & set(self.index.fact_keywords.get(seed_id, []))
            if not shared:
                continue
            weight = self._edge_weight(fact_id, seed_id, query_keywords)
            if weight > best_weight:
                best_weight = weight
                best_keywords = sorted(shared)
        return best_keywords


class KeywordGraphLayeredRetriever(LayeredRetriever):
    """LayeredRetriever variant that appends keyword graph recall candidates."""

    def __init__(self, graph_index: Optional[KeywordFactGraphIndex] = None, enable_graph: bool = True):
        super().__init__()
        self.graph_index = graph_index
        self.enable_graph = enable_graph
        self.last_graph_debug: Dict[str, Any] = {}

    def retrieve_for_query(
        self,
        query_info: Dict[str, Any],
        all_facts: List[Dict[str, Any]],
        all_events: List[Dict[str, Any]],
        all_profiles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        result = super().retrieve_for_query(query_info, all_facts, all_events, all_profiles)
        result["debug"] = {"keyword_graph": self.last_graph_debug}
        return result

    def _retrieve_facts_with_events(
        self,
        query_info: Dict[str, Any],
        all_facts: List[Dict[str, Any]],
        all_events: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        main_facts = self._main_recall_facts(query_info, all_facts)
        main_fact_ids = {fact["fact_id"] for fact in main_facts if fact.get("fact_id")}
        compensation_facts = self._compensation_recall_facts(query_info, all_facts, all_events, main_fact_ids)

        final_facts = self._merge_facts(main_facts, compensation_facts)
        if self.enable_graph and self.graph_index and not self.graph_index.is_empty():
            graph_retriever = KeywordFactGraphRetriever(self.graph_index)
            graph_facts, graph_debug = graph_retriever.retrieve(query_info, final_facts, all_facts)
            self.last_graph_debug = graph_debug
            final_facts = self._merge_facts(final_facts, graph_facts)
        else:
            self.last_graph_debug = {"enabled": False, "reason": "graph_disabled_or_empty"}

        return final_facts, self._build_event_contexts(final_facts, all_events)

    def _merge_facts(self, *fact_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge facts by fact_id while preserving source order."""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for facts in fact_lists:
            for fact in facts:
                fact_id = fact.get("fact_id")
                if not fact_id or fact_id in seen:
                    continue
                seen.add(fact_id)
                merged.append(fact)
        return merged

    def _build_event_contexts(
        self,
        facts: List[Dict[str, Any]],
        all_events: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Build fact_id -> joined event summary mapping for compatibility."""
        events_by_id = {event.get("event_id"): event for event in all_events if event.get("event_id")}
        event_contexts: Dict[str, str] = {}
        for fact in facts:
            summaries = []
            seen_summaries = set()
            for event_id in fact.get("event_ids", []) or []:
                event = events_by_id.get(event_id)
                if not event:
                    continue
                summary = event.get("summary", "")
                if summary and summary not in seen_summaries:
                    summaries.append(summary)
                    seen_summaries.add(summary)
            if summaries:
                event_contexts[fact["fact_id"]] = " | ".join(summaries)
        return event_contexts
