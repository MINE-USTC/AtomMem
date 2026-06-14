# Multi-channel fact graph index and retriever.

import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import config


ROLE_PEOPLE = {"user", "assistant", "system"}


@dataclass
class GraphConfig:
    name: str
    use_keyword: bool = True
    use_event: bool = False
    use_turn: bool = False
    rho_kw: float = 1.0
    rho_event: float = 0.0
    rho_turn: float = 0.0
    event_size_alpha: float = 1.0
    max_event_size: Optional[int] = 30
    max_event_neighbors_per_fact: int = 15
    turn_window: int = 1
    tau_turn: float = 2.0
    max_turn_neighbors_per_fact: int = 8
    graph_recall_top_k: int = 10
    restart_prob: float = 0.35
    max_iter: int = 20
    tol: float = 1e-6
    query_keyword_boost: float = 2.5
    query_penalty_floor: float = 0.45
    query_penalty_tau: float = 0.05
    query_penalty_gamma: float = 0.7
    non_query_penalty_tau: float = 0.10
    non_query_penalty_gamma: float = 1.0
    max_seed_facts: int = 10
    query_keyword_hit_top_k: int = 80
    max_neighbors_per_fact: int = 30
    max_local_nodes: int = 300
    max_hops: int = 2
    edge_epsilon: float = 1e-8
    turn_expand_hops: int = 1
    event_expand_hops: int = 2


def normalize_keyword(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def normalize_people(values: Iterable[Any]) -> List[str]:
    normalized = []
    seen = set()
    for value in values or []:
        if not isinstance(value, str):
            continue
        item = " ".join(value.strip().lower().split())
        if not item or item in ROLE_PEOPLE or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def parse_dia_id(dia_id: Any, fallback_order: int) -> Tuple[str, int]:
    text = str(dia_id or "")
    match = re.match(r"^([^:]+):(\d+)$", text)
    if match:
        return match.group(1), int(match.group(2))
    return "__global__", fallback_order


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []



class MultiChannelFactGraphIndex:
    def __init__(self, facts: List[Dict[str, Any]], events: List[Dict[str, Any]], conv_id: str = ""):
        self.conv_id = conv_id
        self.facts = facts
        self.fact_lookup = {
            fact.get("fact_id"): fact
            for fact in facts
            if fact.get("fact_id")
        }
        self.num_facts = len(self.fact_lookup)
        self.fact_keywords: Dict[str, List[str]] = {}
        self.fact_people: Dict[str, List[str]] = {}
        self.keyword_to_facts: Dict[str, List[str]] = defaultdict(list)
        self.keyword_df: Dict[str, int] = {}
        self.fact_events: Dict[str, List[str]] = {}
        self.event_to_facts: Dict[str, List[str]] = defaultdict(list)
        self.event_size: Dict[str, int] = {}
        self.fact_turn: Dict[str, Tuple[str, int]] = {}
        self.turn_to_facts: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        self._build(events)

    def _build(self, events: List[Dict[str, Any]]) -> None:
        for order, fact in enumerate(self.facts):
            fact_id = fact.get("fact_id")
            if not fact_id:
                continue
            keywords = []
            seen = set()
            for keyword in safe_list(fact.get("keywords")):
                item = normalize_keyword(keyword)
                if item and item not in seen:
                    seen.add(item)
                    keywords.append(item)
                    self.keyword_to_facts[item].append(fact_id)
            self.fact_keywords[fact_id] = keywords
            self.fact_people[fact_id] = normalize_people(fact.get("people", []))

            turn_key = parse_dia_id(fact.get("dia_id"), order)
            self.fact_turn[fact_id] = turn_key
            self.turn_to_facts[turn_key].append(fact_id)

            event_ids = []
            seen_events = set()
            for event_id in safe_list(fact.get("event_ids")):
                if isinstance(event_id, str) and event_id and event_id not in seen_events:
                    seen_events.add(event_id)
                    event_ids.append(event_id)
            self.fact_events[fact_id] = event_ids

        self.keyword_df = {
            keyword: len(set(fact_ids))
            for keyword, fact_ids in self.keyword_to_facts.items()
        }

        event_lookup = {event.get("event_id"): event for event in events if event.get("event_id")}
        for event_id, event in event_lookup.items():
            fact_ids = [
                fact_id for fact_id in safe_list(event.get("fact_ids"))
                if fact_id in self.fact_lookup
            ]
            if fact_ids:
                self.event_to_facts[event_id] = list(dict.fromkeys(fact_ids))
                self.event_size[event_id] = len(self.event_to_facts[event_id])

        # Prefer the event file as the authoritative event membership, but keep
        # fact.event_ids for events that may be missing in the event snapshot.
        for fact_id, event_ids in self.fact_events.items():
            for event_id in event_ids:
                if fact_id not in self.event_to_facts[event_id]:
                    self.event_to_facts[event_id].append(fact_id)
                self.event_size[event_id] = len(set(self.event_to_facts[event_id]))

    def idf(self, keyword: str) -> float:
        return math.log((self.num_facts + 1) / (self.keyword_df.get(keyword, 0) + 1))

    def df_ratio(self, keyword: str) -> float:
        return self.keyword_df.get(keyword, 0) / max(self.num_facts, 1)


class MultiChannelFactGraphRetriever:
    def __init__(self, index: MultiChannelFactGraphIndex, graph_config: GraphConfig):
        self.index = index
        self.config = graph_config

    def retrieve(
        self,
        query_info: Dict[str, Any],
        seed_facts: List[Dict[str, Any]],
        all_facts: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        cfg = self.config
        debug = {
            "config": asdict(cfg),
            "seed_fact_ids": [],
            "local_nodes": 0,
            "local_edges_by_channel": {},
            "graph_recalled_facts": [],
        }
        if self.index.num_facts <= 0:
            debug["enabled"] = False
            debug["reason"] = "empty_index"
            return [], debug
        self._keyword_weight_cache: Dict[Tuple[str, bool], float] = {}

        query_keywords = [
            normalize_keyword(k)
            for k in safe_list(query_info.get("keywords"))
            if normalize_keyword(k)
        ]
        query_keywords = list(dict.fromkeys(query_keywords))
        query_keyword_set = set(query_keywords)
        query_people = set(normalize_people(query_info.get("people", [])))

        seed_scores = self._build_seed_scores(query_info, seed_facts, all_facts, query_keyword_set, query_people)
        if not seed_scores:
            debug["enabled"] = False
            debug["reason"] = "no_seed_scores"
            return [], debug

        seed_fact_ids = set(fact.get("fact_id") for fact in seed_facts if fact.get("fact_id"))
        debug["seed_fact_ids"] = [
            fact_id for fact_id, _score in sorted(seed_scores.items(), key=lambda item: (-item[1], item[0]))
        ][: cfg.max_seed_facts]

        local_ids = self._build_local_node_set(seed_scores, query_info, query_keyword_set, query_people)
        if not local_ids:
            debug["enabled"] = False
            debug["reason"] = "empty_local_nodes"
            return [], debug

        channel_adjacency, edge_debug = self._build_channel_adjacency(local_ids, query_keyword_set)
        debug["local_nodes"] = len(local_ids)
        debug["local_edges_by_channel"] = edge_debug

        ppr_scores = self._run_ppr(local_ids, channel_adjacency, seed_scores)
        ranked = sorted(ppr_scores.items(), key=lambda item: (-item[1], item[0]))

        graph_facts = []
        for fact_id, score in ranked:
            if fact_id in seed_fact_ids:
                continue
            fact = self.index.fact_lookup.get(fact_id)
            if not fact:
                continue
            item = dict(fact)
            item["source"] = f"graph:{cfg.name}"
            item["score"] = score
            item["graph_score"] = score
            graph_facts.append(item)
            if len(graph_facts) >= cfg.graph_recall_top_k:
                break

        debug["graph_recalled_facts"] = [
            {
                "fact_id": fact.get("fact_id"),
                "dia_id": fact.get("dia_id"),
                "score": round(float(fact.get("graph_score", 0.0)), 6),
                "fact": fact.get("fact", ""),
            }
            for fact in graph_facts
        ]
        return graph_facts, debug

    def _build_seed_scores(
        self,
        query_info: Dict[str, Any],
        seed_facts: List[Dict[str, Any]],
        all_facts: List[Dict[str, Any]],
        query_keywords: Set[str],
        query_people: Set[str],
    ) -> Dict[str, float]:
        cfg = self.config
        existing_scores = {}
        for fact in seed_facts[: cfg.max_seed_facts]:
            fact_id = fact.get("fact_id")
            if not fact_id:
                continue
            existing_scores[fact_id] = max(existing_scores.get(fact_id, 0.0), float(fact.get("score", 0.0) or 0.0))

        query_match_scores = {}
        for fact in all_facts:
            fact_id = fact.get("fact_id")
            if not fact_id or not self._fact_matches_query_filters(fact, query_info, query_people):
                continue
            score = self._query_keyword_match_score(fact_id, query_keywords)
            if score > 0:
                query_match_scores[fact_id] = score

        query_match_scores = dict(
            sorted(query_match_scores.items(), key=lambda item: (-item[1], item[0]))[: cfg.query_keyword_hit_top_k]
        )
        existing_max = max(existing_scores.values(), default=0.0) or 1.0
        query_max = max(query_match_scores.values(), default=0.0) or 1.0

        raw = {}
        for fact_id in set(existing_scores) | set(query_match_scores):
            existing_norm = existing_scores.get(fact_id, 0.0) / existing_max
            query_norm = query_match_scores.get(fact_id, 0.0) / query_max
            score = 0.6 * existing_norm + 0.4 * query_norm
            if score > 0:
                raw[fact_id] = score

        total = sum(raw.values())
        return {fact_id: score / total for fact_id, score in raw.items()} if total > 0 else {}

    def _build_local_node_set(
        self,
        seed_scores: Dict[str, float],
        query_info: Dict[str, Any],
        query_keywords: Set[str],
        query_people: Set[str],
    ) -> Set[str]:
        cfg = self.config
        local_ids: Set[str] = set()
        ranked_seed_ids = [
            fact_id for fact_id, _score in sorted(seed_scores.items(), key=lambda item: (-item[1], item[0]))
        ]
        for fact_id in ranked_seed_ids:
            if fact_id in self.index.fact_lookup:
                local_ids.add(fact_id)
            if len(local_ids) >= cfg.max_local_nodes:
                return local_ids

        frontier = set(local_ids)
        for hop in range(cfg.max_hops):
            next_frontier = set()
            for fact_id in sorted(frontier):
                for neighbor_id, _weight in self._top_neighbors_for_expansion(
                    fact_id,
                    query_info,
                    query_keywords,
                    query_people,
                    hop=hop + 1,
                ):
                    if neighbor_id in local_ids:
                        continue
                    local_ids.add(neighbor_id)
                    next_frontier.add(neighbor_id)
                    if len(local_ids) >= cfg.max_local_nodes:
                        return local_ids
            if not next_frontier:
                break
            frontier = next_frontier
        return local_ids

    def _top_neighbors_for_expansion(
        self,
        fact_id: str,
        query_info: Dict[str, Any],
        query_keywords: Set[str],
        query_people: Set[str],
        hop: int,
    ) -> List[Tuple[str, float]]:
        cfg = self.config
        candidates: Dict[str, float] = defaultdict(float)

        if cfg.use_keyword:
            for neighbor_id in sorted(self._keyword_neighbor_ids(fact_id)):
                weight = self._keyword_edge_weight(fact_id, neighbor_id, query_keywords)
                if weight > 0:
                    candidates[neighbor_id] += cfg.rho_kw * weight

        if cfg.use_event and hop <= cfg.event_expand_hops:
            for neighbor_id in sorted(self._event_neighbor_ids(fact_id)):
                weight = self._event_edge_weight(fact_id, neighbor_id)
                if weight > 0:
                    candidates[neighbor_id] += cfg.rho_event * weight

        if cfg.use_turn and hop <= cfg.turn_expand_hops:
            for neighbor_id in sorted(self._turn_neighbor_ids(fact_id)):
                weight = self._turn_edge_weight(fact_id, neighbor_id)
                if weight > 0:
                    candidates[neighbor_id] += cfg.rho_turn * weight

        scored = []
        for neighbor_id, score in candidates.items():
            fact = self.index.fact_lookup.get(neighbor_id)
            if not fact or not self._fact_matches_query_filters(fact, query_info, query_people):
                continue
            scored.append((neighbor_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: cfg.max_neighbors_per_fact]

    def _build_channel_adjacency(
        self,
        local_ids: Set[str],
        query_keywords: Set[str],
    ) -> Tuple[Dict[str, Dict[str, List[Tuple[str, float]]]], Dict[str, int]]:
        cfg = self.config
        channels = []
        if cfg.use_keyword:
            channels.append("keyword")
        if cfg.use_event:
            channels.append("event")
        if cfg.use_turn:
            channels.append("turn")

        adjacency = {
            channel: {fact_id: [] for fact_id in local_ids}
            for channel in channels
        }
        local_lookup = set(local_ids)

        for left_id in sorted(local_ids):
            if cfg.use_keyword:
                for right_id in sorted(self._keyword_neighbor_ids(left_id)):
                    if right_id not in local_lookup:
                        continue
                    weight = self._keyword_edge_weight(left_id, right_id, query_keywords)
                    if weight > 0:
                        adjacency["keyword"][left_id].append((right_id, weight))

            if cfg.use_event:
                for right_id in sorted(self._event_neighbor_ids(left_id)):
                    if right_id not in local_lookup:
                        continue
                    weight = self._event_edge_weight(left_id, right_id)
                    if weight > 0:
                        adjacency["event"][left_id].append((right_id, weight))

            if cfg.use_turn:
                for right_id in sorted(self._turn_neighbor_ids(left_id)):
                    if right_id not in local_lookup:
                        continue
                    weight = self._turn_edge_weight(left_id, right_id)
                    if weight > 0:
                        adjacency["turn"][left_id].append((right_id, weight))

        edge_debug = {}
        for channel, rows in adjacency.items():
            limit = cfg.max_neighbors_per_fact
            if channel == "event":
                limit = cfg.max_event_neighbors_per_fact
            elif channel == "turn":
                limit = cfg.max_turn_neighbors_per_fact
            edge_count = 0
            for fact_id, neighbors in rows.items():
                neighbors.sort(key=lambda item: (-item[1], item[0]))
                rows[fact_id] = neighbors[:limit]
                edge_count += len(rows[fact_id])
            edge_debug[channel] = edge_count
        return adjacency, edge_debug

    def _run_ppr(
        self,
        local_ids: Set[str],
        channel_adjacency: Dict[str, Dict[str, List[Tuple[str, float]]]],
        seed_scores: Dict[str, float],
    ) -> Dict[str, float]:
        cfg = self.config
        local_seed = {
            fact_id: score
            for fact_id, score in seed_scores.items()
            if fact_id in local_ids and score > 0
        }
        total = sum(local_seed.values())
        if total <= 0:
            return {}
        seed = {fact_id: score / total for fact_id, score in local_seed.items()}
        scores = {fact_id: seed.get(fact_id, 0.0) for fact_id in local_ids}
        channel_rhos = {
            "keyword": cfg.rho_kw if cfg.use_keyword else 0.0,
            "event": cfg.rho_event if cfg.use_event else 0.0,
            "turn": cfg.rho_turn if cfg.use_turn else 0.0,
        }

        for _iteration in range(cfg.max_iter):
            next_scores = {fact_id: cfg.restart_prob * seed.get(fact_id, 0.0) for fact_id in local_ids}
            sink_mass = 0.0
            for source_id in sorted(local_ids):
                source_score = scores.get(source_id, 0.0)
                active_channels = []
                for channel, rho in channel_rhos.items():
                    if rho <= 0:
                        continue
                    neighbors = channel_adjacency.get(channel, {}).get(source_id, [])
                    if neighbors:
                        active_channels.append((channel, rho, neighbors))
                active_rho_sum = sum(rho for _channel, rho, _neighbors in active_channels)
                if active_rho_sum <= 0:
                    sink_mass += source_score
                    continue

                for channel, rho, neighbors in active_channels:
                    adjusted_rho = rho / active_rho_sum
                    edge_total = sum(weight for _target, weight in neighbors)
                    if edge_total <= 0:
                        continue
                    for target_id, weight in neighbors:
                        next_scores[target_id] = next_scores.get(target_id, 0.0) + (
                            (1 - cfg.restart_prob) * source_score * adjusted_rho * weight / edge_total
                        )

            if sink_mass:
                for fact_id, seed_score in sorted(seed.items()):
                    next_scores[fact_id] = next_scores.get(fact_id, 0.0) + (
                        (1 - cfg.restart_prob) * sink_mass * seed_score
                    )

            delta = sum(abs(next_scores.get(fid, 0.0) - scores.get(fid, 0.0)) for fid in local_ids)
            scores = next_scores
            if delta < cfg.tol:
                break
        return scores

    def _keyword_neighbor_ids(self, fact_id: str) -> Set[str]:
        neighbor_ids = set()
        for keyword in self.index.fact_keywords.get(fact_id, []):
            for neighbor_id in self.index.keyword_to_facts.get(keyword, []):
                if neighbor_id != fact_id and self._passes_person_gate(fact_id, neighbor_id):
                    neighbor_ids.add(neighbor_id)
        return neighbor_ids

    def _event_neighbor_ids(self, fact_id: str) -> Set[str]:
        neighbor_ids = set()
        for event_id in self.index.fact_events.get(fact_id, []):
            size = self.index.event_size.get(event_id, 0)
            if self.config.max_event_size is not None and size > self.config.max_event_size:
                continue
            for neighbor_id in self.index.event_to_facts.get(event_id, []):
                if neighbor_id != fact_id:
                    neighbor_ids.add(neighbor_id)
        return neighbor_ids

    def _turn_neighbor_ids(self, fact_id: str) -> Set[str]:
        prefix, turn_idx = self.index.fact_turn.get(fact_id, ("", -1))
        if not prefix:
            return set()
        neighbor_ids = set()
        for offset in range(-self.config.turn_window, self.config.turn_window + 1):
            for neighbor_id in self.index.turn_to_facts.get((prefix, turn_idx + offset), []):
                if neighbor_id != fact_id:
                    neighbor_ids.add(neighbor_id)
        return neighbor_ids

    def _keyword_edge_weight(self, left_id: str, right_id: str, query_keywords: Set[str]) -> float:
        if not self._passes_person_gate(left_id, right_id):
            return 0.0
        left_keywords = set(self.index.fact_keywords.get(left_id, []))
        right_keywords = set(self.index.fact_keywords.get(right_id, []))
        shared = left_keywords & right_keywords
        if not shared:
            return 0.0
        shared_weight = sum(self._keyword_weight(keyword, query_keywords) for keyword in shared)
        left_total = sum(self._keyword_weight(keyword, query_keywords) for keyword in left_keywords)
        right_total = sum(self._keyword_weight(keyword, query_keywords) for keyword in right_keywords)
        denominator = math.sqrt(left_total * right_total + self.config.edge_epsilon)
        return shared_weight / denominator if denominator > 0 else 0.0

    def _event_edge_weight(self, left_id: str, right_id: str) -> float:
        shared_events = set(self.index.fact_events.get(left_id, [])) & set(self.index.fact_events.get(right_id, []))
        if not shared_events:
            return 0.0
        total = 0.0
        for event_id in shared_events:
            size = self.index.event_size.get(event_id, 0)
            if size <= 1:
                continue
            if self.config.max_event_size is not None and size > self.config.max_event_size:
                continue
            total += 1.0 / (max(size - 1, 1) ** self.config.event_size_alpha)
        return total

    def _turn_edge_weight(self, left_id: str, right_id: str) -> float:
        left_turn = self.index.fact_turn.get(left_id)
        right_turn = self.index.fact_turn.get(right_id)
        if not left_turn or not right_turn or left_turn[0] != right_turn[0]:
            return 0.0
        distance = abs(left_turn[1] - right_turn[1])
        if distance > self.config.turn_window:
            return 0.0
        if not self._turn_soft_person_gate(left_id, right_id):
            return 0.0
        return math.exp(-distance / self.config.tau_turn)

    def _turn_soft_person_gate(self, left_id: str, right_id: str) -> bool:
        if self._passes_person_gate(left_id, right_id):
            return True
        shared_keywords = set(self.index.fact_keywords.get(left_id, [])) & set(self.index.fact_keywords.get(right_id, []))
        if shared_keywords:
            return True
        shared_events = set(self.index.fact_events.get(left_id, [])) & set(self.index.fact_events.get(right_id, []))
        return bool(shared_events)

    def _passes_person_gate(self, left_id: str, right_id: str) -> bool:
        left_people = set(self.index.fact_people.get(left_id, []))
        right_people = set(self.index.fact_people.get(right_id, []))
        if left_people and right_people and not (left_people & right_people):
            return False
        return True

    def _keyword_weight(self, keyword: str, query_keywords: Set[str]) -> float:
        cache_key = (keyword, keyword in query_keywords)
        cache = getattr(self, "_keyword_weight_cache", None)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        idf = self.index.idf(keyword)
        if idf <= 0:
            return 0.0
        is_query = keyword in query_keywords
        boost = self.config.query_keyword_boost if is_query else 1.0
        penalty = self._frequency_penalty(keyword, is_query)
        weight = idf * boost * penalty
        if cache is not None:
            cache[cache_key] = weight
        return weight

    def _frequency_penalty(self, keyword: str, is_query_keyword: bool) -> float:
        df_ratio = self.index.df_ratio(keyword)
        if is_query_keyword:
            base = (
                self.config.query_penalty_tau /
                max(df_ratio, self.config.query_penalty_tau)
            ) ** self.config.query_penalty_gamma
            return max(self.config.query_penalty_floor, base)
        return (
            self.config.non_query_penalty_tau /
            max(df_ratio, self.config.non_query_penalty_tau)
        ) ** self.config.non_query_penalty_gamma

    def _query_keyword_match_score(self, fact_id: str, query_keywords: Set[str]) -> float:
        fact_keywords = set(self.index.fact_keywords.get(fact_id, []))
        return sum(self._keyword_weight(keyword, query_keywords) for keyword in fact_keywords & query_keywords)

    def _fact_matches_query_filters(
        self,
        fact: Dict[str, Any],
        query_info: Dict[str, Any],
        query_people: Set[str],
    ) -> bool:
        fact_id = fact.get("fact_id")
        if not fact_id:
            return False
        fact_people = set(self.index.fact_people.get(fact_id, []) or normalize_people(fact.get("people", [])))
        if query_people and fact_people and not (query_people & fact_people):
            return False
        query_time = query_info.get("time")
        if query_time:
            matched = False
            for fact_time in safe_list(fact.get("time")):
                if not fact_time:
                    continue
                for qt in query_time:
                    if str(fact_time).startswith(qt):
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                return False
        return True



