"""Seed-only entity/event/turn graph reranking for AtomMem QA retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import config
from atommem_core.multichannel_graph import (
    GraphConfig,
    MultiChannelFactGraphIndex,
    MultiChannelFactGraphRetriever,
    normalize_keyword,
    normalize_people,
    safe_list,
)


@dataclass
class RerankGraphConfig:
    """Tuned graph reranking hyperparameters used by the public AtomMem pipeline."""

    name: str = "entity_event_turn_graph_rerank"
    rho_kw: float = config.GRAPH_RHO_KEYWORD
    rho_event: float = config.GRAPH_RHO_EVENT
    rho_turn: float = config.GRAPH_RHO_TURN
    restart_prob: float = config.GRAPH_RESTART_PROB
    event_size_alpha: float = config.GRAPH_EVENT_SIZE_ALPHA
    max_event_size: Optional[int] = config.GRAPH_MAX_EVENT_SIZE
    max_event_neighbors_per_fact: int = config.GRAPH_MAX_EVENT_NEIGHBORS_PER_FACT
    turn_window: int = config.GRAPH_TURN_WINDOW
    tau_turn: float = config.GRAPH_TAU_TURN
    max_turn_neighbors_per_fact: int = config.GRAPH_MAX_TURN_NEIGHBORS_PER_FACT
    max_seed_facts: int = config.GRAPH_MAX_SEED_FACTS
    seed_score_power: float = config.GRAPH_SEED_SCORE_POWER
    max_neighbors_per_fact: int = config.GRAPH_MAX_NEIGHBORS_PER_FACT
    max_local_nodes: int = config.GRAPH_MAX_LOCAL_NODES
    max_hops: int = config.GRAPH_MAX_HOPS
    final_top_k: int = config.GRAPH_FINAL_TOP_K

    def to_graph_config(self) -> GraphConfig:
        return GraphConfig(
            name=self.name,
            use_keyword=self.rho_kw > 0,
            use_event=self.rho_event > 0,
            use_turn=self.rho_turn > 0,
            rho_kw=self.rho_kw,
            rho_event=self.rho_event,
            rho_turn=self.rho_turn,
            restart_prob=self.restart_prob,
            event_size_alpha=self.event_size_alpha,
            max_event_size=self.max_event_size,
            max_event_neighbors_per_fact=self.max_event_neighbors_per_fact,
            turn_window=self.turn_window,
            tau_turn=self.tau_turn,
            max_turn_neighbors_per_fact=self.max_turn_neighbors_per_fact,
            max_seed_facts=self.max_seed_facts,
            max_neighbors_per_fact=self.max_neighbors_per_fact,
            max_local_nodes=self.max_local_nodes,
            max_hops=self.max_hops,
            graph_recall_top_k=self.final_top_k,
            turn_expand_hops=1,
            event_expand_hops=2,
            query_keyword_hit_top_k=0,
            max_iter=config.GRAPH_MAX_ITER,
            tol=config.GRAPH_TOL,
            query_keyword_boost=config.GRAPH_QUERY_KEYWORD_BOOST,
            query_penalty_floor=config.GRAPH_QUERY_PENALTY_FLOOR,
            query_penalty_tau=config.GRAPH_QUERY_PENALTY_TAU,
            query_penalty_gamma=config.GRAPH_QUERY_PENALTY_GAMMA,
            non_query_penalty_tau=config.GRAPH_NON_QUERY_PENALTY_TAU,
            non_query_penalty_gamma=config.GRAPH_NON_QUERY_PENALTY_GAMMA,
        )


class SeedOnlyGraphReranker(MultiChannelFactGraphRetriever):
    """Run RWR/PPR from the no-graph seed set and rank all local nodes."""

    def __init__(self, index: MultiChannelFactGraphIndex, rerank_config: RerankGraphConfig):
        self.rerank_config = rerank_config
        super().__init__(index, rerank_config.to_graph_config())

    def _build_seed_scores(
        self,
        query_info: Dict[str, Any],
        seed_facts: List[Dict[str, Any]],
        all_facts: List[Dict[str, Any]],
        query_keywords: Set[str],
        query_people: Set[str],
    ) -> Dict[str, float]:
        _ = query_info, all_facts, query_keywords, query_people
        raw: Dict[str, float] = {}
        for rank, fact in enumerate(seed_facts[: self.config.max_seed_facts]):
            fact_id = fact.get("fact_id")
            if not fact_id:
                continue
            retrieval_score = float(fact.get("score", 0.0) or 0.0)
            rank_prior = 1.0 / (rank + 1)
            raw[fact_id] = max(raw.get(fact_id, 0.0), retrieval_score + 0.05 * rank_prior)

        if not raw:
            return {}

        max_score = max(raw.values()) or 1.0
        powered = {}
        for fact_id, score in raw.items():
            normalized = max(score / max_score, 1e-9)
            if self.rerank_config.seed_score_power == 0:
                powered[fact_id] = 1.0
            else:
                powered[fact_id] = normalized ** self.rerank_config.seed_score_power

        total = sum(powered.values())
        return {fact_id: score / total for fact_id, score in powered.items()} if total > 0 else {}

    def retrieve_ranked_topk(
        self,
        query_info: Dict[str, Any],
        seed_facts: List[Dict[str, Any]],
        all_facts: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        top_k = top_k or self.rerank_config.final_top_k
        debug = {
            "config": asdict(self.rerank_config),
            "seed_fact_ids": [],
            "local_nodes": 0,
            "local_edges_by_channel": {},
            "graph_ranked_facts": [],
        }

        if self.index.num_facts <= 0:
            debug["enabled"] = False
            debug["reason"] = "empty_index"
            return [], debug

        self._keyword_weight_cache = {}
        query_keywords = [
            normalize_keyword(keyword)
            for keyword in safe_list(query_info.get("keywords"))
            if normalize_keyword(keyword)
        ]
        query_keyword_set = set(dict.fromkeys(query_keywords))
        query_people = set(normalize_people(query_info.get("people", [])))

        seed_scores = self._build_seed_scores(query_info, seed_facts, all_facts, query_keyword_set, query_people)
        if not seed_scores:
            debug["enabled"] = False
            debug["reason"] = "no_seed_scores"
            return [], debug

        debug["seed_fact_ids"] = [
            fact_id for fact_id, _score in sorted(seed_scores.items(), key=lambda item: (-item[1], item[0]))
        ][: self.config.max_seed_facts]

        local_ids = self._build_local_node_set(seed_scores, query_info, query_keyword_set, query_people)
        if not local_ids:
            debug["enabled"] = False
            debug["reason"] = "empty_local_nodes"
            return [], debug

        channel_adjacency, edge_debug = self._build_channel_adjacency(local_ids, query_keyword_set)
        debug["local_nodes"] = len(local_ids)
        debug["local_edges_by_channel"] = edge_debug

        graph_scores = self._run_ppr(local_ids, channel_adjacency, seed_scores)
        ranked = sorted(graph_scores.items(), key=lambda item: (-item[1], item[0]))

        ranked_facts: List[Dict[str, Any]] = []
        for fact_id, score in ranked[:top_k]:
            fact = self.index.fact_lookup.get(fact_id)
            if not fact:
                continue
            item = dict(fact)
            item["source"] = f"graph_rerank:{self.rerank_config.name}"
            item["score"] = score
            item["graph_score"] = score
            item["is_graph_seed"] = fact_id in seed_scores
            ranked_facts.append(item)

        debug["graph_ranked_facts"] = [
            {
                "fact_id": fact.get("fact_id"),
                "dia_id": fact.get("dia_id"),
                "score": round(float(fact.get("graph_score", 0.0)), 6),
                "is_graph_seed": bool(fact.get("is_graph_seed")),
            }
            for fact in ranked_facts
        ]
        debug["enabled"] = True
        return ranked_facts, debug
