# src/retrieval.py
# Retrieval Module (v2.0 - Layered Retrieval)

from typing import List, Dict, Any, Tuple
from src.utils import cosine_similarity, jaccard_similarity
from src.embedding import EmbeddingModel
import config


class LayeredRetriever:
    """Layered retrieval for Facts, Events, and Profiles."""
    
    def __init__(self):
        """Initialize retriever."""
        self.embedding_model = EmbeddingModel()
        self.alpha = config.EMBEDDING_WEIGHT_ALPHA
        self.beta = config.KEYWORD_WEIGHT_BETA
    
    def retrieve_for_query(self,
                          query_info: Dict[str, Any],
                          all_facts: List[Dict[str, Any]],
                          all_events: List[Dict[str, Any]],
                          all_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main retrieval function based on query intent.
        
        Args:
            query_info: Query information with need_specific (always True), need_attribute, people, keywords, time, embedding
            all_facts: All facts in the system
            all_events: All events in the system
            all_profiles: All profiles in the system
            
        Returns:
            {
                "facts": List[Dict],         # Retrieved facts (always retrieved)
                "profiles": List[Dict],      # Retrieved profiles (if need_attribute=True)
                "event_contexts": Dict       # fact_id -> event_summary mapping
            }
        """
        result = {
            "facts": [],
            "profiles": [],
            "event_contexts": {}
        }
        
        # Always retrieve facts (need_specific is always True)
        facts, event_contexts = self._retrieve_facts_with_events(query_info, all_facts, all_events)
        result["facts"] = facts
        result["event_contexts"] = event_contexts
        
        # Retrieve profiles if needed
        if query_info.get("need_attribute", False):
            result["profiles"] = self._retrieve_profiles(query_info, all_profiles)
        
        return result
    
    def _retrieve_profiles(self, query_info: Dict[str, Any], all_profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Retrieve profiles for attribute query.
        
        Args:
            query_info: Query information
            all_profiles: All profiles
            
        Returns:
            Top-K profiles
        """
        # Filter by person
        candidates = []
        query_people = set(query_info.get("people", []))
        
        for profile in all_profiles:
            if profile.get("person") in query_people:
                candidates.append(profile)
        
        if not candidates:
            return []
        
        # Calculate similarity (embedding + keyword)
        query_embedding = query_info.get("embedding")
        query_keywords = query_info.get("keywords", [])
        
        for profile in candidates:
            emb_sim = cosine_similarity(profile["embedding"], query_embedding)
            kw_sim = jaccard_similarity(profile.get("keywords", []), query_keywords)
            profile["score"] = self.alpha * emb_sim + self.beta * kw_sim
        
        # Sort and return Top-K
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:config.PROFILE_RECALL_TOP_K]
    
    def _retrieve_facts_with_events(self,
                                    query_info: Dict[str, Any],
                                    all_facts: List[Dict[str, Any]],
                                    all_events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Retrieve facts using main recall + compensation recall strategy.
        
        Args:
            query_info: Query information
            all_facts: All facts
            all_events: All events
            
        Returns:
            (facts: List[Dict], event_contexts: Dict[fact_id, event_summary])
        """
        # Main recall: Direct fact retrieval (Top-5)
        main_facts = self._main_recall_facts(query_info, all_facts)
        main_fact_ids = set([f["fact_id"] for f in main_facts])
        
        # Compensation recall: Event-based retrieval (Top-5)
        compensation_facts = self._compensation_recall_facts(query_info, all_facts, all_events, main_fact_ids)
        
        # Merge results
        final_facts = main_facts + compensation_facts
        
        # Build event contexts. A fact may belong to multiple events, so keep all
        # available summaries instead of silently using only the first event.
        event_contexts = {}
        for fact in final_facts:
            if fact.get("event_ids"):
                summaries = []
                seen_summaries = set()
                for event_id in fact.get("event_ids", []):
                    event = next((e for e in all_events if e["event_id"] == event_id), None)
                    if not event:
                        continue
                    summary = event.get("summary", "")
                    if summary and summary not in seen_summaries:
                        summaries.append(summary)
                        seen_summaries.add(summary)
                if summaries:
                    event_contexts[fact["fact_id"]] = " | ".join(summaries)
        
        return final_facts, event_contexts
    
    def _main_recall_facts(self, query_info: Dict[str, Any], all_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main recall: Direct fact retrieval.
        
        Returns:
            Top-5 facts
        """
        # Filter by people and time
        candidates = self._filter_facts(query_info, all_facts)
        
        if not candidates:
            return []
        
        # Calculate similarity
        query_embedding = query_info.get("embedding")
        query_keywords = query_info.get("keywords", [])
        
        for fact in candidates:
            emb_sim = cosine_similarity(fact["embedding"], query_embedding)
            kw_sim = jaccard_similarity(fact.get("keywords", []), query_keywords)
            fact["score"] = self.alpha * emb_sim + self.beta * kw_sim
            fact["source"] = "main"
        
        # Sort and return Top-K
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:config.MAIN_RECALL_TOP_K]
    
    def _compensation_recall_facts(self,
                                   query_info: Dict[str, Any],
                                   all_facts: List[Dict[str, Any]],
                                   all_events: List[Dict[str, Any]],
                                   main_fact_ids: set) -> List[Dict[str, Any]]:
        """
        Compensation recall: Event-based fact retrieval.
        
        Args:
            query_info: Query information
            all_facts: All facts
            all_events: All events
            main_fact_ids: Fact IDs from main recall (for deduplication)
            
        Returns:
            Top-5 facts from compensation recall
        """
        # Step 1: Retrieve Top-K events
        top_events = self._retrieve_events(query_info, all_events)
        
        if not top_events:
            return []
        
        # Step 2: Expand events to facts and track event scores
        query_embedding = query_info.get("embedding")
        query_keywords = query_info.get("keywords", [])
        
        event_score_map = {}  # fact_id -> [(event_id, event_score)]
        fact_pool = {}  # fact_id -> fact
        
        for event in top_events:
            for fact_id in event.get("fact_ids", []):
                # Skip facts from main recall
                if fact_id in main_fact_ids:
                    continue
                
                # Get fact object
                fact = next((f for f in all_facts if f["fact_id"] == fact_id), None)
                if not fact:
                    continue
                
                # Record event score for this fact
                if fact_id not in event_score_map:
                    event_score_map[fact_id] = []
                event_score_map[fact_id].append({
                    "event_id": event["event_id"],
                    "event_score": event["score"]
                })
                
                # Add to pool
                if fact_id not in fact_pool:
                    fact_pool[fact_id] = fact
        
        # Step 3: Calculate fusion scores
        compensation_candidates = []
        
        for fact_id, fact in fact_pool.items():
            # Calculate fact's own similarity
            emb_sim = cosine_similarity(fact["embedding"], query_embedding)
            kw_sim = jaccard_similarity(fact.get("keywords", []), query_keywords)
            fact_self_score = self.alpha * emb_sim + self.beta * kw_sim
            
            # Get max event score
            max_event_score = max([e["event_score"] for e in event_score_map[fact_id]])
            
            # Fusion score
            final_score = (max_event_score * config.COMPENSATION_EVENT_WEIGHT +
                          fact_self_score * config.COMPENSATION_FACT_WEIGHT)
            
            fact["score"] = final_score
            fact["source"] = "compensation"
            compensation_candidates.append(fact)
        
        # Step 4: Sort and return Top-K
        compensation_candidates.sort(key=lambda x: x["score"], reverse=True)
        return compensation_candidates[:config.COMPENSATION_FACT_TOP_K]
    
    def _retrieve_events(self, query_info: Dict[str, Any], all_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Retrieve events for compensation recall.
        
        Returns:
            Top-K events
        """
        # Filter by people and time
        candidates = self._filter_events(query_info, all_events)
        
        if not candidates:
            return []
        
        # Calculate similarity
        query_embedding = query_info.get("embedding")
        query_keywords = query_info.get("keywords", [])
        
        for event in candidates:
            emb_sim = cosine_similarity(event["embedding"], query_embedding)
            kw_sim = jaccard_similarity(event.get("keywords", []), query_keywords)
            event["score"] = self.alpha * emb_sim + self.beta * kw_sim
        
        # Sort and return Top-K
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:config.EVENT_RECALL_TOP_K]
    
    def _filter_facts(self, query_info: Dict[str, Any], all_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter facts by people and time.
        
        Args:
            query_info: Query information
            all_facts: All facts
            
        Returns:
            Filtered facts
        """
        candidates = []
        query_people = set(query_info.get("people", []))
        query_time = query_info.get("time")
        
        for fact in all_facts:
            # People filter: only apply when query specifies named people
            # If query_people is empty (e.g. "I", "me", or no person), skip this filter
            if query_people:
                fact_people = set(fact.get("people", []))
                if not (query_people & fact_people):
                    continue
            
            # Time filter (prefix matching)
            if query_time:
                time_match = False
                for fact_time in fact.get("time", ["", ""]):
                    if fact_time:
                        for qt in query_time:
                            if fact_time.startswith(qt):
                                time_match = True
                                break
                    if time_match:
                        break
                
                if not time_match:
                    continue
            
            candidates.append(fact)
        
        return candidates
    
    def _filter_events(self, query_info: Dict[str, Any], all_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter events by people and time.
        
        Args:
            query_info: Query information
            all_events: All events
            
        Returns:
            Filtered events
        """
        candidates = []
        query_people = set(query_info.get("people", []))
        query_time = query_info.get("time")
        
        for event in all_events:
            # People filter: only apply when query specifies named people
            # If query_people is empty (e.g. "I", "me", or no person), skip this filter
            if query_people:
                event_people = set(event.get("people", []))
                if not (query_people & event_people):
                    continue
            
            # Time filter (range matching for events)
            if query_time:
                time_match = False
                event_time_range = event.get("time", ["", ""])
                
                # Check if any query time falls within or matches the event time range
                for qt in query_time:
                    if self._time_in_event_range(qt, event_time_range):
                        time_match = True
                        break
                
                if not time_match:
                    continue
            
            candidates.append(event)
        
        return candidates
    
    def _time_in_event_range(self, query_time: str, event_range: List[str]) -> bool:
        """
        Check if query_time falls within or matches the event time range.
        
        Args:
            query_time: Query time string (e.g., "2023-05-07", "2023-05", "2023")
            event_range: Event time range [start_time, end_time] (e.g., ["2023-05-01", "2023-05-31"])
        
        Returns:
            True if query_time is within the range or matches start/end
        """
        start_time = event_range[0] if len(event_range) > 0 else ""
        end_time = event_range[1] if len(event_range) > 1 else ""
        
        if not start_time and not end_time:
            return False
        
        # If event has only one time point (start == end or end is empty)
        if not end_time or start_time == end_time:
            # Check if query_time matches or is a prefix of start_time
            return start_time.startswith(query_time) or query_time.startswith(start_time)
        
        # Event has a time range [start_time, end_time]
        # Check if query_time falls within the range (inclusive)
        # Use string comparison (works for YYYY-MM-DD format)
        
        # Handle different granularities
        # If query is "2023-05", it should match events in May 2023
        # If query is "2023", it should match events in 2023
        
        # Normalize comparison: pad query_time to match comparison
        if query_time <= end_time and query_time >= start_time:
            return True
        
        # Also check if query_time is a prefix match with start or end
        # This handles cases like query="2023" matching event=["2023-05-01", "2023-05-31"]
        if start_time.startswith(query_time) or end_time.startswith(query_time):
            return True
        
        # Check if query_time as a range overlaps with event range
        # For partial dates like "2023-05", treat as the entire month
        if len(query_time) <= len(start_time):
            # query_time could be a partial date representing a period
            # Check if this period overlaps with [start_time, end_time]
            if query_time <= end_time and (query_time + "~") >= start_time:
                return True
        
        return False


# Backward compatibility: keep FactRetriever for any existing code
FactRetriever = LayeredRetriever
