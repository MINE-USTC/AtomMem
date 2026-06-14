# src/fact_storage.py
# Fact storage and processing module.

import time
from typing import List, Dict, Any, Optional
from src.llm_interface import LLMInterface
from src.event_manager import EventManager
from src.profile_manager import ProfileManager
from src.file_storage import FileStorage
from src.utils import cosine_similarity, jaccard_similarity
import config


def people_sets_compatible(left_people: set, right_people: set) -> bool:
    """Reject only when both people sets are non-empty and disjoint."""
    return not left_people or not right_people or bool(left_people & right_people)


class FactStorageManager:
    """Manage fact storage, deduplication, conflict detection, and event/profile integration."""
    
    def __init__(self, conversation_id: str):
        """
        Initialize storage manager.
        
        Args:
            conversation_id: Conversation or memory namespace ID.
        """
        self.conversation_id = conversation_id
        self.storage = FileStorage(conversation_id)
        self.event_manager = EventManager()
        self.profile_manager = ProfileManager()
        self.llm = LLMInterface()
    
    def process_new_fact(self, new_fact: Dict[str, Any],
                          use_batch_event_filter: bool = False,
                          skip_post_processing: bool = False) -> Dict[str, Any]:
        """
        Process a new fact through the complete pipeline:
        1. Deduplication check
        2. Conflict detection
        3. Storage
        4. Event attribution        (skipped when skip_post_processing=True)
        5. Profile extraction check (skipped when skip_post_processing=True)

        Args:
            new_fact: New fact to process
            use_batch_event_filter: Whether to use batch LLM filtering for event attribution
            skip_post_processing: When True, skip steps 4 & 5 (e.g., for assistant-role facts
                                  that don't carry user-relevant event/profile information)

        Returns:
            Processing result dictionary with latency_breakdown for efficiency analysis.
        """
        latency: Dict[str, float] = {}

        # Load existing data
        all_facts = self.storage.load_facts()
        all_events = self.storage.load_events()
        all_profiles = self.storage.load_profiles()
        
        # Step 1: Deduplication check
        _t = time.time()
        is_dup = self._is_duplicate(new_fact, all_facts)
        latency["dedup_check_ms"] = (time.time() - _t) * 1000
        if is_dup:
            return {
                "action": "IGNORE",
                "reason": "Duplicate fact detected (similarity > 0.95)",
                "fact_id": None,
                "latency_breakdown": latency,
            }
        
        # Step 2: Conflict detection
        _t = time.time()
        conflict_fact = self._detect_conflict(new_fact, all_facts)
        latency["conflict_detect_ms"] = (time.time() - _t) * 1000
        if conflict_fact:
            conflict_fact["fact"] = new_fact["fact"]
            conflict_fact["embedding"] = new_fact["embedding"]
            conflict_fact["keywords"] = new_fact["keywords"]
            conflict_fact["time"] = new_fact["time"]
            self.storage.save_facts(all_facts)
            return {
                "action": "CONFLICT_RESOLVED",
                "reason": "Conflicting fact updated with new information",
                "fact_id": conflict_fact["fact_id"],
                "latency_breakdown": latency,
            }
        
        # Step 3: Create and store new fact
        _t = time.time()
        all_facts.append(new_fact)
        self.storage.save_facts(all_facts)
        latency["storage_ms"] = (time.time() - _t) * 1000

        if skip_post_processing:
            latency["event_attribution_ms"] = 0.0
            latency["profile_check_ms"] = 0.0
            return {
                "action": "CREATE",
                "reason": "New fact created successfully (post-processing skipped)",
                "fact_id": new_fact["fact_id"],
                "event_ids": [],
                "profile_triggered": False,
                "latency_breakdown": latency,
            }

        # Step 4: Event attribution
        _t = time.time()
        event_ids = self.event_manager.process_event_attribution(
            new_fact, all_facts, all_events, use_batch_filter=use_batch_event_filter
        )
        latency["event_attribution_ms"] = (time.time() - _t) * 1000
        new_fact["event_ids"] = event_ids
        
        # Save updated facts and events
        self.storage.save_facts(all_facts)
        self.storage.save_events(all_events)
        
        # Step 5: Check and trigger profile extraction
        _t = time.time()
        profile_triggered = self.profile_manager.check_and_trigger_batch_extraction(all_facts, all_profiles)
        latency["profile_check_ms"] = (time.time() - _t) * 1000
        if profile_triggered:
            self.storage.save_facts(all_facts)
            self.storage.save_profiles(all_profiles)
        
        return {
            "action": "CREATE",
            "reason": "New fact created successfully",
            "fact_id": new_fact["fact_id"],
            "event_ids": event_ids,
            "profile_triggered": profile_triggered,
            "latency_breakdown": latency,
        }
    
    def _is_duplicate(self, new_fact: Dict[str, Any], existing_facts: List[Dict[str, Any]]) -> bool:
        """
        Check if new fact is duplicate with existing facts.
        
        Args:
            new_fact: New fact
            existing_facts: All existing facts
            
        Returns:
            True if duplicate, False otherwise
        """
        # Reject only when both people sets are non-empty and disjoint.
        candidates = []
        new_people = set(new_fact.get("people", []))
        
        for fact in existing_facts:
            fact_people = set(fact.get("people", []))
            if not people_sets_compatible(new_people, fact_people):
                continue
            candidates.append(fact)
        
        # Check embedding similarity
        for fact in candidates:
            similarity = cosine_similarity(fact["embedding"], new_fact["embedding"])
            if similarity > config.DUPLICATE_THRESHOLD:
                return True
        
        return False
    
    def _detect_conflict(self, new_fact: Dict[str, Any], existing_facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Detect if new fact conflicts with existing facts.
        
        Args:
            new_fact: New fact
            existing_facts: All existing facts
            
        Returns:
            Conflicting fact if found, None otherwise
        """
        # Step 1: Filter by people and time
        candidates = []
        new_people = set(new_fact.get("people", []))
        new_time = new_fact.get("time", ["", ""])
        
        for fact in existing_facts:
            # Reject only when both people sets are non-empty and disjoint.
            fact_people = set(fact.get("people", []))
            if not people_sets_compatible(new_people, fact_people):
                continue
            
            # Filter by time (optional, relaxed check)
            # If both facts have time info, check for overlap
            fact_time = fact.get("time", ["", ""])
            if new_time[0] and fact_time[0]:
                # Check if time prefixes match or are close
                # This is a relaxed check - facts with similar time are more likely to conflict
                pass  # Include all facts with matching people for now
            
            candidates.append(fact)
        
        if not candidates:
            return None
        
        # Step 2: Calculate similarity and filter high-similarity candidates
        alpha = config.EMBEDDING_WEIGHT_ALPHA
        beta = config.KEYWORD_WEIGHT_BETA
        
        high_similarity_candidates = []
        for fact in candidates:
            emb_sim = cosine_similarity(fact["embedding"], new_fact["embedding"])
            kw_sim = jaccard_similarity(fact.get("keywords", []), new_fact.get("keywords", []))
            similarity = alpha * emb_sim + beta * kw_sim
            
            # Only consider facts with high similarity as potential conflicts
            if similarity > 0.6:
                fact["similarity"] = similarity
                high_similarity_candidates.append(fact)
        
        if not high_similarity_candidates:
            return None
        
        # Step 3: Sort by similarity and take top candidates for LLM judgment
        high_similarity_candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top_candidates = high_similarity_candidates[:5]  # Top-5 most similar
        
        # Step 4: Use LLM to detect logical conflict
        conflict_fact = self._llm_detect_conflict(new_fact, top_candidates)
        return conflict_fact
    
    def _llm_detect_conflict(self, new_fact: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Use LLM to detect if there is a logical conflict.
        
        Returns:
            Conflicting fact if found, None otherwise
        """
        # Build prompt
        system_prompt = """You are an AI assistant specialized in detecting logical conflicts between facts.

Task: Determine if the new fact conflicts with any of the candidate facts.

A conflict exists when:
- Two facts make contradictory claims about the same thing
- Examples: different ages, different locations, opposite preferences

Output Format (JSON):
{
  "has_conflict": true/false,
  "conflict_fact_id": "F1" or null,
  "reason": "Explanation"
}
"""
        
        new_fact_str = f"New Fact: {new_fact['fact']}"
        candidates_str = "\n".join([f"- {f['fact_id']}: {f['fact']}" for f in candidates])
        
        user_prompt = f"""
{new_fact_str}

Candidate Facts:
{candidates_str}

Determine if there is a logical conflict.
"""
        
        response = self.llm.call_with_retry(system_prompt, user_prompt, response_format="json", call_type="conflict_detection")
        
        if "error" in response or not response.get("has_conflict", False):
            return None
        
        conflict_id = response.get("conflict_fact_id")
        if conflict_id:
            return next((f for f in candidates if f["fact_id"] == conflict_id), None)
        
        return None
    
    def force_profile_extraction(self):
        """
        Force profile extraction for all pending facts (called at session end).
        """
        all_facts = self.storage.load_facts()
        all_profiles = self.storage.load_profiles()
        
        self.profile_manager.force_extract_all_pending(all_facts, all_profiles)
        
        # Save updated data
        self.storage.save_facts(all_facts)
        self.storage.save_profiles(all_profiles)
