# src/event_manager.py
# Event management module.

import os
from typing import List, Dict, Any
from src.llm_interface import LLMInterface
from src.embedding import EmbeddingModel
from src.utils import generate_event_id, cosine_similarity, jaccard_similarity
import config


class EventManager:
    """Manage Event attribution, creation, and updates."""
    
    def __init__(self):
        """Initialize event manager."""
        self.llm = LLMInterface()
        self.embedding_model = EmbeddingModel()
        self.attribution_prompt_file = os.path.join(config.PROMPTS_DIR, "event_attribution_prompt.txt")
        self.generation_prompt_file = os.path.join(config.PROMPTS_DIR, "event_generation_prompt.txt")
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompt templates."""
        with open(self.attribution_prompt_file, 'r', encoding='utf-8') as f:
            self.attribution_prompt = f.read()
        with open(self.generation_prompt_file, 'r', encoding='utf-8') as f:
            self.generation_prompt = f.read()
    
    def process_event_attribution(self, 
                                  new_fact: Dict[str, Any],
                                  all_facts: List[Dict[str, Any]],
                                  all_events: List[Dict[str, Any]],
                                  use_batch_filter: bool = False) -> List[str]:
        """
        Process event attribution for a new fact.
        
        Args:
            new_fact: The new fact to process
            all_facts: All existing facts
            all_events: All existing events
            use_batch_filter: Whether to use batch LLM filtering for candidate events
            
        Returns:
            List of event_ids that the new fact should join (empty if independent)
        """
        # Step 1: Retrieve similar facts
        similar_facts = self._retrieve_similar_facts(new_fact, all_facts)
        
        if not similar_facts:
            return []  # No similar facts, remain independent
        
        # Step 2: LLM judges whether to cluster
        should_cluster, selected_fact_ids, reason = self._llm_judge_clustering(new_fact, similar_facts)
        
        if not should_cluster or not selected_fact_ids:
            return []  # Remain independent
        
        # Step 3: Collect candidate event_ids from selected facts
        candidate_event_ids = set()
        for fact_id in selected_fact_ids:
            fact = next((f for f in all_facts if f["fact_id"] == fact_id), None)
            if fact and fact.get("event_ids"):
                candidate_event_ids.update(fact["event_ids"])
        
        # Step 4: Create new event or join existing event(s)
        if not candidate_event_ids:
            # Case 1: Create new event (no existing events)
            selected_facts = [f for f in all_facts if f["fact_id"] in selected_fact_ids]
            new_event = self._create_new_event(new_fact, selected_facts, all_events)
            
            # Update selected facts' event_ids
            for fact in selected_facts:
                if "event_ids" not in fact:
                    fact["event_ids"] = []
                fact["event_ids"].append(new_event["event_id"])
            
            all_events.append(new_event)
            return [new_event["event_id"]]
        else:
            # Case 2: Filter and join relevant existing events
            if use_batch_filter:
                # Use batch LLM filtering to select relevant events
                relevant_event_ids = self._batch_filter_relevant_events(
                    new_fact, candidate_event_ids, all_events
                )
                
                if not relevant_event_ids:
                    # No relevant existing events, create new event with selected facts
                    selected_facts = [f for f in all_facts if f["fact_id"] in selected_fact_ids]
                    new_event = self._create_new_event(new_fact, selected_facts, all_events)
                    
                    # Update selected facts' event_ids
                    for fact in selected_facts:
                        if "event_ids" not in fact:
                            fact["event_ids"] = []
                        fact["event_ids"].append(new_event["event_id"])
                    
                    all_events.append(new_event)
                    return [new_event["event_id"]]
                
                event_ids_list = relevant_event_ids
            else:
                # Original behavior: join all candidate events
                event_ids_list = list(candidate_event_ids)
            
            # Update each relevant event
            for event_id in event_ids_list:
                event = next((e for e in all_events if e["event_id"] == event_id), None)
                if event:
                    event["fact_ids"].append(new_fact["fact_id"])
                    self._update_event_info(event, new_fact, all_facts)
            
            return event_ids_list
    
    def _retrieve_similar_facts(self, new_fact: Dict[str, Any], all_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K similar facts for event attribution.
        
        Args:
            new_fact: The new fact
            all_facts: All existing facts
            
        Returns:
            Top-K similar facts with scores
        """
        # Filter facts with overlapping people
        candidates = []
        new_people = set(new_fact.get("people", []))
        
        for fact in all_facts:
            if fact["fact_id"] == new_fact.get("fact_id"):
                continue  # Skip self
            
            fact_people = set(fact.get("people", []))
            if new_people & fact_people:  # Has intersection
                candidates.append(fact)
        
        if not candidates:
            return []
        
        # Calculate similarity scores
        alpha = config.EMBEDDING_WEIGHT_ALPHA
        beta = config.KEYWORD_WEIGHT_BETA
        
        for fact in candidates:
            emb_sim = cosine_similarity(fact["embedding"], new_fact["embedding"])
            kw_sim = jaccard_similarity(fact.get("keywords", []), new_fact.get("keywords", []))
            fact["score"] = alpha * emb_sim + beta * kw_sim
        
        # Sort and return Top-K
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:config.SIMILAR_FACTS_TOP_K]
    
    def _llm_judge_clustering(self, new_fact: Dict[str, Any], similar_facts: List[Dict[str, Any]]) -> tuple:
        """
        Use LLM to judge whether new fact should cluster with similar facts.
        
        Returns:
            (should_cluster: bool, selected_fact_ids: List[str], reason: str)
        """
        # Build user prompt
        user_prompt = f"""
New Fact:
{self._format_fact_for_prompt(new_fact)}

Similar Facts (Top-{len(similar_facts)}):
{self._format_similar_facts_for_prompt(similar_facts)}

Please analyze and determine if the new fact should be clustered with any of the similar facts.
"""
        
        # Call LLM
        response = self.llm.call_with_retry(self.attribution_prompt, user_prompt, response_format="json", call_type="event_attribution")
        
        if "error" in response:
            return False, [], "LLM error"
        
        should_cluster = response.get("should_cluster", False)
        selected_fact_ids = response.get("selected_fact_ids", [])
        reason = response.get("reason", "")
        
        return should_cluster, selected_fact_ids, reason
    
    def _create_new_event(self, new_fact: Dict[str, Any], related_facts: List[Dict[str, Any]], all_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a new event from new fact and related facts.
        
        Returns:
            New event dictionary
        """
        all_facts = [new_fact] + related_facts
        
        # Generate event ID
        event_id = generate_event_id(all_events)
        
        # LLM generates summary and keywords
        event_info = self._llm_generate_event_info(all_facts)
        
        # Aggregate people
        all_people = set()
        for fact in all_facts:
            all_people.update(fact.get("people", []))
        
        # Extract time range
        all_times = []
        for fact in all_facts:
            time_info = fact.get("time", ["", ""])
            if time_info[0]:  # event_time
                all_times.append(time_info[0])
        
        time_range = [min(all_times), max(all_times)] if all_times else ["", ""]
        
        # Generate embedding for summary
        summary_embedding = self.embedding_model.encode(event_info["summary"])
        
        new_event = {
            "event_id": event_id,
            "summary": event_info["summary"],
            "fact_ids": [f["fact_id"] for f in all_facts],
            "people": list(all_people),
            "keywords": event_info["keywords"],
            "time": time_range,
            "embedding": summary_embedding
        }
        
        return new_event
    
    def _update_event_info(self, event: Dict[str, Any], new_fact: Dict[str, Any], all_facts: List[Dict[str, Any]]):
        """
        Update event's summary and keywords after adding a new fact.
        
        Args:
            event: The event to update
            new_fact: The newly added fact
            all_facts: All facts (to retrieve event's facts)
        """
        # Use incremental update: only pass the new fact to LLM
        # LLM will update based on existing summary + new fact
        event_info = self._llm_generate_event_info(
            [new_fact],  # Only pass the new fact for incremental update
            existing_summary=event.get("summary"),
            existing_keywords=event.get("keywords", []),
            is_incremental_update=True
        )
        
        # Update event
        event["summary"] = event_info["summary"]
        event["keywords"] = event_info["keywords"]
        event["embedding"] = self.embedding_model.encode(event_info["summary"])
        
        # Update aggregated people
        all_people = set(event.get("people", []))
        all_people.update(new_fact.get("people", []))
        event["people"] = list(all_people)
        
        # Update time range if necessary
        new_time = new_fact.get("time", ["", ""])[0]
        if new_time:
            event_time = event.get("time", ["", ""])
            if not event_time[0] or new_time < event_time[0]:
                event_time[0] = new_time
            if not event_time[1] or new_time > event_time[1]:
                event_time[1] = new_time
            event["time"] = event_time
    
    def _llm_generate_event_info(self, facts: List[Dict[str, Any]], existing_summary: str = None, existing_keywords: List[str] = None, is_incremental_update: bool = False) -> Dict[str, Any]:
        """
        Use LLM to generate event summary and keywords.
        
        Args:
            facts: List of facts (all facts for new event, or only new fact for update)
            existing_summary: Existing event summary (for updates)
            existing_keywords: Existing event keywords (for updates)
            is_incremental_update: If True, only the new fact is provided, update incrementally
        
        Returns:
            {"summary": str, "keywords": List[str]}
        """
        # Build user prompt based on mode
        if is_incremental_update and existing_summary:
            # Incremental update mode: only include the new fact
            new_fact = facts[0]  # Only one new fact in incremental mode
            user_prompt = f"""
Current Event Summary:
{existing_summary}

Current Event Keywords:
{existing_keywords}

New Fact to Add:
- {new_fact['fact_id']}: {new_fact['fact']}
  People: {new_fact.get('people', [])}
  Keywords: {new_fact.get('keywords', [])}

Please update the event summary and keywords to incorporate this new fact. Keep the summary concise and coherent.
"""
        else:
            # Full generation mode: include all facts (for creating new events)
            facts_str = "\n".join([f"- {f['fact_id']}: {f['fact']}" for f in facts])
            
            user_prompt = f"""
Input Facts:
{facts_str}
"""
            
            if existing_summary:
                user_prompt += f"\nExisting Summary: {existing_summary}"
            if existing_keywords:
                user_prompt += f"\nExisting Keywords: {existing_keywords}"
            
            user_prompt += "\n\nGenerate or update the summary and keywords for this Event."
        
        # Call LLM
        response = self.llm.call_with_retry(self.generation_prompt, user_prompt, response_format="json", call_type="event_generation")
        
        if "error" in response:
            # Fallback: simple concatenation
            return {
                "summary": " ".join([f["fact"] for f in facts[:3]]),
                "keywords": list(set([kw for f in facts for kw in f.get("keywords", [])]))[:6]
            }
        
        return {
            "summary": response.get("summary", ""),
            "keywords": response.get("keywords", [])
        }
    
    def _format_fact_for_prompt(self, fact: Dict[str, Any]) -> str:
        """Format a single fact for LLM prompt."""
        return f"""{{
  "fact_id": "{fact.get('fact_id', 'N/A')}",
  "fact": "{fact['fact']}",
  "people": {fact.get('people', [])},
  "keywords": {fact.get('keywords', [])}
}}"""
    
    def _format_similar_facts_for_prompt(self, facts: List[Dict[str, Any]]) -> str:
        """Format similar facts list for LLM prompt."""
        formatted = []
        for fact in facts:
            formatted.append(f"""  {{
    "fact_id": "{fact['fact_id']}",
    "fact": "{fact['fact']}",
    "people": {fact.get('people', [])},
    "keywords": {fact.get('keywords', [])},
    "event_ids": {fact.get('event_ids', [])}
  }}""")
        return "[\n" + ",\n".join(formatted) + "\n]"
    
    def _batch_filter_relevant_events(self,
                                      new_fact: Dict[str, Any],
                                      candidate_event_ids: set,
                                      all_events: List[Dict[str, Any]],
                                      batch_size: int = 10) -> List[str]:
        """
        Batch filter relevant events using LLM judgment.
        
        Args:
            new_fact: New fact to attribute
            candidate_event_ids: Set of candidate event IDs
            all_events: All existing events
            batch_size: Number of events to process in each batch (default: 10)
        
        Returns:
            List of relevant event IDs
        """
        # Get candidate events with full information
        candidate_events = []
        for event_id in candidate_event_ids:
            event = next((e for e in all_events if e["event_id"] == event_id), None)
            if event:
                candidate_events.append(event)
        
        if not candidate_events:
            return []
        
        # Process in batches
        relevant_event_ids = []
        for i in range(0, len(candidate_events), batch_size):
            batch = candidate_events[i:i+batch_size]
            batch_result = self._llm_judge_relevant_events(new_fact, batch)
            relevant_event_ids.extend(batch_result)
        
        return relevant_event_ids
    
    def _llm_judge_relevant_events(self,
                                   new_fact: Dict[str, Any],
                                   candidate_events: List[Dict[str, Any]]) -> List[str]:
        """
        Use LLM to judge which candidate events the new fact should join (single batch call).
        
        Args:
            new_fact: New fact to attribute
            candidate_events: List of candidate events (up to 10)
        
        Returns:
            List of relevant event IDs
        """
        # Build system prompt
        system_prompt = """You are an AI assistant specialized in event attribution.

Task: Determine which candidate events (if any) the new fact should be added to.

Guidelines:
- The fact should be directly related to the event's theme/topic
- Consider the event summary and keywords carefully
- It's OK if the fact doesn't belong to any of the candidates
- The fact can belong to multiple events if they are all relevant

Output Format (JSON):
{
  "relevant_event_ids": ["E1", "E3", ...] or [],
  "reason": "Brief explanation of why these events are relevant / Explanation of why no events match"
}

"""
        
        # Build candidate events description
        events_description = []
        for event in candidate_events:
            events_description.append(f"""
Event {event['event_id']}:
  Summary: {event['summary']}
  Keywords: {', '.join(event.get('keywords', []))}
""")
        
        # Build user prompt
        user_prompt = f"""
New Fact to Attribute:
  Fact: {new_fact['fact']}
  People: {', '.join(new_fact.get('people', []))}
  Keywords: {', '.join(new_fact.get('keywords', []))}

Candidate Events:
{''.join(events_description)}

Please determine which events (if any) this new fact should be added to.
"""
        
        # Call LLM
        response = self.llm.call_with_retry(system_prompt, user_prompt, response_format="json", call_type="event_attribution")
        
        if "error" in response:
            print(f"  Warning: LLM error in event filtering: {response.get('error')}")
            return []
        
        relevant_ids = response.get("relevant_event_ids", [])
        return relevant_ids
