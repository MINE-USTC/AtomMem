# Pipeline for processing pre-extracted facts from batch extraction

import json
import os
from typing import List, Dict, Any, Tuple

from src.llm_interface import LLMInterface
from src.embedding import EmbeddingModel
from src.fact_storage import FactStorageManager
from src.query_response import QueryResponder
from src.utils import generate_fact_id, parse_session_datetime
import time


class PreExtractedFactsPipeline:
    """Pipeline for processing pre-extracted facts."""
    
    def __init__(self, conversation_id: str, output_dir: str = "runs/atommem"):
        """
        Initialize the pipeline.
        
        Args:
            conversation_id: Conversation ID for storage
            output_dir: Directory to save output files
        """
        self.conversation_id = conversation_id
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize modules
        self.llm = LLMInterface()
        self.embedding_model = EmbeddingModel()
        self.fact_storage = FactStorageManager(conversation_id)
        # Pass the LLM instance to QueryResponder to share token statistics
        self.query_responder = QueryResponder(conversation_id, llm=self.llm)
        
        # Load metadata extraction prompt
        self.metadata_prompt_file = os.path.join("prompts", "fact_metadata_extraction_prompt.txt")
        self._load_metadata_prompt()
        
    
    def _load_metadata_prompt(self):
        """Load fact metadata extraction prompt."""
        with open(self.metadata_prompt_file, 'r', encoding='utf-8') as f:
            self.metadata_prompt = f.read()
    
    def load_preextracted_facts(self, facts_file: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load pre-extracted facts from JSON file.
        
        Args:
            facts_file: Path to facts JSON file (e.g., output/facts_s26-end.json)
        
        Returns:
            Tuple of (metadata, dialogue_results)
        """
        with open(facts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        results = data.get('results', [])
        
        return metadata, results
    
    def extract_fact_metadata(self, 
                             fact_text: str,
                             dia_id: str,
                             session_time: str,
                             speaker: str,
                             previous_context: str = "") -> Dict[str, Any]:
        """
        Extract metadata for a pre-extracted fact using LLM.
        
        Args:
            fact_text: The fact text (already extracted)
            dia_id: Dialogue ID
            session_time: Session time string
            speaker: Speaker name
            previous_context: Previous dialogue turn for context
        
        Returns:
            Metadata dict with people, keywords, time, needs_profile_extraction
        """
        # Parse interaction time
        interaction_time = parse_session_datetime(session_time)
        
        # Build user prompt
        user_prompt = f"""
Fact: "{fact_text}"
Session Time: "{session_time}"
Previous Context: {previous_context if previous_context else "(No previous context)"}

Extract metadata for this fact.
"""
        
        # Call LLM
        response = self.llm.call_with_retry(self.metadata_prompt, user_prompt, response_format="json", call_type="metadata_extraction")
        
        if "error" in response:
            # Fallback: basic extraction
            print(f"  Warning: LLM error for fact metadata extraction, using fallback")
            return {
                "people": [],
                "keywords": [],
                "time": ["", interaction_time],
                "needs_profile_extraction": False
            }
        
        # Validate and fix time field
        time_info = response.get("time", ["", interaction_time])
        if len(time_info) < 2:
            time_info = time_info + [""] * (2 - len(time_info))
        
        # Ensure interaction_time is set
        time_info[1] = interaction_time
        
        return {
            "people": response.get("people", []),
            "keywords": response.get("keywords", []),
            "time": time_info,
            "needs_profile_extraction": response.get("needs_profile_extraction", False)
        }
    
    def build_fact_tuple(self,
                        fact_text: str,
                        dia_id: str,
                        session_time: str,
                        speaker: str,
                        previous_context: str,
                        existing_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build complete fact tuple from pre-extracted fact text.
        
        Args:
            fact_text: Pre-extracted fact text
            dia_id: Dialogue ID
            session_time: Session time
            speaker: Speaker name
            previous_context: Previous dialogue for context
            existing_facts: Existing facts (for ID generation)
        
        Returns:
            Complete fact dictionary
        """
        # Extract metadata using LLM
        metadata = self.extract_fact_metadata(
            fact_text, dia_id, session_time, speaker, previous_context
        )
        
        # Generate embedding
        embedding = self.embedding_model.encode(fact_text)
        
        # Generate unique fact ID from the in-memory build state. Re-reading the
        # full facts JSON for every incoming fact is both slow and memory-heavy.
        fact_id = generate_fact_id(existing_facts)
        
        # Build complete fact tuple
        fact_tuple = {
            "fact_id": fact_id,
            "fact": fact_text,
            "embedding": embedding,
            "people": metadata["people"],
            "keywords": metadata["keywords"],
            "time": metadata["time"],
            "dia_id": dia_id,
            "event_ids": [],
            "needs_profile_extraction": metadata["needs_profile_extraction"]
        }
        
        return fact_tuple

    @staticmethod
    def _count_session_turns(dialogue_results: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for dialogue in dialogue_results:
            session = dialogue.get('session', 'unknown')
            counts[session] = counts.get(session, 0) + 1
        return counts

    @staticmethod
    def _print_session_progress(session: str, completed_turns: int, total_turns: int) -> None:
        total = max(total_turns, 1)
        completed = min(max(completed_turns, 0), total)
        width = 30
        filled = int(width * completed / total)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r[{session.upper()}] [{bar}] {completed}/{total_turns} turns", end="", flush=True)

    def _finish_session_progress(self, session: str, completed_turns: int, total_turns: int) -> None:
        if completed_turns < total_turns:
            self._print_session_progress(session, completed_turns, total_turns)
        print()
        print(f"[{session.upper()}] Completed", flush=True)
    
    def process_all_facts(self, dialogue_results: List[Dict[str, Any]], start_session: str = None):
        """
        Process all pre-extracted facts in order.
        
        Args:
            dialogue_results: List of dialogue results with facts
            start_session: Optional session name to start from (e.g., 's02', 'afternoon')
        
        Returns:
            Dict with runtime, per_fact_latencies, and storage_snapshots.
        """
        start_time = time.time()
        
        total_dialogues = 0
        total_facts_processed = 0
        existing_facts = list(self.fact_storage.storage.load_facts())
        session_turn_counts = self._count_session_turns(dialogue_results)

        # Efficiency tracking
        per_fact_latencies: List[Dict[str, float]] = []
        storage_snapshots: List[Dict[str, Any]] = []
        cumulative_dialogue_count = 0
        
        # Group dialogues by session for context
        current_session = None
        session_dialogues = []
        session_turn_progress = 0
        
        should_process = (start_session is None)
        
        for idx, dialogue in enumerate(dialogue_results):
            session = dialogue.get('session', 'unknown')
            session_time = dialogue.get('session_time', '')
            dia_id = dialogue.get('dia_id', f'D{idx}')
            speaker = dialogue.get('speaker', 'Unknown')
            facts = dialogue.get('facts', [])
            
            # Track session changes
            if current_session != session:
                # Record storage snapshot at end of previous session
                if current_session is not None and should_process:
                    self._finish_session_progress(
                        current_session,
                        session_turn_progress,
                        session_turn_counts.get(current_session, session_turn_progress),
                    )
                    stored_facts = self.fact_storage.storage.load_facts()
                    stored_events = self.fact_storage.storage.load_events()
                    stored_profiles = self.fact_storage.storage.load_profiles()
                    storage_snapshots.append({
                        "session": current_session,
                        "cumulative_dialogues": cumulative_dialogue_count,
                        "num_facts": len(stored_facts),
                        "num_events": len(stored_events),
                        "num_profiles": len(stored_profiles),
                        "elapsed_seconds": time.time() - start_time,
                    })

                current_session = session
                session_dialogues = []
                session_turn_progress = 0
                
                # Check if this is the session to start from
                if start_session and not should_process:
                    if session.lower() == start_session.lower():
                        should_process = True
                        print(f"\n[{session.upper()}] Processing session...", flush=True)
                        self._print_session_progress(
                            session,
                            session_turn_progress,
                            session_turn_counts.get(session, 0),
                        )
                    else:
                        continue
                elif should_process:
                    print(f"\n[{session.upper()}] Processing session...", flush=True)
                    self._print_session_progress(
                        session,
                        session_turn_progress,
                        session_turn_counts.get(session, 0),
                    )
            
            # Skip if we haven't reached the start session yet
            if not should_process:
                continue
            
            session_dialogues.append(dialogue)
            total_dialogues += 1
            cumulative_dialogue_count += 1
            
            # Build previous context (previous dialogue in same session)
            previous_context = ""
            if len(session_dialogues) > 1:
                prev_dialogue = session_dialogues[-2]
                prev_speaker = prev_dialogue.get('speaker', 'Unknown')
                prev_text = prev_dialogue.get('text', '')
                previous_context = f"{prev_speaker}: {prev_text}"
            
            # Process each fact
            for fact_text in facts:
                # Time the full fact processing (metadata extraction + pipeline)
                _fact_start = time.time()

                # Build complete fact tuple
                fact_tuple = self.build_fact_tuple(
                    fact_text=fact_text,
                    dia_id=dia_id,
                    session_time=session_time,
                    speaker=speaker,
                    previous_context=previous_context,
                    existing_facts=existing_facts
                )
                metadata_ms = (time.time() - _fact_start) * 1000

                # Process fact through pipeline (with batch event filter enabled)
                result = self.fact_storage.process_new_fact(
                    fact_tuple, 
                    use_batch_event_filter=True
                )

                total_fact_ms = (time.time() - _fact_start) * 1000

                # Collect per-fact latency record
                breakdown = result.get("latency_breakdown", {})
                per_fact_latencies.append({
                    "action": result["action"],
                    "metadata_extraction_ms": metadata_ms,
                    "dedup_check_ms": breakdown.get("dedup_check_ms", 0.0),
                    "conflict_detect_ms": breakdown.get("conflict_detect_ms", 0.0),
                    "storage_ms": breakdown.get("storage_ms", 0.0),
                    "event_attribution_ms": breakdown.get("event_attribution_ms", 0.0),
                    "profile_check_ms": breakdown.get("profile_check_ms", 0.0),
                    "total_ms": total_fact_ms,
                })

                if result["action"] != "IGNORE":
                    existing_facts.append(fact_tuple)
                    total_facts_processed += 1

            session_turn_progress += 1
            self._print_session_progress(
                session,
                session_turn_progress,
                session_turn_counts.get(session, session_turn_progress),
            )

        # Record snapshot for the final session
        if current_session is not None and should_process:
            self._finish_session_progress(
                current_session,
                session_turn_progress,
                session_turn_counts.get(current_session, session_turn_progress),
            )
            stored_facts = self.fact_storage.storage.load_facts()
            stored_events = self.fact_storage.storage.load_events()
            stored_profiles = self.fact_storage.storage.load_profiles()
            storage_snapshots.append({
                "session": current_session,
                "cumulative_dialogues": cumulative_dialogue_count,
                "num_facts": len(stored_facts),
                "num_events": len(stored_events),
                "num_profiles": len(stored_profiles),
                "elapsed_seconds": time.time() - start_time,
            })
        
        self.fact_storage.force_profile_extraction()
        
        end_time = time.time()
        runtime = end_time - start_time

        return {
            "runtime": runtime,
            "total_dialogues": total_dialogues,
            "total_facts_processed": total_facts_processed,
            "per_fact_latencies": per_fact_latencies,
            "storage_snapshots": storage_snapshots,
        }
    

