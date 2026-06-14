# src/query_response.py
# Query response module with layered retrieval.

import os
import time
from typing import List, Dict, Any
from src.llm_interface import LLMInterface
from src.embedding import EmbeddingModel
from src.retrieval import LayeredRetriever
from src.file_storage import FileStorage
import config


class QueryResponder:
    """Handle user queries and generate responses using layered retrieval."""
    
    def __init__(self, conversation_id: str, llm: LLMInterface = None):
        """
        Initialize query responder.
        
        Args:
            conversation_id: Conversation ID for fact retrieval
            llm: Optional LLM interface instance. If not provided, creates a new one.
        """
        self.conversation_id = conversation_id
        self.storage = FileStorage(conversation_id)
        self.retriever = LayeredRetriever()
        self.llm = llm if llm is not None else LLMInterface()
        self.embedding_model = EmbeddingModel()
        
        self.intent_prompt_file = os.path.join(config.PROMPTS_DIR, "query_intent_prompt.txt")
        self.answer_prompt_file = os.path.join(config.PROMPTS_DIR, "answer_generation_prompt.txt")
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompt templates."""
        with open(self.intent_prompt_file, 'r', encoding='utf-8') as f:
            self.intent_prompt = f.read()
        with open(self.answer_prompt_file, 'r', encoding='utf-8') as f:
            self.answer_prompt = f.read()
    
    def _load_answer_prompt(self) -> str:
        """Load the generic answer generation prompt."""
        return self._load_category_prompt(None)

    def _load_category_prompt(self, category: Any = None) -> str:
        """Load the answer prompt for the LoCoMo question category."""
        try:
            normalized_category = int(category)
        except (TypeError, ValueError):
            normalized_category = None

        if normalized_category == 2:
            prompt_name = "answer_generation_prompt_cat2.txt"
        elif normalized_category == 3:
            prompt_name = "answer_generation_prompt_cat3.txt"
        else:
            prompt_name = "answer_generation_prompt.txt"

        prompt_file = os.path.join(config.PROMPTS_DIR, prompt_name)
        if not os.path.exists(prompt_file):
            prompt_file = self.answer_prompt_file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def answer_query(self, query: str, category: Any = None) -> Dict[str, Any]:
        """
        Answer user query using layered retrieval.
        
        Args:
            query: User query string
            category: Optional LoCoMo question category for prompt selection
            Returns:
            Dictionary with answer, retrieved evidence, and latency_breakdown.
        """
        latency: Dict[str, float] = {}

        # Step 1: Extract query intent (need_specific, need_attribute, etc.)
        _t = time.time()
        query_info = self._extract_query_intent(query)
        latency["intent_extraction_ms"] = (time.time() - _t) * 1000

        # Step 2: Generate query embedding
        _t = time.time()
        query_embedding = self.embedding_model.encode(query)
        latency["embedding_ms"] = (time.time() - _t) * 1000
        query_info["embedding"] = query_embedding
        
        # Step 3: Load all data
        all_facts = self.storage.load_facts()
        all_events = self.storage.load_events()
        all_profiles = self.storage.load_profiles()
        
        # Step 4: Retrieve relevant information (layered)
        _t = time.time()
        retrieval_result = self.retriever.retrieve_for_query(
            query_info, 
            all_facts, 
            all_events, 
            all_profiles
        )
        latency["retrieval_ms"] = (time.time() - _t) * 1000
        retrieval_rounds = 1

        # Step 5: Generate the final answer directly from retrieved facts/profiles.
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
        
        # Extract dia_ids from retrieved facts
        evidence = [fact.get("dia_id") for fact in retrieval_result["facts"] if fact.get("dia_id")]
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_evidence": evidence,
            "retrieved_facts": retrieval_result["facts"],
            "retrieved_profiles": retrieval_result["profiles"],
            "event_contexts": retrieval_result["event_contexts"],
            "retrieval_rounds": retrieval_rounds,
            "query_info": {
                "need_specific": query_info.get("need_specific", False),
                "need_attribute": query_info.get("need_attribute", False),
                "people": query_info.get("people", []),
                "keywords": query_info.get("keywords", []),
                "time": query_info.get("time")
            },
            "latency_breakdown": latency,
        }
    
    def _extract_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract query intent using LLM.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with need_specific (always True), need_attribute, people, keywords, time
        """
        user_prompt = f"""
Query: "{query}"

Extract the relevant information for fact retrieval.
"""
        
        response = self.llm.call_with_retry(self.intent_prompt, user_prompt, response_format="json", call_type="query_intent")
        
        if "error" in response:
            # Return default structure if LLM fails
            return {
                "need_specific": True,  # Always True
                "need_attribute": False,
                "people": [],
                "keywords": query.lower().split()[:5],
                "time": None
            }
        
        return {
            "need_specific": True,  # Always True - always retrieve specific facts
            "need_attribute": response.get("need_attribute", False),
            "people": response.get("people", []),
            "keywords": response.get("keywords", []),
            "time": response.get("time")
        }

    def _generate_answer(self,
                        query: str,
                        relevant_facts: List[Dict[str, Any]],
                        relevant_profiles: List[Dict[str, Any]],
                        event_contexts: Dict[str, str],
                        query_info: Dict[str, Any] = None,
                        category: Any = None) -> str:
        """
        Generate answer using LLM based on retrieved information.
        
        Args:
            query: User query
            relevant_facts: Retrieved relevant facts
            relevant_profiles: Retrieved relevant profiles
            event_contexts: Event contexts for facts (fact_id -> event_summary)
            query_info: Parsed query intent used for retrieval
            category: Optional LoCoMo question category for prompt selection
        Returns:
            Generated answer string
        """
        answer_prompt = self._load_category_prompt(category)
        _ = event_contexts, query_info

        profiles_str = self._format_profiles_for_prompt(relevant_profiles)
        facts_str = self._format_facts_for_prompt(relevant_facts, event_contexts={})

        user_prompt = f"""
Query: "{query}"

Selected Profiles:
{profiles_str}

Selected Facts:
{facts_str}

Generate your answer based only on the selected evidence.
"""
        
        response = self.llm.call_with_retry(answer_prompt, user_prompt, response_format="text", call_type="answer_generation")
        
        if "error" in response:
            return "I'm sorry, I encountered an error while generating the answer."
        
        return response.get("content", "No answer generated.")

    def _format_profiles_for_prompt(self, profiles: List[Dict[str, Any]]) -> str:
        """Format profiles with only fields exposed to evidence selection/answering."""
        if not profiles:
            return "(No profiles found)"
        
        formatted = []
        for i, profile in enumerate(profiles, 1):
            profile_id = profile.get('profile_id', 'P?')
            content = profile.get('content', '')
            formatted.append(f"- {profile_id}: \"{content}\"")
        
        return "\n".join(formatted)

    def _format_facts_for_prompt(self, facts: List[Dict[str, Any]], event_contexts: Dict[str, str]) -> str:
        """Format facts with only fields exposed to evidence selection/answering."""
        _ = event_contexts
        if not facts:
            return "(No facts found)"
        
        formatted = []
        for i, fact in enumerate(facts, 1):
            fact_id = fact.get('fact_id', 'F?')
            fact_text = fact.get('fact', '')
            time_field = fact.get('time', [])
            recorded_date = time_field[1] if isinstance(time_field, list) and len(time_field) > 1 else ''
            recorded_str = f" recorded_date={recorded_date}" if recorded_date else " recorded_date=None"

            fact_str = (
                f"- {fact_id}: \"{fact_text}\""
                f"{recorded_str}"
            )
            formatted.append(fact_str)
        
        return "\n".join(formatted)
