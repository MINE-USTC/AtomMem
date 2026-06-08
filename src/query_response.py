# src/query_response.py
# Query Response Module (v2.0 - With layered retrieval)

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
        self.evidence_summary_prompt_file = os.path.join(config.PROMPTS_DIR, "evidence_summary_prompt.txt")
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompt templates."""
        with open(self.intent_prompt_file, 'r', encoding='utf-8') as f:
            self.intent_prompt = f.read()
        with open(self.answer_prompt_file, 'r', encoding='utf-8') as f:
            self.answer_prompt = f.read()
        with open(self.evidence_summary_prompt_file, 'r', encoding='utf-8') as f:
            self.evidence_summary_prompt = f.read()
    
    def _load_answer_prompt(self) -> str:
        """Load the generic answer generation prompt."""
        with open(self.answer_prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Answer user query using layered retrieval.
        
        Args:
            query: User query string
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

        # Step 5: Compress retrieved facts/profiles into a query-conditioned evidence state.
        # Event summaries are intentionally not exposed here; events only affect compensation retrieval.
        _t = time.time()
        evidence_summary = self._summarize_retrieved_evidence(
            query,
            query_info,
            retrieval_result["facts"],
            retrieval_result["profiles"],
        )
        latency["evidence_summary_ms"] = (time.time() - _t) * 1000

        # Step 5b: Keep retrieving while evidence is insufficient. A round is useful
        # only if the summarizer adopts new supporting facts/profiles.
        active_query_info = query_info
        consecutive_no_useful_rounds = 0
        no_enough_information = False
        max_no_useful_rounds = 3

        while not evidence_summary.get("is_sufficient", False):
            if not self._should_run_followup_retrieval(evidence_summary):
                no_enough_information = True
                break

            previous_support_ids = self._get_supporting_evidence_ids(evidence_summary)
            expanded_query_info = self._build_followup_query_info(query, active_query_info, evidence_summary)
            retrieval_rounds += 1

            _t = time.time()
            expanded_query_text = self._build_followup_query_text(query, expanded_query_info)
            expanded_query_info["embedding"] = self.embedding_model.encode(expanded_query_text)
            latency[f"followup_{retrieval_rounds}_embedding_ms"] = (time.time() - _t) * 1000

            _t = time.time()
            followup_result = self.retriever.retrieve_for_query(
                expanded_query_info,
                all_facts,
                all_events,
                all_profiles,
            )
            latency[f"followup_{retrieval_rounds}_retrieval_ms"] = (time.time() - _t) * 1000

            retrieval_result = self._merge_retrieval_results(retrieval_result, followup_result)
            active_query_info = expanded_query_info

            _t = time.time()
            evidence_summary = self._summarize_retrieved_evidence(
                query,
                active_query_info,
                retrieval_result["facts"],
                retrieval_result["profiles"],
            )
            latency[f"followup_{retrieval_rounds}_evidence_summary_ms"] = (time.time() - _t) * 1000

            current_support_ids = self._get_supporting_evidence_ids(evidence_summary)
            if current_support_ids - previous_support_ids:
                consecutive_no_useful_rounds = 0
            else:
                consecutive_no_useful_rounds += 1

            if consecutive_no_useful_rounds >= max_no_useful_rounds:
                no_enough_information = True
                break
        
        # Step 6: Generate the final answer unless repeated retrieval failed
        # to add useful supporting evidence.
        if no_enough_information and not evidence_summary.get("is_sufficient", False):
            answer = "no enough information"
            latency["answer_generation_ms"] = 0.0
        else:
            _t = time.time()
            answer = self._generate_answer(
                query, 
                retrieval_result["facts"],
                retrieval_result["profiles"],
                retrieval_result["event_contexts"],
                query_info=active_query_info,
                evidence_summary=evidence_summary,
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
            "evidence_summary": evidence_summary,
            "retrieval_rounds": retrieval_rounds,
            "no_useful_retrieval_rounds": consecutive_no_useful_rounds,
            "stopped_for_insufficient_evidence": no_enough_information,
            "query_info": {
                "need_specific": active_query_info.get("need_specific", False),
                "need_attribute": active_query_info.get("need_attribute", False),
                "people": active_query_info.get("people", []),
                "keywords": active_query_info.get("keywords", []),
                "time": active_query_info.get("time")
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

    def _summarize_retrieved_evidence(self,
                                      query: str,
                                      query_info: Dict[str, Any],
                                      relevant_facts: List[Dict[str, Any]],
                                      relevant_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select retrieved facts/profiles that are useful for the query.

        Event summaries are not provided to this step. Events remain useful only for
        compensation retrieval, where they help surface additional raw facts.
        """
        _ = query_info
        if not relevant_facts and not relevant_profiles:
            return {
                "is_sufficient": False,
                "useful_fact_ids": [],
                "useful_profile_ids": [],
                "missing_information": "No supporting memory evidence was retrieved.",
                "needs_more_retrieval": False,
                "followup_query": "",
                "followup_keywords": [],
            }

        user_prompt = f"""
Query: "{query}"

Retrieved Profiles:
{self._format_profiles_for_prompt(relevant_profiles)}

Retrieved Facts:
{self._format_facts_for_prompt(relevant_facts, event_contexts={})}

Select the useful evidence IDs for answering the query.
"""

        response = self.llm.call_with_retry(
            self.evidence_summary_prompt,
            user_prompt,
            response_format="json",
            call_type="evidence_summary",
        )

        if "error" in response:
            return self._fallback_evidence_summary(relevant_facts, relevant_profiles)

        useful_fact_ids = self._normalize_selected_ids(
            response.get("useful_fact_ids", response.get("supporting_fact_ids", [])),
            relevant_facts,
            "fact_id",
        )
        useful_profile_ids = self._normalize_selected_ids(
            response.get("useful_profile_ids", response.get("supporting_profile_ids", [])),
            relevant_profiles,
            "profile_id",
        )

        has_selection_fields = (
            "useful_fact_ids" in response
            or "useful_profile_ids" in response
            or "supporting_fact_ids" in response
            or "supporting_profile_ids" in response
        )
        if not has_selection_fields:
            return self._fallback_evidence_summary(relevant_facts, relevant_profiles)

        return {
            "is_sufficient": bool(response.get("is_sufficient", False)),
            "useful_fact_ids": useful_fact_ids,
            "useful_profile_ids": useful_profile_ids,
            "missing_information": response.get("missing_information", ""),
            "needs_more_retrieval": bool(response.get("needs_more_retrieval", False)),
            "followup_query": response.get("followup_query", ""),
            "followup_keywords": response.get("followup_keywords", []),
        }

    def _should_run_followup_retrieval(self, evidence_summary: Dict[str, Any]) -> bool:
        """Decide whether to run another retrieval round based on the evidence state."""
        if evidence_summary.get("is_sufficient", False):
            return False

        followup_query = evidence_summary.get("followup_query", "")
        followup_keywords = evidence_summary.get("followup_keywords", [])
        has_followup_query = isinstance(followup_query, str) and bool(followup_query.strip())
        has_followup_keywords = isinstance(followup_keywords, list) and bool(followup_keywords)

        return bool(evidence_summary.get("needs_more_retrieval", False) and (has_followup_query or has_followup_keywords))

    def _get_supporting_evidence_ids(self, evidence_summary: Dict[str, Any]) -> set:
        """Return normalized supporting fact/profile ids accepted by the summarizer."""
        support_ids = set()
        fact_ids = evidence_summary.get("useful_fact_ids", evidence_summary.get("supporting_fact_ids", []))
        profile_ids = evidence_summary.get("useful_profile_ids", evidence_summary.get("supporting_profile_ids", []))

        for fact_id in fact_ids or []:
            if isinstance(fact_id, str) and fact_id.strip():
                support_ids.add(f"fact:{fact_id.strip()}")
        for profile_id in profile_ids or []:
            if isinstance(profile_id, str) and profile_id.strip():
                support_ids.add(f"profile:{profile_id.strip()}")
        return support_ids

    def _build_followup_query_info(self,
                                   query: str,
                                   query_info: Dict[str, Any],
                                   evidence_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Build a fresh follow-up query intent from the evidence gap."""
        followup_keywords = self._dedupe_preserve_order([
            keyword
            for keyword in evidence_summary.get("followup_keywords", []) or []
            if isinstance(keyword, str) and keyword.strip()
        ])

        expanded_query_info = {
            "need_specific": query_info.get("need_specific", True),
            "need_attribute": query_info.get("need_attribute", False),
            "people": list(query_info.get("people", []) or []),
            "keywords": followup_keywords,
            "time": None,
            "followup_query": self._normalize_followup_query(query, evidence_summary),
        }

        return expanded_query_info

    def _build_followup_query_text(self, query: str, query_info: Dict[str, Any]) -> str:
        """Build embedding text for follow-up retrieval."""
        followup_query = query_info.get("followup_query")
        keywords = " ".join(query_info.get("keywords", []) or [])
        people = " ".join(query_info.get("people", []) or [])
        return " ".join(part for part in [followup_query, people, keywords] if part) or query

    def _normalize_followup_query(self, original_query: str, evidence_summary: Dict[str, Any]) -> str:
        """Use the summarizer's rewritten query, falling back to generated keywords."""
        followup_query = evidence_summary.get("followup_query", "")
        if isinstance(followup_query, str) and followup_query.strip():
            return followup_query.strip()

        keywords = [
            keyword.strip()
            for keyword in evidence_summary.get("followup_keywords", []) or []
            if isinstance(keyword, str) and keyword.strip()
        ]
        if keywords:
            return " ".join(keywords)

        return original_query

    def _merge_retrieval_results(self,
                                 first: Dict[str, Any],
                                 second: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two retrieval results while preserving first-round ordering."""
        return {
            "facts": self._merge_items_by_id(first.get("facts", []), second.get("facts", []), "fact_id"),
            "profiles": self._merge_items_by_id(first.get("profiles", []), second.get("profiles", []), "profile_id"),
            "event_contexts": {
                **first.get("event_contexts", {}),
                **second.get("event_contexts", {}),
            },
        }

    def _merge_items_by_id(self,
                           first_items: List[Dict[str, Any]],
                           second_items: List[Dict[str, Any]],
                           id_field: str) -> List[Dict[str, Any]]:
        """Merge retrieved items by id without reordering first-round evidence."""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for item in first_items + second_items:
            if not isinstance(item, dict):
                continue
            item_id = item.get(id_field)
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            merged.append(item)
        return merged

    def _dedupe_preserve_order(self, values: List[str]) -> List[str]:
        """Deduplicate strings while preserving order."""
        result = []
        seen = set()
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            value = value.strip()
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    def _normalize_selected_ids(self,
                                selected_ids: Any,
                                available_items: List[Dict[str, Any]],
                                id_field: str) -> List[str]:
        """Keep selected IDs that exist in the retrieved evidence, preserving retrieval order."""
        if not isinstance(selected_ids, list):
            return []

        selected_set = {
            item.strip()
            for item in selected_ids
            if isinstance(item, str) and item.strip()
        }
        if not selected_set:
            return []

        normalized = []
        for item in available_items:
            item_id = item.get(id_field)
            if item_id in selected_set and item_id not in normalized:
                normalized.append(item_id)
        return normalized

    def _select_items_by_ids(self,
                             items: List[Dict[str, Any]],
                             selected_ids: List[str],
                             id_field: str) -> List[Dict[str, Any]]:
        """Return items whose ID was selected, preserving selected ID order."""
        lookup = {
            item.get(id_field): item
            for item in items
            if isinstance(item, dict) and item.get(id_field)
        }
        selected = []
        seen = set()
        for item_id in selected_ids or []:
            if item_id in seen:
                continue
            item = lookup.get(item_id)
            if item:
                selected.append(item)
                seen.add(item_id)
        return selected

    def _fallback_evidence_summary(self,
                                   relevant_facts: List[Dict[str, Any]],
                                   relevant_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select all retrieved evidence if LLM evidence selection fails."""
        return {
            "is_sufficient": bool(relevant_facts or relevant_profiles),
            "useful_fact_ids": [f.get("fact_id") for f in relevant_facts if f.get("fact_id")],
            "useful_profile_ids": [p.get("profile_id") for p in relevant_profiles if p.get("profile_id")],
            "missing_information": "",
            "needs_more_retrieval": False,
            "followup_query": "",
            "followup_keywords": [],
        }
    
    def _generate_answer(self,
                        query: str,
                        relevant_facts: List[Dict[str, Any]],
                        relevant_profiles: List[Dict[str, Any]],
                        event_contexts: Dict[str, str],
                        query_info: Dict[str, Any] = None,
                        evidence_summary: Dict[str, Any] = None) -> str:
        """
        Generate answer using LLM based on retrieved information.
        
        Args:
            query: User query
            relevant_facts: Retrieved relevant facts
            relevant_profiles: Retrieved relevant profiles
            event_contexts: Event contexts for facts (fact_id -> event_summary)
            query_info: Parsed query intent used for retrieval
            evidence_summary: Query-conditioned evidence state built from facts/profiles
        Returns:
            Generated answer string
        """
        answer_prompt = self._load_answer_prompt()
        _ = event_contexts, query_info

        selected_facts = relevant_facts
        selected_profiles = relevant_profiles
        if evidence_summary:
            useful_fact_ids = evidence_summary.get(
                "useful_fact_ids",
                evidence_summary.get("supporting_fact_ids", []),
            )
            useful_profile_ids = evidence_summary.get(
                "useful_profile_ids",
                evidence_summary.get("supporting_profile_ids", []),
            )
            selected_facts = self._select_items_by_ids(relevant_facts, useful_fact_ids, "fact_id")
            selected_profiles = self._select_items_by_ids(relevant_profiles, useful_profile_ids, "profile_id")

        profiles_str = self._format_profiles_for_prompt(selected_profiles)
        facts_str = self._format_facts_for_prompt(selected_facts, event_contexts={})

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

    def _format_evidence_summary_for_prompt(self, evidence_summary: Dict[str, Any]) -> str:
        """Format evidence-selection metadata for final answer generation."""
        useful_fact_ids = evidence_summary.get(
            "useful_fact_ids",
            evidence_summary.get("supporting_fact_ids", []),
        )
        useful_profile_ids = evidence_summary.get(
            "useful_profile_ids",
            evidence_summary.get("supporting_profile_ids", []),
        )
        lines = [
            f"- is_sufficient: {bool(evidence_summary.get('is_sufficient', False))}",
            f"- useful_fact_ids: {useful_fact_ids}",
            f"- useful_profile_ids: {useful_profile_ids}",
        ]
        missing_information = evidence_summary.get("missing_information")
        if missing_information:
            lines.extend(["- missing_information:", str(missing_information)])

        return "\n".join(lines)

    def _format_query_intent_for_prompt(self, query_info: Dict[str, Any]) -> str:
        """Format query intent so the answer model sees the retrieval target."""
        people = query_info.get("people", [])
        keywords = query_info.get("keywords", [])
        time_info = query_info.get("time")
        need_attribute = query_info.get("need_attribute", False)
        followup_query = query_info.get("followup_query")

        lines = [
            f"- people: {people if people else '[]'}",
            f"- keywords: {keywords if keywords else '[]'}",
            f"- time: {time_info if time_info else 'None'}",
            f"- needs_profile: {bool(need_attribute)}",
        ]
        if followup_query:
            lines.append(f"- followup_query: {followup_query}")
        return "\n".join(lines)
    
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
