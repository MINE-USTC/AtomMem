# src/fact_extraction.py
# Fact Extraction Module (v2.0 - With Personal Info Detection)

import os
from typing import List, Dict, Any, Tuple
from src.llm_interface import LLMInterface
from src.embedding import EmbeddingModel
from src.utils import load_json, generate_fact_id
import config


class FactExtractor:
    """Extract facts from conversation messages with personal information detection."""
    
    def __init__(self):
        """Initialize fact extractor."""
        self.llm = LLMInterface()
        self.embedding_model = EmbeddingModel()
        self.prompt_file = os.path.join(config.PROMPTS_DIR, "extraction_prompt.txt")
        self._load_prompt()
    
    def _load_prompt(self):
        """Load extraction prompt template."""
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
    
    def extract_facts(self, 
                     current_message: Dict[str, Any],
                     context_messages: List[Dict[str, Any]],
                     interaction_time: str,
                     existing_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract facts from current message with context.
        
        Args:
            current_message: Current message dict with speaker, dia_id, text
            context_messages: Recent context messages (k=1 round)
            interaction_time: Session date in YYYY-MM-DD format
            existing_facts: Existing facts for ID generation
            
        Returns:
            List of extracted fact dictionaries with needs_profile_extraction flag
        """
        # Build context string
        context_str = self._build_context_string(context_messages)
        current_str = self._build_message_string(current_message) 
        
        # Build user prompt
        user_prompt = f"""
Previous Context:
{context_str}

Extract facts from the Current Message: {current_str}

Interaction Time: {interaction_time}

Remember to:
1. Make each fact complete and standalone (with resolved pronouns)
2. If necessary, preserve logical relationships based on context (e.g. cause effect, conditions)
3. Determine if each fact contains personal information (needs_profile_extraction)
4. If image information is provided (blip_caption/query), use it to understand what the image shows
"""
        
        # Call LLM
        response = self.llm.call_with_retry(self.system_prompt, user_prompt, response_format="json")
        
        if "error" in response:
            return []
        
        # Parse response and generate fact tuples
        facts = []
        
        # Handle different response formats
        if isinstance(response, dict):
            extracted_facts = response.get("facts", [])
        elif isinstance(response, list):
            extracted_facts = response
        else:
            extracted_facts = []
        
        for fact_data in extracted_facts:
            # Skip if fact text is empty
            if not fact_data.get("fact"):
                continue
                
            # Generate embedding
            fact_text = fact_data["fact"]
            embedding = self.embedding_model.encode(fact_text)
            
            # Generate unique fact ID
            fact_id = generate_fact_id(existing_facts + facts)
            
            # Process time information
            time_info = fact_data.get("time", ["", ""])
            if len(time_info) < 2:
                time_info = time_info + [""] * (2 - len(time_info))
            time_info[1] = interaction_time  # Set interaction time
            
            # Build fact tuple (v2.0 with new fields)
            fact_tuple = {
                "fact_id": fact_id,
                "fact": fact_text,
                "embedding": embedding,
                "people": fact_data.get("people", []),
                "keywords": fact_data.get("keywords", []),
                "time": time_info,
                "dia_id": current_message["dia_id"],
                "event_ids": [],  # Empty initially
                "needs_profile_extraction": fact_data.get("needs_profile_extraction", False)
            }
            
            facts.append(fact_tuple)
        
        return facts
    
    def _build_message_string(self, message: Dict[str, Any]) -> str:
        """
        Build message string with optional image information.
        
        Args:
            message: Message dict
            
        Returns:
            Formatted message string
        """
        msg_str = f"{message['speaker']}: {message['text']}"
        
        # Check if message contains image information
        if "img_url" in message and message.get("img_url"):
            # Add image auxiliary information
            image_info_parts = []
            
            if "blip_caption" in message and message["blip_caption"]:
                image_info_parts.append(f"[Image Description: {message['blip_caption']}]")
            
            if "query" in message and message["query"]:
                image_info_parts.append(f"[Image Keywords: {message['query']}]")
            
            if image_info_parts:
                image_info = " ".join(image_info_parts)
                msg_str = f"{msg_str}\n  {image_info}"
        
        return msg_str
    
    def _build_context_string(self, messages: List[Dict[str, Any]]) -> str:
        """
        Build context string from messages with image information support.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Formatted context string
        """
        if not messages:
            return "(No previous context)"
        
        context_lines = []
        for msg in messages:
            msg_str = self._build_message_string(msg)
            context_lines.append(msg_str)
        
        return "\n".join(context_lines)

