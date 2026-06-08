# src/profile_manager.py
# Profile Management Module (v2.0)

import os
from typing import List, Dict, Any
from src.llm_interface import LLMInterface
from src.embedding import EmbeddingModel
from src.utils import generate_profile_id, cosine_similarity, jaccard_similarity
import config


class ProfileManager:
    """Manage Profile extraction, updates, and batch processing."""
    
    def __init__(self):
        """Initialize profile manager."""
        self.llm = LLMInterface()
        self.embedding_model = EmbeddingModel()
        self.prompt_file = os.path.join(config.PROMPTS_DIR, "profile_extraction_prompt.txt")
        self._load_prompt()
        self.pending_count = 0  # Counter for batch trigger
    
    def _load_prompt(self):
        """Load extraction prompt template."""
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            self.extraction_prompt = f.read()
    
    def check_and_trigger_batch_extraction(self, all_facts: List[Dict[str, Any]], all_profiles: List[Dict[str, Any]]) -> bool:
        """
        Check if batch extraction should be triggered.
        
        Args:
            all_facts: All facts in the system
            all_profiles: All existing profiles
            
        Returns:
            True if batch extraction was triggered, False otherwise
        """
        # Count facts marked for extraction
        pending_facts = [f for f in all_facts if f.get("needs_profile_extraction", False)]
        
        if len(pending_facts) >= config.PROFILE_BATCH_TRIGGER:
            self.batch_extract_profiles(pending_facts, all_facts, all_profiles)
            return True
        
        return False
    
    def batch_extract_profiles(self, pending_facts: List[Dict[str, Any]], all_facts: List[Dict[str, Any]], all_profiles: List[Dict[str, Any]]):
        """
        Batch extract profiles from facts marked with needs_profile_extraction=True.
        
        Args:
            pending_facts: Facts marked for profile extraction
            all_facts: All facts (to update the needs_profile_extraction flag)
            all_profiles: All existing profiles (to append new ones and check duplicates)
        """
        if not pending_facts:
            return
        
        # Group facts by person
        facts_by_person = {}
        for fact in pending_facts:
            for person in fact.get("people", []):
                if person not in facts_by_person:
                    facts_by_person[person] = []
                facts_by_person[person].append(fact)
        
        # Process each person
        for person, facts in facts_by_person.items():
            existing_profiles = [p for p in all_profiles if p["person"] == person]
            
            # LLM extracts profiles
            new_profiles_data = self._llm_extract_profiles(person, facts, existing_profiles)
            
            # Process each new profile
            for profile_data in new_profiles_data:
                if not isinstance(profile_data, dict) or not profile_data.get("content", "").strip():
                    continue
                # Check for duplicates
                if self._is_duplicate_profile(
                    profile_data["content"],
                    profile_data.get("keywords", []),
                    existing_profiles
                ):
                    continue
                
                # Create new profile
                profile_id = generate_profile_id(all_profiles)
                new_profile = {
                    "profile_id": profile_id,
                    "person": person,
                    "content": profile_data["content"],
                    "embedding": self.embedding_model.encode(profile_data["content"]),
                    "keywords": profile_data.get("keywords", []),
                    "evidence": profile_data.get("evidence", [])
                }
                
                all_profiles.append(new_profile)
        
        # Clear flags for processed facts
        for fact in pending_facts:
            fact["needs_profile_extraction"] = False
    
    def _llm_extract_profiles(self, person: str, facts: List[Dict[str, Any]], existing_profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use LLM to extract profile statements from facts.
        
        Args:
            person: Person name
            facts: Facts marked for profile extraction
            existing_profiles: Existing profiles for this person
            
        Returns:
            List of profile data dicts: [{"content": str, "keywords": List[str], "evidence": List[str]}]
        """
        # Build facts string
        facts_str = "\n".join([
            f"- {f['fact_id']}: {f['fact']}"
            for f in facts
        ])
        
        # Build user prompt
        user_prompt = f"""
person: "{person}"
facts:
{facts_str}
"""
        
        if existing_profiles:
            profiles_str = "\n".join([f"- {p['content']}" for p in existing_profiles])
            user_prompt += f"\n\nExisting Profiles for {person}:\n{profiles_str}"
        
        user_prompt += "\n\nExtract stable personal attributes from these facts and generate profile statements."
        
        # Call LLM
        response = self.llm.call_with_retry(self.extraction_prompt, user_prompt, response_format="json", call_type="profile_extraction")
        
        if "error" in response:
            return []
        
        profiles = response.get("profiles", [])
        # 过滤掉 LLM 返回的格式不合规条目（缺少 content 字段或非 dict）
        profiles = [
            p for p in profiles
            if isinstance(p, dict) and p.get("content", "").strip()
        ]
        return profiles
    
    def _is_duplicate_profile(self, content: str, keywords: List[str], existing_profiles: List[Dict[str, Any]]) -> bool:
        """
        Check if a profile content is duplicate with existing profiles.
        
        Args:
            content: New profile content
            keywords: New profile keywords
            existing_profiles: Existing profiles for the same person
            
        Returns:
            True if duplicate, False otherwise
        """
        if not existing_profiles:
            return False
        
        new_embedding = self.embedding_model.encode(content)
        alpha = config.EMBEDDING_WEIGHT_ALPHA
        beta = config.KEYWORD_WEIGHT_BETA
        
        for profile in existing_profiles:
            emb_sim = cosine_similarity(new_embedding, profile["embedding"])
            kw_sim = jaccard_similarity(keywords, profile.get("keywords", []))
            similarity = alpha * emb_sim + beta * kw_sim
            
            if similarity > config.PROFILE_DUPLICATE_THRESHOLD:
                return True
        
        return False
    
    def force_extract_all_pending(self, all_facts: List[Dict[str, Any]], all_profiles: List[Dict[str, Any]]):
        """
        Force extract profiles from all pending facts (used at session end).
        
        Args:
            all_facts: All facts in the system
            all_profiles: All existing profiles
        """
        pending_facts = [f for f in all_facts if f.get("needs_profile_extraction", False)]
        
        if pending_facts:
            self.batch_extract_profiles(pending_facts, all_facts, all_profiles)

