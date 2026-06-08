# src/file_storage.py
# File Storage Interface (v2.0 - With Event and Profile support)

import os
from typing import List, Dict, Any, Optional
from src.utils import load_json, save_json, ensure_directory
import config


class FileStorage:
    """File-based storage for facts, events, and profiles."""
    
    def __init__(self, conversation_id: str):
        """
        Initialize file storage.
        
        Args:
            conversation_id: Conversation or memory namespace ID.
        """
        self.conversation_id = conversation_id
        self.facts_file = os.path.join(config.FACTS_DIR, f"facts_{conversation_id}.json")
        self.events_file = os.path.join(config.FACTS_DIR, f"events_{conversation_id}.json")
        self.profiles_file = os.path.join(config.FACTS_DIR, f"profiles_{conversation_id}.json")
        self.entity_graph_file = os.path.join(config.FACTS_DIR, f"entity_graph_{conversation_id}.json")
        ensure_directory(config.FACTS_DIR)
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage files if not exist."""
        if not os.path.exists(self.facts_file):
            save_json({"conversation_id": self.conversation_id, "facts": []}, self.facts_file)
        if not os.path.exists(self.events_file):
            save_json({"conversation_id": self.conversation_id, "events": []}, self.events_file)
        if not os.path.exists(self.profiles_file):
            save_json({"conversation_id": self.conversation_id, "profiles": []}, self.profiles_file)
    
    # ==================== Facts Operations ====================
    
    def load_facts(self) -> List[Dict[str, Any]]:
        """Load all facts from storage."""
        data = load_json(self.facts_file)
        return data.get("facts", [])
    
    def save_facts(self, facts: List[Dict[str, Any]]) -> None:
        """Save facts to storage."""
        save_json({"conversation_id": self.conversation_id, "facts": facts}, self.facts_file)
    
    def add_fact(self, fact: Dict[str, Any]) -> None:
        """Add a new fact to storage."""
        facts = self.load_facts()
        facts.append(fact)
        self.save_facts(facts)
    
    def update_fact(self, fact_id: str, updated_fact: Dict[str, Any]) -> bool:
        """Update an existing fact."""
        facts = self.load_facts()
        for i, fact in enumerate(facts):
            if fact.get("fact_id") == fact_id:
                facts[i] = updated_fact
                self.save_facts(facts)
                return True
        return False
    
    def get_fact_by_id(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Get fact by ID."""
        facts = self.load_facts()
        for fact in facts:
            if fact.get("fact_id") == fact_id:
                return fact
        return None
    
    def clear_facts(self) -> None:
        """Clear all facts from storage."""
        save_json({"conversation_id": self.conversation_id, "facts": []}, self.facts_file)
    
    # ==================== Events Operations ====================
    
    def load_events(self) -> List[Dict[str, Any]]:
        """Load all events from storage."""
        data = load_json(self.events_file)
        return data.get("events", [])
    
    def save_events(self, events: List[Dict[str, Any]]) -> None:
        """Save events to storage."""
        save_json({"conversation_id": self.conversation_id, "events": events}, self.events_file)
    
    def add_event(self, event: Dict[str, Any]) -> None:
        """Add a new event to storage."""
        events = self.load_events()
        events.append(event)
        self.save_events(events)
    
    def update_event(self, event_id: str, updated_event: Dict[str, Any]) -> bool:
        """Update an existing event."""
        events = self.load_events()
        for i, event in enumerate(events):
            if event.get("event_id") == event_id:
                events[i] = updated_event
                self.save_events(events)
                return True
        return False
    
    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get event by ID."""
        events = self.load_events()
        for event in events:
            if event.get("event_id") == event_id:
                return event
        return None
    
    def clear_events(self) -> None:
        """Clear all events from storage."""
        save_json({"conversation_id": self.conversation_id, "events": []}, self.events_file)
    
    # ==================== Profiles Operations ====================
    
    def load_profiles(self) -> List[Dict[str, Any]]:
        """Load all profiles from storage."""
        data = load_json(self.profiles_file)
        return data.get("profiles", [])
    
    def save_profiles(self, profiles: List[Dict[str, Any]]) -> None:
        """Save profiles to storage."""
        save_json({"conversation_id": self.conversation_id, "profiles": profiles}, self.profiles_file)
    
    def add_profile(self, profile: Dict[str, Any]) -> None:
        """Add a new profile to storage."""
        profiles = self.load_profiles()
        profiles.append(profile)
        self.save_profiles(profiles)
    
    def update_profile(self, profile_id: str, updated_profile: Dict[str, Any]) -> bool:
        """Update an existing profile."""
        profiles = self.load_profiles()
        for i, profile in enumerate(profiles):
            if profile.get("profile_id") == profile_id:
                profiles[i] = updated_profile
                self.save_profiles(profiles)
                return True
        return False
    
    def get_profile_by_id(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get profile by ID."""
        profiles = self.load_profiles()
        for profile in profiles:
            if profile.get("profile_id") == profile_id:
                return profile
        return None
    
    def clear_profiles(self) -> None:
        """Clear all profiles from storage."""
        save_json({"conversation_id": self.conversation_id, "profiles": []}, self.profiles_file)

    # ==================== Entity Graph Operations ====================

    def load_entity_graph(self) -> Dict[str, Any]:
        """Load keyword fact graph index from storage."""
        if not os.path.exists(self.entity_graph_file):
            return {}
        return load_json(self.entity_graph_file)

    def save_entity_graph(self, graph_index: Dict[str, Any]) -> None:
        """Save keyword fact graph index to storage."""
        save_json(graph_index, self.entity_graph_file)

    def clear_entity_graph(self) -> None:
        """Clear keyword fact graph index."""
        if os.path.exists(self.entity_graph_file):
            os.remove(self.entity_graph_file)
    
    # ==================== Batch Operations ====================
    
    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all data (facts, events, profiles).
        
        Returns:
            {"facts": [...], "events": [...], "profiles": [...]}
        """
        return {
            "facts": self.load_facts(),
            "events": self.load_events(),
            "profiles": self.load_profiles()
        }
    
    def save_all(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Save all data (facts, events, profiles).
        
        Args:
            data: {"facts": [...], "events": [...], "profiles": [...]}
        """
        if "facts" in data:
            self.save_facts(data["facts"])
        if "events" in data:
            self.save_events(data["events"])
        if "profiles" in data:
            self.save_profiles(data["profiles"])
    
    def clear_all(self) -> None:
        """Clear all data."""
        self.clear_facts()
        self.clear_events()
        self.clear_profiles()
        self.clear_entity_graph()
