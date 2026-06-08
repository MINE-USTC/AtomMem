# src/utils.py
# Utility Functions

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
import config


def parse_session_datetime(session_datetime: str) -> str:
    """
    Parse session datetime string to YYYY-MM-DD format.
    
    Example: "1:56 pm on 8 May, 2023" -> "2023-05-08"
    """
    try:
        # Extract date components
        pattern = r'on (\d+) (\w+), (\d{4})'
        match = re.search(pattern, session_datetime)
        if match:
            day, month_name, year = match.groups()
            # Convert month name to number
            month_map = {
                'January': '01', 'February': '02', 'March': '03', 'April': '04',
                'May': '05', 'June': '06', 'July': '07', 'August': '08',
                'September': '09', 'October': '10', 'November': '11', 'December': '12'
            }
            month = month_map.get(month_name, '01')
            day = day.zfill(2)
            return f"{year}-{month}-{day}"
    except Exception:
        pass
    return datetime.now().strftime(config.TIME_FORMAT)


def load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_directory(directory: str) -> None:
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)


def calculate_jaccard_similarity(list1: List[str], list2: List[str]) -> float:
    """
    Calculate Jaccard similarity between two lists (exact string matching).
    This is the original version with case-insensitive exact matching.
    """
    if not list1 or not list2:
        return 0.0
    set1 = set([item.lower() for item in list1])
    set2 = set([item.lower() for item in list2])
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def calculate_semantic_jaccard_similarity(
    list1: List[str], 
    list2: List[str],
    embedding_model=None,
    similarity_threshold: float = 0.9
) -> float:
    """
    Calculate Jaccard similarity using semantic matching (embedding-based).
    
    Two keywords are considered matching if their embedding similarity > threshold.
    
    Args:
        list1: First list of keywords
        list2: Second list of keywords
        embedding_model: EmbeddingModel instance (if None, will create one)
        similarity_threshold: Threshold for considering two keywords as matching (default: 0.9)
        
    Returns:
        Semantic Jaccard similarity score
    """
    if not list1 or not list2:
        return 0.0
    
    # Import here to avoid circular dependency
    if embedding_model is None:
        from src.embedding import EmbeddingModel
        embedding_model = EmbeddingModel()
    
    # Get embeddings for all keywords
    embeddings1 = [embedding_model.encode(kw) for kw in list1]
    embeddings2 = [embedding_model.encode(kw) for kw in list2]
    
    # Build matching matrix: which keywords from list1 match which from list2
    matched_from_list1 = set()
    matched_from_list2 = set()
    
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            sim = calculate_cosine_similarity(emb1, emb2)
            if sim >= similarity_threshold:
                matched_from_list1.add(i)
                matched_from_list2.add(j)
    
    # Calculate semantic Jaccard
    # Intersection: keywords that have a match
    intersection_size = len(matched_from_list1)
    
    # Union: all unique keywords (considering semantic matches)
    # Keywords from list1 that have no match + keywords from list2 that have no match + matched pairs
    unmatched_from_list1 = len(list1) - len(matched_from_list1)
    unmatched_from_list2 = len(list2) - len(matched_from_list2)
    matched_pairs = len(matched_from_list1)  # or len(matched_from_list2), should be similar
    
    union_size = unmatched_from_list1 + unmatched_from_list2 + matched_pairs
    
    if union_size == 0:
        return 0.0
    
    return intersection_size / union_size


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    if not vec1 or not vec2:
        return 0.0
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def generate_fact_id(existing_facts: List[Dict]) -> str:
    """Generate unique fact ID."""
    if not existing_facts:
        return "F1"
    # Extract existing IDs and find the maximum number
    max_num = 0
    for fact in existing_facts:
        fact_id = fact.get('fact_id', 'F0')
        if fact_id.startswith('F'):
            try:
                num = int(fact_id[1:])
                max_num = max(max_num, num)
            except ValueError:
                pass
    return f"F{max_num + 1}"


def generate_event_id(existing_events: List[Dict]) -> str:
    """Generate unique event ID."""
    if not existing_events:
        return "E1"
    # Extract existing IDs and find the maximum number
    max_num = 0
    for event in existing_events:
        event_id = event.get('event_id', 'E0')
        if event_id.startswith('E'):
            try:
                num = int(event_id[1:])
                max_num = max(max_num, num)
            except ValueError:
                pass
    return f"E{max_num + 1}"


def generate_profile_id(existing_profiles: List[Dict]) -> str:
    """Generate unique profile ID."""
    if not existing_profiles:
        return "P1"
    # Extract existing IDs and find the maximum number
    max_num = 0
    for profile in existing_profiles:
        profile_id = profile.get('profile_id', 'P0')
        if profile_id.startswith('P'):
            try:
                num = int(profile_id[1:])
                max_num = max(max_num, num)
            except ValueError:
                pass
    return f"P{max_num + 1}"


# Alias for backward compatibility
cosine_similarity = calculate_cosine_similarity
jaccard_similarity = calculate_jaccard_similarity


def extract_people_intersection(people1: List[str], people2: List[str]) -> bool:
    """Check if two people lists have intersection."""
    if not people1 or not people2:
        return False
    set1 = set([p.lower() for p in people1])
    set2 = set([p.lower() for p in people2])
    return bool(set1 & set2)

