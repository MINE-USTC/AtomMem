# src/embedding.py
# Embedding Interface

import os
from typing import List

# AtomMem only needs sentence-transformers inference. Disabling TensorFlow/Flax
# avoids optional backend imports that can fail on lightweight CPU machines.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from sentence_transformers import SentenceTransformer
import config


class EmbeddingModel:
    """Embedding model interface using sentence-transformers."""
    
    def __init__(self):
        """Initialize embedding model."""
        self.dimension = config.EMBEDDING_DIMENSION
        self.model = None if config.DRY_RUN_EMBEDDING else SentenceTransformer(config.EMBEDDING_MODEL)
    
    def encode(self, text: str) -> List[float]:
        """
        Encode text to embedding vector.
        
        Args:
            text: Input text
            
        Returns:
            List of float values representing the embedding
        """
        if self.model is None:
            return [0.0] * self.dimension
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts to embedding vectors.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            return [[0.0] * self.dimension for _ in texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
