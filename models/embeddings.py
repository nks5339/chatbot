"""
Embeddings module using Ollama
"""
import ollama
import numpy as np
from typing import List, Union
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingModel:
    """Embedding model using Ollama"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self._verify_model()
        logger.info(f"Embedding model initialized: {self.model_name}")
    
    def _verify_model(self):
        """Verify that the embedding model is available"""
        try:
            # Test embedding
            self.client.embeddings(model=self.model_name, prompt="test")
            logger.info(f"Embedding model '{self.model_name}' verified successfully")
        except Exception as e:
            logger.error(f"Failed to verify embedding model: {e}")
            logger.info(f"Attempting to pull model: {self.model_name}")
            try:
                self.client.pull(self.model_name)
                logger.info(f"Model '{self.model_name}' pulled successfully")
            except Exception as pull_error:
                logger.error(f"Failed to pull model: {pull_error}")
                raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        embeddings = []
        for text in texts:
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                embeddings.append([0.0] * settings.QDRANT_VECTOR_SIZE)
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query (alias for consistency)"""
        return self.embed_text(query)

# Global embedding model instance
embedding_model = EmbeddingModel()