"""
Configuration module for Sarthi AI Chatbot
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "Sarthi AI - Rajasthan Procurement Assistant"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DOCUMENTS_DIR: Path = BASE_DIR / "documents"
    DATA_DIR: Path = BASE_DIR / "data"
    QDRANT_PATH: Path = DATA_DIR / "qdrant_db"
    GRAPH_PATH: Path = DATA_DIR / "graph_db"
    MEMORY_PATH: Path = DATA_DIR / "memory_store"
    CACHE_PATH: Path = DATA_DIR / "cache"
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # LLM Models
    LLM_MODEL: str = "deepseek-r1:8b"  # or "gpt-oss:latest"
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    
    # Qdrant Configuration
    QDRANT_COLLECTION_NAME: str = "rajasthan_procurement_docs"
    QDRANT_VECTOR_SIZE: int = 768
    QDRANT_DISTANCE: str = "Cosine"
    
    # Chunking Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CHUNKS_PER_DOC: int = 500
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = 8
    RERANK_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # LLM Generation
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    
    # Memory Configuration
    MAX_CONVERSATION_HISTORY: int = 10
    MEMORY_WINDOW_SIZE: int = 5
    
    # Processing
    BATCH_SIZE: int = 16
    MAX_WORKERS: int = 4
    
    # Cache
    CACHE_TTL: int = 86400  # 24 hours
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories
        for path in [self.DATA_DIR, self.QDRANT_PATH, self.GRAPH_PATH, 
                     self.MEMORY_PATH, self.CACHE_PATH, self.DOCUMENTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()