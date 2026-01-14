"""
Configuration for Lexio Agent.
"""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Local PostgreSQL (complete E-Sbírka data) - matches MCP agent database
    DATABASE_URL: str = "postgresql://esbirka:esbirka_dev_2024@localhost:5432/esbirka_laws"
    
    # OpenAI (for embeddings only)
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # OpenRouter (for ALL AI models)
    OPENROUTER_API_KEY: str = ""
    AGENT_MODEL: str = "google/gemini-3-flash-preview"  # Gemini 3 Flash
    
    # E-Sbírka REST API (for fallback when data missing from DB)
    ESBIRKA_API_KEY: str = ""
    ESBIRKA_API_BASE: str = "https://api.e-sbirka.cz"
    ESBIRKA_API_TIMEOUT: int = 60
    ESBIRKA_ENABLE_FALLBACK: bool = True  # Toggle API fallback on/off
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    
    class Config:
        env_file = ".env"


settings = Settings()
