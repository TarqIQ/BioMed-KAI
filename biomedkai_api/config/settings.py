import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import yaml

# Load environment variables
tt = load_dotenv()
print("load_dotenv", tt)
# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

print("MODEL_DIR", MODEL_DIR)
print("CONFIG_DIR", CONFIG_DIR)
print("LOG_DIR", LOG_DIR)
print("BASE_DIR", BASE_DIR)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGCHAIN_API_KEY") is None:
    os.environ["LANGCHAIN_API_KEY"] = "***"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
if os.getenv("LANGSMITH_API_KEY") is None:
    os.environ["LANGSMITH_API_KEY"] = "***"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
# Disable LangSmith if no API key
if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = Field(default="MedicalAIAgent", env="APP_NAME")
    app_env: str = Field(default="development", env="APP_ENV")
    app_debug: bool = Field(default=False, env="APP_DEBUG")
    app_port: int = Field(default=8080, env="APP_PORT")
    websocket_port: int = Field(default=8081, env="WEBSOCKET_PORT")
    
    # Model Configuration
    model_path: str = Field(env="MODEL_PATH")
    context_size: int = Field(default=65536, env="CONTEXT_SIZE")
    gpu_layers: int = Field(default=35, env="GPU_LAYERS")
    threads: int = Field(default=8, env="THREADS")
    batch_size: int = Field(default=512, env="BATCH_SIZE")
    
    # Embedding Model
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    
    # Neo4j
    neo4j_uri: str = Field(env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="medical", env="NEO4J_DATABASE")
    
    # PostgreSQL
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(env="POSTGRES_DB")
    postgres_user: str = Field(env="POSTGRES_USER")
    postgres_password: str = Field(env="POSTGRES_PASSWORD")
    
    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # API Keys
    google_search_api_key: Optional[str] = Field(default=None, env="GOOGLE_SEARCH_API_KEY")
    google_search_engine_id: Optional[str] = Field(default=None, env="GOOGLE_SEARCH_ENGINE_ID")
    pubmed_api_key: Optional[str] = Field(default=None, env="PUBMED_API_KEY")
    clinicaltrials_api_key: Optional[str] = Field(default=None, env="CLINICALTRIALS_API_KEY")
    drugbank_api_key: Optional[str] = Field(default=None, env="DRUGBANK_API_KEY")
    
    # LangSmith
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langchain_tracing_v2: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="medical-ai-agent", env="LANGCHAIN_PROJECT")
    
    # Security
    jwt_secret_key: str = Field(env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    encryption_key: str = Field(env="ENCRYPTION_KEY")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")
    
    # Monitoring
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: str = Field(default="logs/medical_ai.log", env="LOG_FILE")
    
    # Feature Flags
    enable_web_search: bool = Field(default=True, env="ENABLE_WEB_SEARCH")
    enable_emergency_triage: bool = Field(default=True, env="ENABLE_EMERGENCY_TRIAGE")
    enable_human_review: bool = Field(default=True, env="ENABLE_HUMAN_REVIEW")
    confidence_threshold: float = Field(default=0.7, env="CONFIDENCE_THRESHOLD")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def load_agent_config() -> Dict[str, Any]:
    """Load agent configuration from YAML"""
    config_path = CONFIG_DIR / "agents_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompts() -> Dict[str, Dict[str, str]]:
    """Load agent prompts from YAML files"""
    prompts = {}
    prompt_dir = CONFIG_DIR / "prompts"
    
    for prompt_file in prompt_dir.glob("*.yaml"):
        agent_name = prompt_file.stem.replace("_prompts", "")
        with open(prompt_file, 'r') as f:
            prompts[agent_name] = yaml.safe_load(f)
    
    return prompts


# Singleton instance
settings = Settings()
agent_config = load_agent_config()
prompts = load_prompts()