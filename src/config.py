import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Automatically load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    # Project paths
    project_root: Path = PROJECT_ROOT
    storage_dir: Path = PROJECT_ROOT / "storage"
    chroma_dir: Path = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "storage/chroma")
    history_dir: Path = PROJECT_ROOT / os.getenv("CHAT_HISTORY_DIR", "storage/history")
    data_dir: Path = PROJECT_ROOT / "data"

    # API Keys & URLs
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    nvidia_api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", "").strip())
    openrouter_api_key: str = field(default_factory=lambda: (os.getenv("OPEN_ROUTE_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")).strip())
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip())

    # Provider & Model Defaults
    default_llm_provider: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_PROVIDER", "groq").lower())
    groq_default_model: str = field(default_factory=lambda: os.getenv("GROQ_DEFAULT_MODEL", "qwen/qwen3.8-27b"))
    nvidia_default_model: str = field(default_factory=lambda: os.getenv("NVIDIA_DEFAULT_MODEL", "meta/llama-3.2-11b-vision-instruct"))
    openrouter_default_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_DEFAULT_MODEL", "openrouter/free"))
    ollama_default_model: str = field(default_factory=lambda: os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.1:8b"))

    # Embedding Settings
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "huggingface").lower())
    embedding_model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))

    # Reranker Settings (Phase 3)
    reranker_model_name: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_TOP_K", "4")))
    rerank_candidates_k: int = field(default_factory=lambda: int(os.getenv("RERANK_CANDIDATES_K", "15")))

    # Query Transformation & Adaptive Routing Settings (Phase 4)
    query_transform_mode: str = field(default_factory=lambda: os.getenv("QUERY_TRANSFORM_MODE", "none").lower())
    multi_query_count: int = field(default_factory=lambda: int(os.getenv("MULTI_QUERY_COUNT", "3")))
    hyde_max_tokens: int = field(default_factory=lambda: int(os.getenv("HYDE_MAX_TOKENS", "180")))

    # Vector store defaults
    collection_name: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "rag_knowledge_base"))

    # Tested and verified working model catalogs
    available_models: Dict[str, List[str]] = field(default_factory=lambda: {
        "groq": [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ],
        "openrouter": [
            "openrouter/free",
            "google/gemma-4-31b-it:free",
            "google/gemma-4-26b-a4b-it:free",
            "nvidia/nemotron-3.5-lightning:free",
            "poolside/laguna-s-2.1:free"
        ],
        "nvidia": [
            "meta/llama-3.2-11b-vision-instruct",
            "meta/codellama-70b"
        ],
        "ollama": [
            "llama3.1:8b",
            "qwen3:14b",
            "llama3:latest"
        ]
    })

    def __post_init__(self):
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def validate_provider(self, provider: str) -> Tuple[bool, str]:
        """Validates if credentials for the chosen provider are set."""
        p = provider.lower()
        if p == "groq":
            if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
                return False, "GROQ_API_KEY is missing or not set in .env"
            return True, "Groq API key found"
        elif p == "openrouter":
            if not self.openrouter_api_key or self.openrouter_api_key == "your_openrouter_api_key_here":
                return False, "OPEN_ROUTE_API_KEY is missing or not set in .env"
            return True, "OpenRouter API key found"
        elif p == "nvidia":
            if not self.nvidia_api_key or self.nvidia_api_key == "your_nvidia_api_key_here":
                return False, "NVIDIA_API_KEY is missing or not set in .env"
            return True, "NVIDIA API key found"
        elif p == "ollama":
            return True, f"Ollama configured at {self.ollama_base_url}"
        else:
            return False, f"Unknown provider: {provider}"


# Global settings singleton
settings = Settings()
