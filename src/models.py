import logging
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.config import settings

logger = logging.getLogger(__name__)


def get_chat_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    streaming: bool = True,
    max_tokens: Optional[int] = None
) -> BaseChatModel:
    """
    Factory function to instantiate an LLM based on provider:
    - groq: Ultra-fast inference via Groq Cloud
    - openrouter: Multi-model router with free models
    - nvidia: Cloud inference via NVIDIA NIM
    - ollama: Local open-source models via Ollama
    """
    provider = (provider or settings.default_llm_provider).lower()
    token_limit = max_tokens or 450

    if provider == "groq":
        valid, msg = settings.validate_provider("groq")
        if not valid:
            raise ValueError(f"Groq configuration error: {msg}")

        chosen_model = model_name or settings.groq_default_model
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=chosen_model,
                groq_api_key=settings.groq_api_key,
                temperature=temperature,
                max_tokens=token_limit,
                streaming=streaming
            )
        except Exception:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=chosen_model,
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=temperature,
                max_tokens=token_limit,
                streaming=streaming
            )

    elif provider == "openrouter":
        valid, msg = settings.validate_provider("openrouter")
        if not valid:
            raise ValueError(f"OpenRouter configuration error: {msg}")

        chosen_model = model_name or settings.openrouter_default_model
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=chosen_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            streaming=streaming,
            default_headers={"HTTP-Referer": "https://github.com", "X-Title": "RAG Studio"}
        )

    elif provider == "nvidia":
        valid, msg = settings.validate_provider("nvidia")
        if not valid:
            raise ValueError(f"NVIDIA configuration error: {msg}")

        chosen_model = model_name or settings.nvidia_default_model
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=chosen_model,
            api_key=settings.nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=temperature,
            streaming=streaming
        )

    elif provider == "ollama":
        chosen_model = model_name or settings.ollama_default_model
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=chosen_model,
                base_url=settings.ollama_base_url,
                temperature=temperature,
                streaming=streaming
            )
        except Exception:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=chosen_model,
                base_url=settings.ollama_base_url,
                temperature=temperature
            )

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Choose 'groq', 'openrouter', 'nvidia', or 'ollama'.")


def get_embedding_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None
) -> Embeddings:
    """
    Factory function to instantiate embeddings:
    - huggingface: Local sentence-transformers (Default, zero API costs, runs on CPU/GPU)
    - ollama: Local Ollama embeddings (e.g., nomic-embed-text)
    - nvidia: NVIDIA NIM embeddings
    """
    provider = (provider or settings.embedding_provider).lower()

    if provider == "huggingface":
        chosen_model = model_name or settings.embedding_model_name
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=chosen_model)
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=chosen_model)

    elif provider == "ollama":
        chosen_model = model_name or "nomic-embed-text"
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model=chosen_model,
                base_url=settings.ollama_base_url
            )
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(
                model=chosen_model,
                base_url=settings.ollama_base_url
            )

    elif provider == "nvidia":
        valid, msg = settings.validate_provider("nvidia")
        if not valid:
            raise ValueError(f"NVIDIA configuration error: {msg}")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name or "nvidia/llama-3.2-nv-embedqa-1b-v1",
            openai_api_key=settings.nvidia_api_key,
            openai_api_base="https://integrate.api.nvidia.com/v1"
        )

    else:
        raise ValueError(f"Unsupported embedding provider: '{provider}'. Choose 'huggingface', 'ollama', or 'nvidia'.")
