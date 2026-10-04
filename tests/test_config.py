import pytest
from src.config import Settings, settings

def test_settings_initialization():
    s = Settings()
    assert s.project_root.exists()
    assert s.default_llm_provider in ["groq", "nvidia", "openrouter", "ollama"]
    assert s.embedding_provider in ["huggingface", "ollama", "nvidia"]
    assert s.rerank_top_k == 4
    assert s.rerank_candidates_k == 15

def test_validate_provider_success():
    valid, msg = settings.validate_provider("groq")
    assert valid is True
    assert "Groq API key found" in msg

    valid, msg = settings.validate_provider("ollama")
    assert valid is True
    assert "Ollama configured" in msg

def test_validate_provider_unknown():
    valid, msg = settings.validate_provider("non_existent_provider")
    assert valid is False
    assert "Unknown provider" in msg
