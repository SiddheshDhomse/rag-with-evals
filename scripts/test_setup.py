"""
Health Check & Setup Verification Script.
Checks:
1. .env configuration and API keys (Groq, OpenRouter, NVIDIA, Ollama)
2. Embedding model initialization
3. ChromaDB storage and persistence
4. LLM connectivity across all active providers
Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.models import get_chat_llm, get_embedding_model
from src.vectorstore import VectorStoreManager


def test_environment():
    print("=" * 60)
    print("1. ENVIRONMENT & KEYS CHECK")
    print("=" * 60)
    for p in ["groq", "openrouter", "nvidia", "ollama"]:
        valid, msg = settings.validate_provider(p)
        status = "[OK]" if valid else "[MISSING]"
        print(f" - {p.upper():<11}: {status} {msg}")


def test_embeddings_and_chroma():
    print("\n" + "=" * 60)
    print("2. EMBEDDINGS & CHROMADB CHECK")
    print("=" * 60)
    try:
        print(f"Loading embedding model: {settings.embedding_model_name}...")
        embed_model = get_embedding_model()
        test_vec = embed_model.embed_query("Testing RAG vector database connectivity")
        print(f" [OK] Embedding generated successfully (Dimension: {len(test_vec)})")

        vsm = VectorStoreManager(collection_name="test_healthcheck", embedding_model=embed_model)
        vsm.add_texts(["Antigravity is an advanced agentic AI coding assistant."], metadatas=[{"source": "test_doc"}])
        res = vsm.similarity_search_with_score("What is Antigravity?", k=1)
        print(f" [OK] ChromaDB test read/write passed (Retrieved: '{res[0][0].page_content[:30]}...', Score: {res[0][1]:.4f})")
        vsm.clear_collection()
    except Exception as e:
        print(f" [FAIL] Embedding/Chroma check error: {e}")


def test_llms():
    print("\n" + "=" * 60)
    print("3. LLM PROVIDERS CONNECTIVITY TEST")
    print("=" * 60)
    for provider in ["groq", "openrouter", "nvidia", "ollama"]:
        valid, msg = settings.validate_provider(provider)
        if not valid:
            print(f" [SKIP] {provider.upper()}: {msg}")
            continue

        try:
            llm = get_chat_llm(provider=provider)
            resp = llm.invoke("What is 2+2? Reply only with the number.")
            ans = resp.content.strip() if hasattr(resp, "content") and resp.content else "OK"
            print(f" [OK] {provider.upper():<10} -> Response: '{ans}'")
        except Exception as e:
            print(f" [FAIL] {provider.upper():<10} -> Error: {e}")


if __name__ == "__main__":
    test_environment()
    test_embeddings_and_chroma()
    test_llms()
    print("\n" + "=" * 60)
    print("Health check finished!")
    print("=" * 60)
