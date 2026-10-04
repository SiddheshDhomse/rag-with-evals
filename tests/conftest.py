import os
import sys
import tempfile
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment variables
os.environ["GROQ_API_KEY"] = "mock_groq_test_credential"
os.environ["NVIDIA_API_KEY"] = "mock_nvidia_test_credential"
os.environ["OPEN_ROUTE_API_KEY"] = "mock_openrouter_test_credential"
os.environ["DEFAULT_LLM_PROVIDER"] = "groq"
os.environ["EMBEDDING_PROVIDER"] = "huggingface"

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)
