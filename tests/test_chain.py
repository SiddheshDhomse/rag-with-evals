import pytest
from src.chain import sanitize_reformulated_query

def test_sanitize_clean_query():
    original = "What is the global impact of the ruling?"
    rephrased = ' "What is the global impact of the 2022 Supreme Court abortion ruling?" '
    cleaned = sanitize_reformulated_query(rephrased, original)
    assert cleaned == "What is the global impact of the 2022 Supreme Court abortion ruling?"

def test_sanitize_strips_prefixes():
    original = "What is the global impact?"
    rephrased = "Standalone Question: What is the global impact of the abortion ruling?"
    cleaned = sanitize_reformulated_query(rephrased, original)
    assert cleaned == "What is the global impact of the abortion ruling?"

def test_sanitize_refusal_fallback():
    original = "What did the court say?"
    # If the LLM generates a refusal instead of a standalone query
    refusal = "I cannot find sufficient information in the provided documents to answer."
    cleaned = sanitize_reformulated_query(refusal, original)
    # Should safely fallback to original query
    assert cleaned == original

def test_sanitize_multiline_fallback():
    original = "What is the impact?"
    multiline = "Line 1\nLine 2\nLine 3\nLine 4"
    cleaned = sanitize_reformulated_query(multiline, original)
    assert cleaned == original
