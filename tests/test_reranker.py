import pytest
import numpy as np
from src.reranker import sigmoid, CrossEncoderReranker
from langchain_core.documents import Document

def test_sigmoid_math():
    assert sigmoid(0.0) == pytest.approx(0.5, abs=1e-5)
    assert sigmoid(10.0) > 0.999
    assert sigmoid(-10.0) < 0.001
    # Check monotonicity
    assert sigmoid(2.0) > sigmoid(1.0)
    assert sigmoid(-1.0) > sigmoid(-2.0)

def test_reranker_empty_input():
    reranker = CrossEncoderReranker()
    # reranking empty docs should safely return empty lists without model load
    top_docs, audit = reranker.rerank(query="test", docs_with_scores=[], top_k=4)
    assert top_docs == []
    assert audit == []
