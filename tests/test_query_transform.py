import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from src.query_transform import QueryTransformer

def test_heuristic_chitchat_routing():
    mock_llm = FakeListChatModel(responses=['{"route": "FACT_LOOKUP", "reasoning": "test"}'])
    transformer = QueryTransformer(llm=mock_llm)

    greetings = ["hi", "hello", "hey", "good morning", "thank you", "thanks"]
    for q in greetings:
        route_result = transformer.route_query(q)
        assert route_result["route"] == "DIRECT"
        assert route_result["strategy"] == "direct"
        assert route_result["confidence"] == 1.0

def test_llm_routing_fallback():
    mock_llm = FakeListChatModel(responses=['{"route": "MULTI_HOP", "reasoning": "Comparative analysis"}'])
    transformer = QueryTransformer(llm=mock_llm)

    complex_q = "Compare private vs state-owned emitters in the Americas"
    route_result = transformer.route_query(complex_q)
    assert route_result["route"] == "MULTI_HOP"
    assert route_result["strategy"] == "multi_query"
