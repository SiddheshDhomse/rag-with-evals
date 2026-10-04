import pytest
from src.memory import ChatHistoryManager
from langchain_core.messages import HumanMessage, AIMessage

def test_chat_history_lifecycle(temp_dir):
    manager = ChatHistoryManager(history_dir=temp_dir)
    session_id = "test_session_abc"

    # Initially empty
    assert manager.get_messages(session_id) == []

    # Add user message
    manager.add_message(session_id, role="user", content="Hello, what is RAG?")
    msgs = manager.get_messages(session_id)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello, what is RAG?"

    # Add assistant response with mock sources
    sources = [{"source": "test_doc.pdf", "page": 1, "score": 0.95}]
    manager.add_message(session_id, role="assistant", content="RAG stands for Retrieval-Augmented Generation.", sources=sources)
    msgs = manager.get_messages(session_id)
    assert len(msgs) == 2
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["sources"] == sources

    # Test LangChain message conversion
    lc_msgs = manager.get_langchain_messages(session_id)
    assert len(lc_msgs) == 2
    assert isinstance(lc_msgs[0], HumanMessage)
    assert isinstance(lc_msgs[1], AIMessage)

    # Test clear session
    manager.clear_session(session_id)
    assert manager.get_messages(session_id) == []

    # Test delete session
    manager.delete_session(session_id)
    assert not (temp_dir / f"{session_id}.json").exists()

def test_session_listing_and_filtering(temp_dir):
    manager = ChatHistoryManager(history_dir=temp_dir)
    manager.add_message("chat_1", role="user", content="Hi")
    manager.add_message("eval_01", role="user", content="Eval question")
    manager.add_message("test_internal", role="user", content="Test")

    user_sessions = manager.list_sessions(include_internal=False)
    assert "chat_1" in user_sessions
    assert "eval_01" not in user_sessions
    assert "test_internal" not in user_sessions

    all_sessions = manager.list_sessions(include_internal=True)
    assert "chat_1" in all_sessions
    assert "eval_01" in all_sessions
    assert "test_internal" in all_sessions
