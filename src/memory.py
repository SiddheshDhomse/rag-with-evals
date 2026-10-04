import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.config import settings


class ChatHistoryManager:
    """
    Manages persistent multi-session chat histories stored as JSON files.
    Allows saving queries, assistant responses, and retrieved source citations.
    """

    def __init__(self, history_dir: Optional[Path] = None):
        self.history_dir = history_dir or settings.history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")).strip()
        if not safe_id:
            safe_id = "default"
        return self.history_dir / f"{safe_id}.json"

    def list_sessions(self, include_internal: bool = False) -> List[str]:
        """Lists all existing conversation session IDs, filtering out internal eval runs by default."""
        files = list(self.history_dir.glob("*.json"))
        sessions = [f.stem for f in files]
        if not include_internal:
            sessions = [
                s for s in sessions
                if not s.startswith("eval_") and not s.startswith("test_") and not s.startswith("ragas_")
            ]
        if not sessions:
            sessions = ["default"]
        elif "default" not in sessions:
            sessions.insert(0, "default")
        return sorted(list(set(sessions)))

    def get_messages(self, session_id: str = "default") -> List[Dict[str, Any]]:
        """Loads all message dicts for a given session."""
        path = self._get_session_path(session_id)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("messages", [])
        except Exception:
            return []

    def get_langchain_messages(self, session_id: str = "default", limit: int = 10) -> List[BaseMessage]:
        """Returns the recent conversation turns as LangChain BaseMessage objects."""
        raw_msgs = self.get_messages(session_id)
        if limit and len(raw_msgs) > limit:
            raw_msgs = raw_msgs[-limit:]

        lc_messages: List[BaseMessage] = []
        for m in raw_msgs:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        return lc_messages

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None
    ):
        """Appends a new turn to the session file."""
        path = self._get_session_path(session_id)
        messages = self.get_messages(session_id)

        msg_entry = {
            "role": role,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sources": sources or [],
            "candidates": candidates or []
        }
        messages.append(msg_entry)

        with open(path, "w", encoding="utf-8") as f:
            json.dump({"session_id": session_id, "messages": messages}, f, indent=2, ensure_ascii=False)

    def clear_session(self, session_id: str):
        """Clears all messages from a specific session or initializes an empty one on disk."""
        path = self._get_session_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"session_id": session_id, "messages": []}, f, indent=2)

    def create_new_session(self) -> str:
        """Generates a unique chat session ID and initializes its file on disk."""
        existing = self.list_sessions(include_internal=False)
        idx = 1
        while f"chat_{idx}" in existing:
            idx += 1
        new_id = f"chat_{idx}"
        self.clear_session(new_id)
        return new_id

    def delete_session(self, session_id: str):
        """Deletes a session file permanently."""
        path = self._get_session_path(session_id)
        if path.exists():
            path.unlink()
