from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import json
import uuid


@dataclass
class HistoryEvent:
    event_id: str
    timestamp: str
    event_type: str
    summary: str
    payload: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "summary": self.summary,
            "payload": self.payload,
        }


class BrainHistory:
    """
    Persistent operational history for BrAIn.

    Stores:
    - ingests
    - commands
    - merge approvals/rejections/applications
    - future maintenance actions
    """

    def __init__(self, history_path: str | Path = "brain_history.json"):
        self.history_path = Path(history_path)
        self.events: List[Dict[str, object]] = []
        self._load()

    def _load(self) -> None:
        if not self.history_path.exists():
            self.events = []
            return

        with open(self.history_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.events = data
        else:
            self.events = []

    def save(self) -> None:
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2)

    def log(self, event_type: str, summary: str, payload: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        event = HistoryEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            summary=summary,
            payload=payload or {},
        ).to_dict()

        self.events.append(event)
        self.save()
        return event

    def recent(self, limit: int = 10) -> List[Dict[str, object]]:
        return self.events[-limit:]

    def by_type(self, event_type: str, limit: int = 20) -> List[Dict[str, object]]:
        matches = [event for event in self.events if event.get("event_type") == event_type]
        return matches[-limit:]

    def clear(self) -> Dict[str, object]:
        cleared_count = len(self.events)
        self.events = []
        self.save()
        return {
            "cleared_count": cleared_count,
            "history_path": str(self.history_path),
        }