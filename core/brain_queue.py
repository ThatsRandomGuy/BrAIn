from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class QueueItem:
    left: str
    right: str
    score: float
    reason: str
    status: str = "pending"  # pending | approved | rejected

    def to_dict(self) -> Dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "score": round(self.score, 3),
            "reason": self.reason,
            "status": self.status,
        }


class BrainQueue:
    """
    Stores merge suggestions for human review.

    Flow:
    - similarity engine proposes candidate merges
    - queue stores them if unseen
    - user can approve or reject later
    - approvals can be handed to BrainMerge
    """

    def __init__(self, queue_path: str | Path = "brain_merge_queue.json"):
        self.queue_path = Path(queue_path)
        self.items: List[Dict[str, object]] = []
        self._load()

    def _load(self) -> None:
        if not self.queue_path.exists():
            self.items = []
            return

        with open(self.queue_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.items = data
        else:
            self.items = []

    def save(self) -> None:
        with open(self.queue_path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, indent=2)

    def add_suggestions(self, suggestions: List[Dict[str, object]]) -> List[Dict[str, object]]:
        added: List[Dict[str, object]] = []

        for suggestion in suggestions:
            left = str(suggestion.get("left", "")).strip()
            right = str(suggestion.get("right", "")).strip()
            score = float(suggestion.get("score", 0.0))
            reason = str(suggestion.get("reason", "unknown")).strip()

            if not left or not right:
                continue

            if self._find_item(left, right) is not None:
                continue

            item = QueueItem(
                left=left,
                right=right,
                score=score,
                reason=reason,
                status="pending",
            ).to_dict()

            self.items.append(item)
            added.append(item)

        if added:
            self.save()

        return added

    def approve(self, left: str, right: str) -> Optional[Dict[str, object]]:
        item = self._find_item(left, right)
        if item is None:
            return None

        item["status"] = "approved"
        self.save()
        return item

    def reject(self, left: str, right: str) -> Optional[Dict[str, object]]:
        item = self._find_item(left, right)
        if item is None:
            return None

        item["status"] = "rejected"
        self.save()
        return item

    def get_pending(self) -> List[Dict[str, object]]:
        return [item for item in self.items if item.get("status") == "pending"]

    def get_approved(self) -> List[Dict[str, object]]:
        return [item for item in self.items if item.get("status") == "approved"]

    def get_rejected(self) -> List[Dict[str, object]]:
        return [item for item in self.items if item.get("status") == "rejected"]

    def clear_approved(self) -> List[Dict[str, object]]:
        approved = self.get_approved()
        if not approved:
            return []

        self.items = [item for item in self.items if item.get("status") != "approved"]
        self.save()
        return approved

    def _find_item(self, left: str, right: str) -> Optional[Dict[str, object]]:
        for item in self.items:
            existing_left = str(item.get("left", "")).strip()
            existing_right = str(item.get("right", "")).strip()

            if (existing_left == left and existing_right == right) or (
                existing_left == right and existing_right == left
            ):
                return item

        return None