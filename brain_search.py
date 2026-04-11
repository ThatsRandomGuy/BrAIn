from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class SearchHit:
    source: str
    title: str
    score: int
    path: str
    snippet: str
    payload: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "title": self.title,
            "score": self.score,
            "path": self.path,
            "snippet": self.snippet,
            "payload": self.payload,
        }


class BrainSearch:
    """
    Lightweight search over:
    - operational history
    - saved artifact JSON files in brain_output

    This is intentionally simple and deterministic.
    """

    def __init__(
        self,
        history_path: str | Path = "brain_history.json",
        output_dir: str | Path = "brain_output",
    ):
        self.history_path = Path(history_path)
        self.output_dir = Path(output_dir)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, object]]:
        q = query.strip().lower()
        if not q:
            return []

        hits: List[SearchHit] = []
        hits.extend(self._search_history(q))
        hits.extend(self._search_artifacts(q))

        hits.sort(key=lambda h: h.score, reverse=True)
        return [hit.to_dict() for hit in hits[:limit]]

    def _search_history(self, query: str) -> List[SearchHit]:
        if not self.history_path.exists():
            return []

        with open(self.history_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        hits: List[SearchHit] = []

        for event in data:
            if not isinstance(event, dict):
                continue

            searchable = self._flatten_json(event).lower()
            score = self._score_text(searchable, query)
            if score <= 0:
                continue

            event_type = str(event.get("event_type", "unknown"))
            summary = str(event.get("summary", ""))
            timestamp = str(event.get("timestamp", ""))

            hits.append(
                SearchHit(
                    source="history",
                    title=f"{event_type} | {timestamp}",
                    score=score,
                    path=str(self.history_path),
                    snippet=summary[:220],
                    payload=event,
                )
            )

        return hits

    def _search_artifacts(self, query: str) -> List[SearchHit]:
        if not self.output_dir.exists():
            return []

        hits: List[SearchHit] = []

        for path in sorted(self.output_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            searchable = self._flatten_json(data).lower()
            score = self._score_text(searchable, query)
            if score <= 0:
                continue

            title = path.name
            snippet = self._best_snippet(searchable, query)

            hits.append(
                SearchHit(
                    source="artifact",
                    title=title,
                    score=score,
                    path=str(path),
                    snippet=snippet,
                    payload={"file": path.name},
                )
            )

        return hits

    def recent_artifacts(self, limit: int = 10) -> List[Dict[str, object]]:
        if not self.output_dir.exists():
            return []

        files = sorted(
            self.output_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        out: List[Dict[str, object]] = []
        for path in files[:limit]:
            out.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "modified_ts": path.stat().st_mtime,
                }
            )
        return out

    def _score_text(self, text: str, query: str) -> int:
        score = 0

        if query in text:
            score += 5

        query_terms = [term for term in query.split() if term]
        for term in query_terms:
            count = text.count(term)
            score += count

        return score

    def _best_snippet(self, text: str, query: str, window: int = 220) -> str:
        idx = text.find(query)
        if idx == -1:
            return text[:window]

        start = max(0, idx - window // 3)
        end = min(len(text), idx + window)
        return text[start:end].strip()

    def _flatten_json(self, obj: object) -> str:
        parts: List[str] = []

        def walk(value: object) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    parts.append(str(k))
                    walk(v)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            else:
                parts.append(str(value))

        walk(obj)
        return " ".join(parts)