from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json


@dataclass
class SimilaritySuggestion:
    left: str
    right: str
    score: float
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "score": round(self.score, 3),
            "reason": self.reason,
        }


class BrainSimilarity:
    """
    Tracks primitive usage contexts and proposes possible merges.

    This does NOT auto-merge.
    It only suggests:
    - these two primitives seem to occur in similar contexts
    - maybe they are aliases or near-duplicates
    """

    MIN_SHARED_CONTEXT = 2
    MIN_SIMILARITY_SCORE = 0.55

    def __init__(self, similarity_path: str | Path = "brain_similarity.json"):
        self.similarity_path = Path(similarity_path)
        self.contexts: Dict[str, Set[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.similarity_path.exists():
            self.contexts = {}
            return

        with open(self.similarity_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded: Dict[str, Set[str]] = {}
        if isinstance(data, dict):
            for primitive, context_list in data.items():
                if isinstance(context_list, list):
                    loaded[str(primitive)] = set(str(x) for x in context_list)
        self.contexts = loaded

    def save(self) -> None:
        serializable = {
            primitive: sorted(list(contexts))
            for primitive, contexts in sorted(self.contexts.items())
        }
        with open(self.similarity_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def observe_memory(self, parsed_memory: Dict[str, object]) -> None:
        primitives = self._extract_primitives(parsed_memory)

        for primitive in primitives:
            self.contexts.setdefault(primitive, set())
            for other in primitives:
                if other != primitive:
                    self.contexts[primitive].add(other)

        self.save()

    def suggest_merges(self) -> List[Dict[str, object]]:
        primitives = sorted(self.contexts.keys())
        suggestions: List[SimilaritySuggestion] = []

        for i in range(len(primitives)):
            for j in range(i + 1, len(primitives)):
                left = primitives[i]
                right = primitives[j]

                if not self._same_type(left, right):
                    continue

                left_ctx = self.contexts.get(left, set())
                right_ctx = self.contexts.get(right, set())

                if not left_ctx or not right_ctx:
                    continue

                shared = left_ctx.intersection(right_ctx)
                union = left_ctx.union(right_ctx)

                if len(shared) < self.MIN_SHARED_CONTEXT or not union:
                    continue

                score = len(shared) / len(union)

                if score >= self.MIN_SIMILARITY_SCORE:
                    suggestions.append(
                        SimilaritySuggestion(
                            left=left,
                            right=right,
                            score=score,
                            reason="context_overlap",
                        )
                    )

        suggestions.sort(key=lambda s: s.score, reverse=True)
        return [s.to_dict() for s in suggestions]

    def _extract_primitives(self, parsed_memory: Dict[str, object]) -> List[str]:
        primitives: List[str] = []

        for field in ["actor", "intent", "action", "object", "topic", "time", "priority"]:
            value = parsed_memory.get(field)
            if isinstance(value, str) and value:
                primitives.append(f"{field}:{value}")

        list_fields = {
            "modifiers": "modifier",
            "negations": "negation",
            "constraints": "constraint",
            "conditions": "condition",
        }

        for field, primitive_type in list_fields.items():
            values = parsed_memory.get(field)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and value:
                        primitives.append(f"{primitive_type}:{value}")

        return primitives

    @staticmethod
    def _same_type(left: str, right: str) -> bool:
        return left.split(":", 1)[0] == right.split(":", 1)[0]