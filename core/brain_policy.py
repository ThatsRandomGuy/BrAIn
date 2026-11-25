from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from brain_codec import BrainCodec, BrainMemory


@dataclass
class RegistrationDecision:
    primitive: str
    value: str
    source: str
    score: float
    action: str  # register | provisional | ignore


class BrainPolicy:
    """
    BrAIn policy layer.

    Decides which parsed values should:
    - be registered immediately
    - remain provisional
    - be ignored for registry growth
    """

    REGISTER_THRESHOLD = 0.80
    PROVISIONAL_THRESHOLD = 0.60

    SINGLE_FIELDS = {"actor", "intent", "action", "object", "topic", "time", "priority"}
    LIST_FIELDS = {"modifiers", "negations", "constraints", "conditions"}

    def evaluate(self, memory: BrainMemory) -> List[RegistrationDecision]:
        decisions: List[RegistrationDecision] = []

        confidence = memory.confidence or {}

        for field_name in self.SINGLE_FIELDS:
            entry = confidence.get(field_name)
            if not isinstance(entry, dict):
                continue

            value = entry.get("value")
            source = entry.get("source", "unknown")
            score = float(entry.get("score", 0.0))

            if not value:
                continue

            primitive_type = self._field_to_primitive_type(field_name)
            primitive = f"{primitive_type}:{value}"
            decisions.append(
                RegistrationDecision(
                    primitive=primitive,
                    value=str(value),
                    source=str(source),
                    score=score,
                    action=self._decide_action(score),
                )
            )

        for field_name in self.LIST_FIELDS:
            entries = confidence.get(field_name)
            if not isinstance(entries, list):
                continue

            primitive_type = self._field_to_primitive_type(field_name)
            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                value = entry.get("value")
                source = entry.get("source", "unknown")
                score = float(entry.get("score", 0.0))

                if not value:
                    continue

                primitive = f"{primitive_type}:{value}"
                decisions.append(
                    RegistrationDecision(
                        primitive=primitive,
                        value=str(value),
                        source=str(source),
                        score=score,
                        action=self._decide_action(score),
                    )
                )

        return self._dedupe(decisions)

    def apply(self, codec: BrainCodec, memory: BrainMemory) -> Dict[str, List[Dict[str, object]]]:
        decisions = self.evaluate(memory)

        registered: List[Dict[str, object]] = []
        provisional: List[Dict[str, object]] = []
        ignored: List[Dict[str, object]] = []

        for decision in decisions:
            record = {
                "primitive": decision.primitive,
                "value": decision.value,
                "source": decision.source,
                "score": round(decision.score, 2),
                "action": decision.action,
            }

            if decision.action == "register":
                codec.register_primitive(decision.primitive)
                registered.append(record)
            elif decision.action == "provisional":
                provisional.append(record)
            else:
                ignored.append(record)

        return {
            "registered": registered,
            "provisional": provisional,
            "ignored": ignored,
        }

    def register_promoted(self, codec: BrainCodec, promoted_items: List[Dict[str, object]]) -> List[Dict[str, object]]:
        registered: List[Dict[str, object]] = []

        for item in promoted_items:
            primitive = str(item.get("primitive", "")).strip()
            if not primitive:
                continue

            codec.register_primitive(primitive)
            registered.append({
                "primitive": primitive,
                "sightings": int(item.get("sightings", 0)),
                "average_score": float(item.get("average_score", 0.0)),
                "action": "register_from_evidence",
            })

        return registered

    def _decide_action(self, score: float) -> str:
        if score >= self.REGISTER_THRESHOLD:
            return "register"
        if score >= self.PROVISIONAL_THRESHOLD:
            return "provisional"
        return "ignore"

    @staticmethod
    def _field_to_primitive_type(field_name: str) -> str:
        mapping = {
            "actor": "actor",
            "intent": "intent",
            "action": "action",
            "object": "object",
            "topic": "topic",
            "time": "time",
            "priority": "priority",
            "modifiers": "modifier",
            "negations": "negation",
            "constraints": "constraint",
            "conditions": "condition",
        }
        return mapping[field_name]

    @staticmethod
    def _dedupe(decisions: List[RegistrationDecision]) -> List[RegistrationDecision]:
        best: Dict[str, RegistrationDecision] = {}

        for decision in decisions:
            existing = best.get(decision.primitive)
            if existing is None or decision.score > existing.score:
                best[decision.primitive] = decision

        return list(best.values())