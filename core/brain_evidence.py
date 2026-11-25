from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


@dataclass
class EvidenceRecord:
    primitive: str
    sightings: int
    total_score: float
    average_score: float
    promoted: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "primitive": self.primitive,
            "sightings": self.sightings,
            "total_score": round(self.total_score, 4),
            "average_score": round(self.average_score, 4),
            "promoted": self.promoted,
        }


class BrainEvidence:
    """
    Tracks repeated sightings of provisional concepts.

    Goal:
    - prevent one-off guesses from becoming permanent vocabulary
    - promote concepts only after enough repeated evidence
    """

    PROMOTION_SIGHTINGS = 2
    PROMOTION_AVG_SCORE = 0.63

    def __init__(self, evidence_path: str | Path = "brain_evidence.json"):
        self.evidence_path = Path(evidence_path)
        self.data: Dict[str, Dict[str, object]] = {}
        self._load()

    def _load(self) -> None:
        if not self.evidence_path.exists():
            self.data = {}
            return

        with open(self.evidence_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        if isinstance(loaded, dict):
            self.data = loaded
        else:
            self.data = {}

    def save(self) -> None:
        with open(self.evidence_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def observe(self, provisional_items: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
        promoted_now: List[Dict[str, object]] = []
        tracked: List[Dict[str, object]] = []

        for item in provisional_items:
            primitive = str(item.get("primitive", "")).strip()
            score = float(item.get("score", 0.0))

            if not primitive:
                continue

            record = self.data.get(
                primitive,
                {
                    "sightings": 0,
                    "total_score": 0.0,
                    "average_score": 0.0,
                    "promoted": False,
                },
            )

            record["sightings"] = int(record["sightings"]) + 1
            record["total_score"] = float(record["total_score"]) + score
            record["average_score"] = float(record["total_score"]) / int(record["sightings"])

            already_promoted = bool(record.get("promoted", False))
            should_promote = (
                not already_promoted
                and int(record["sightings"]) >= self.PROMOTION_SIGHTINGS
                and float(record["average_score"]) >= self.PROMOTION_AVG_SCORE
            )

            if should_promote:
                record["promoted"] = True
                promoted_now.append({
                    "primitive": primitive,
                    "sightings": int(record["sightings"]),
                    "average_score": round(float(record["average_score"]), 2),
                })

            tracked.append({
                "primitive": primitive,
                "sightings": int(record["sightings"]),
                "average_score": round(float(record["average_score"]), 2),
                "promoted": bool(record["promoted"]),
            })

            self.data[primitive] = record

        self.save()

        return {
            "tracked": tracked,
            "promoted_now": promoted_now,
        }

    def get_record(self, primitive: str) -> EvidenceRecord | None:
        record = self.data.get(primitive)
        if not record:
            return None

        return EvidenceRecord(
            primitive=primitive,
            sightings=int(record["sightings"]),
            total_score=float(record["total_score"]),
            average_score=float(record["average_score"]),
            promoted=bool(record["promoted"]),
        )