from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


@dataclass
class MergeDecision:
    original: str
    canonical: str
    reason: str


class BrainMerge:
    """
    Handles primitive aliasing / concept merging.
    """

    def __init__(self, merge_path: str | Path = "brain_merge_map.json"):
        self.merge_path = Path(merge_path)
        self.alias_to_canonical: Dict[str, str] = {}
        self._load()
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        defaults = {
            "object:webapp": "object:website",
            "object:web_app": "object:website",
            "object:site": "object:website",
            "modifier:scary": "modifier:horror",
            "modifier:creepy": "modifier:horror",
            "modifier:comedic": "modifier:funny",
            "modifier:comedy": "modifier:funny",
            "topic:ui": "topic:interface",
            "topic:webapp": "topic:website",
            "topic:web_app": "topic:website",
        }

        for alias, canonical in defaults.items():
            self.alias_to_canonical.setdefault(alias, canonical)

    def _load(self) -> None:
        if not self.merge_path.exists():
            self.alias_to_canonical = {}
            return

        with open(self.merge_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            self.alias_to_canonical = {str(k): str(v) for k, v in data.items()}
        else:
            self.alias_to_canonical = {}

    def save(self) -> None:
        with open(self.merge_path, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(self.alias_to_canonical.items())), f, indent=2)

    def canonicalize_primitive(self, primitive: str) -> str:
        seen = set()
        current = primitive

        while current in self.alias_to_canonical and current not in seen:
            seen.add(current)
            current = self.alias_to_canonical[current]

        return current

    def canonicalize_list(self, primitives: List[str]) -> tuple[List[str], List[MergeDecision]]:
        out: List[str] = []
        decisions: List[MergeDecision] = []

        for primitive in primitives:
            canonical = self.canonicalize_primitive(primitive)
            if canonical != primitive:
                decisions.append(
                    MergeDecision(
                        original=primitive,
                        canonical=canonical,
                        reason="alias_map",
                    )
                )
            out.append(canonical)

        deduped: List[str] = []
        seen = set()
        for primitive in out:
            if primitive not in seen:
                seen.add(primitive)
                deduped.append(primitive)

        return deduped, decisions

    def canonicalize_policy_items(
        self,
        items: List[Dict[str, object]],
    ) -> tuple[List[Dict[str, object]], List[Dict[str, str]]]:
        normalized: List[Dict[str, object]] = []
        decisions: List[Dict[str, str]] = []
        seen = set()

        for item in items:
            primitive = str(item.get("primitive", "")).strip()
            if not primitive:
                continue

            canonical = self.canonicalize_primitive(primitive)
            new_item = dict(item)
            new_item["primitive"] = canonical

            if canonical != primitive:
                decisions.append({
                    "original": primitive,
                    "canonical": canonical,
                    "reason": "alias_map",
                })

            key = (canonical, new_item.get("action"))
            if key in seen:
                continue
            seen.add(key)
            normalized.append(new_item)

        return normalized, decisions

    def add_alias(self, alias: str, canonical: str) -> None:
        if alias == canonical:
            return
        self.alias_to_canonical[alias] = canonical
        self.save()

    def apply_approved_queue_items(
        self,
        approved_items: List[Dict[str, object]],
        prefer: str = "right",
    ) -> List[Dict[str, str]]:
        """
        Converts approved queue items into alias map entries.

        prefer:
        - 'right' => left becomes alias of right
        - 'left'  => right becomes alias of left
        """
        applied: List[Dict[str, str]] = []

        for item in approved_items:
            left = str(item.get("left", "")).strip()
            right = str(item.get("right", "")).strip()

            if not left or not right or left == right:
                continue

            if prefer == "left":
                alias = right
                canonical = left
            else:
                alias = left
                canonical = right

            self.add_alias(alias, canonical)
            applied.append({
                "alias": alias,
                "canonical": canonical,
            })

        return applied