from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from brain_codec import BrainMemory


def _has_text(value: object) -> bool:
    return bool(str(value or "").strip())


@dataclass(frozen=True)
class EnrichmentInference:
    field_name: str
    inferred_value: object
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "field_name": self.field_name,
            "inferred_value": self.inferred_value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PromptEnrichmentResult:
    route_type: str
    original_prompt: str
    original_memory: Dict[str, object]
    enriched_memory: Dict[str, object]
    inferred_fields: List[Dict[str, object]] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_reasons: List[str] = field(default_factory=list)
    next_action: str = "continue"
    clarification_question: str = ""
    missing_fields: List[str] = field(default_factory=list)
    clarification_type: str = ""
    enrichment_applied: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "route_type": self.route_type,
            "original_prompt": self.original_prompt,
            "original_memory": dict(self.original_memory),
            "enriched_memory": dict(self.enriched_memory),
            "inferred_fields": list(self.inferred_fields),
            "needs_clarification": self.needs_clarification,
            "clarification_reasons": list(self.clarification_reasons),
            "next_action": self.next_action,
            "clarification_question": self.clarification_question,
            "missing_fields": list(self.missing_fields),
            "clarification_type": self.clarification_type,
            "enrichment_applied": self.enrichment_applied,
        }


class BrainPromptEnrichment:
    def enrich(
        self,
        *,
        prompt_text: str,
        route_type: str,
        memory: BrainMemory,
    ) -> PromptEnrichmentResult:
        normalized = str(prompt_text or "").strip().lower()
        enriched = BrainMemory.from_dict(memory.to_dict())
        inferred: List[EnrichmentInference] = []
        clarification_reasons: List[str] = []

        if route_type == "planning_request":
            self._enrich_planning(normalized, enriched, inferred, clarification_reasons)
        elif route_type == "goal_request":
            self._enrich_goal(normalized, enriched, inferred, clarification_reasons)
        elif route_type == "transform_request":
            self._enrich_transform(normalized, enriched, inferred, clarification_reasons)
        elif route_type == "hybrid_request":
            self._enrich_hybrid(normalized, enriched, inferred, clarification_reasons)

        self._dedupe_lists(enriched)
        needs_clarification = bool(clarification_reasons)
        missing_fields = self._missing_fields_for_reasons(clarification_reasons)
        clarification_type = self._clarification_type(route_type, missing_fields)
        clarification_question = self._clarification_question(route_type, clarification_type, enriched)

        return PromptEnrichmentResult(
            route_type=route_type,
            original_prompt=prompt_text,
            original_memory=memory.to_dict(),
            enriched_memory=enriched.to_dict(),
            inferred_fields=[item.to_dict() for item in inferred],
            needs_clarification=needs_clarification,
            clarification_reasons=clarification_reasons,
            next_action="ask_user" if needs_clarification else "continue",
            clarification_question=clarification_question if needs_clarification else "",
            missing_fields=missing_fields,
            clarification_type=clarification_type if needs_clarification else "",
            enrichment_applied=bool(inferred),
        )

    def _enrich_planning(
        self,
        normalized: str,
        memory: BrainMemory,
        inferred: List[EnrichmentInference],
        clarification_reasons: List[str],
    ) -> None:
        concrete_actions = {"charge", "repair", "study", "clean", "backup", "practice", "cook", "eat", "drink", "water"}
        if not _has_text(memory.action):
            if "finish" in normalized or "complete" in normalized:
                self._set_if_missing(memory, "action", "finish", "Prompt asks for task completion.", inferred)
            else:
                self._set_if_missing(memory, "action", "plan", "Planning request lacked a concrete action.", inferred)
        if not _has_text(memory.object) and any(token in normalized for token in ("task", "work", "project")):
            self._set_if_missing(memory, "object", "task", "Prompt mentions a task/work target.", inferred)
        if "safe" in normalized or "safely" in normalized:
            self._append_if_missing(memory.constraints, "avoid_risk", "Prompt explicitly prioritizes safety.", inferred, "constraints")
        if not _has_text(memory.object):
            clarification_reasons.append("missing object")
        if str(memory.action or "").strip().lower() not in concrete_actions and not any(token in normalized for token in ("charge", "charging", "repair", "repairing", "study", "studying", "clean", "cleaning", "backup", "backing up", "practice", "practicing", "cook", "cooking", "eat", "eating", "drink", "drinking", "water", "watering")):
            clarification_reasons.append("missing concrete action target")

    def _enrich_goal(
        self,
        normalized: str,
        memory: BrainMemory,
        inferred: List[EnrichmentInference],
        clarification_reasons: List[str],
    ) -> None:
        options_provided = " or " in normalized or " between " in normalized or "," in normalized
        if not _has_text(memory.action):
            self._set_if_missing(memory, "action", "compare", "Goal request implies comparison or selection.", inferred)
        if not _has_text(memory.object) and any(token in normalized for token in ("option", "choice", "path")):
            self._set_if_missing(memory, "object", "option", "Prompt asks to choose among options.", inferred)
        if not _has_text(memory.topic) and "goal" in normalized:
            self._set_if_missing(memory, "topic", "goal_alignment", "Prompt asks for goal fit/alignment.", inferred)
        if not options_provided and not any(token in normalized for token in ("option", "choice", "path", "alternative")):
            clarification_reasons.append("missing options to compare")
        if "goal" not in normalized and not _has_text(memory.topic):
            clarification_reasons.append("missing goal target")
        if _has_text(memory.object) and str(memory.object) == "option" and not options_provided:
            clarification_reasons.append("missing option details")

    def _enrich_transform(
        self,
        normalized: str,
        memory: BrainMemory,
        inferred: List[EnrichmentInference],
        clarification_reasons: List[str],
    ) -> None:
        if not _has_text(memory.action):
            self._set_if_missing(memory, "action", "modify", "Transform request implies modification.", inferred)
        if not _has_text(memory.object) and any(token in normalized for token in ("plan", "draft", "text", "message")):
            self._set_if_missing(memory, "object", "plan", "Prompt names an artifact to transform.", inferred)
        if "risk" in normalized:
            self._append_if_missing(memory.constraints, "avoid_risk", "Prompt asks to reduce risk.", inferred, "constraints")
        if not _has_text(memory.object):
            clarification_reasons.append("missing artifact to transform")

    def _enrich_hybrid(
        self,
        normalized: str,
        memory: BrainMemory,
        inferred: List[EnrichmentInference],
        clarification_reasons: List[str],
    ) -> None:
        self._enrich_planning(normalized, memory, inferred, clarification_reasons)
        if "goal" in normalized and not _has_text(memory.topic):
            self._set_if_missing(memory, "topic", "goal_alignment", "Hybrid prompt references user goals.", inferred)

    def _set_if_missing(
        self,
        memory: BrainMemory,
        field_name: str,
        inferred_value: str,
        reason: str,
        inferred: List[EnrichmentInference],
    ) -> None:
        current_value = getattr(memory, field_name)
        if _has_text(current_value):
            return
        setattr(memory, field_name, inferred_value)
        inferred.append(EnrichmentInference(field_name=field_name, inferred_value=inferred_value, reason=reason))

    def _append_if_missing(
        self,
        collection: List[str],
        inferred_value: str,
        reason: str,
        inferred: List[EnrichmentInference],
        field_name: str,
    ) -> None:
        if inferred_value in collection:
            return
        collection.append(inferred_value)
        inferred.append(EnrichmentInference(field_name=field_name, inferred_value=inferred_value, reason=reason))

    @staticmethod
    def _dedupe_lists(memory: BrainMemory) -> None:
        memory.modifiers = list(dict.fromkeys(str(item) for item in memory.modifiers if _has_text(item)))
        memory.constraints = list(dict.fromkeys(str(item) for item in memory.constraints if _has_text(item)))
        memory.negations = list(dict.fromkeys(str(item) for item in memory.negations if _has_text(item)))
        memory.conditions = list(dict.fromkeys(str(item) for item in memory.conditions if _has_text(item)))

    @staticmethod
    def _missing_fields_for_reasons(clarification_reasons: List[str]) -> List[str]:
        mapping = {
            "missing object": "object",
            "missing concrete action target": "action_target",
            "missing options to compare": "options",
            "missing option details": "options",
            "missing goal target": "goal_target",
            "missing artifact to transform": "object",
        }
        fields: List[str] = []
        for reason in clarification_reasons:
            field_name = mapping.get(reason)
            if field_name and field_name not in fields:
                fields.append(field_name)
        return fields

    @staticmethod
    def _clarification_type(route_type: str, missing_fields: List[str]) -> str:
        if "options" in missing_fields:
            return "ask_options"
        if "goal_target" in missing_fields:
            return "ask_goal_target"
        if "object" in missing_fields or "action_target" in missing_fields:
            if route_type == "planning_request":
                return "ask_task_target"
            if route_type == "transform_request":
                return "ask_transform_target"
        return "ask_missing_detail"

    @staticmethod
    def _clarification_question(route_type: str, clarification_type: str, memory: BrainMemory) -> str:
        if clarification_type == "ask_options":
            return "What options are you choosing between?"
        if clarification_type == "ask_goal_target":
            return "What goal should BrAIn optimize for?"
        if clarification_type == "ask_transform_target":
            return "What plan or text do you want me to transform?"
        if clarification_type == "ask_task_target":
            if str(memory.action or "").strip().lower() in {"finish", "complete"}:
                return "What task are you trying to finish?"
            return "What concrete task or object do you want help with?"
        if route_type == "planning_request":
            return "What concrete task or object do you want help planning?"
        if route_type == "goal_request":
            return "What missing detail should BrAIn use to compare the options?"
        return "What missing detail should BrAIn use to continue?"
