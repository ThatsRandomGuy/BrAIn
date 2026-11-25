from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from brain_concept_abstraction_v2 import ConceptChainStep, ConceptReasoningPath
from brain_input_v1 import BrainInputV1
from brain_planning_simulation_v3 import BrainPlanningSimulationV3, PlanningDecision, PlanningPath
from brain_reasoning_engine import DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class GeneratedExpansionRule:
    rule_id: str
    relation_type: str
    consequent: BrainInputV1
    concept_link_id: Optional[int]
    usage_count: int
    success_count: int
    failure_count: int
    reliability: float
    active: bool = True

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "rule_id": self.rule_id,
            "relation_type": self.relation_type,
            "consequent": self.consequent.to_dict(),
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "reliability": round(self.reliability, 4),
            "active": self.active,
        }
        if self.concept_link_id is not None:
            payload["concept_link_id"] = self.concept_link_id
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GeneratedExpansionRule":
        return cls(
            rule_id=str(data.get("rule_id", "")),
            relation_type=str(data.get("relation_type", "")),
            consequent=BrainInputV1.from_dict(data.get("consequent", {}) if isinstance(data.get("consequent", {}), dict) else {}),
            concept_link_id=int(data["concept_link_id"]) if data.get("concept_link_id") is not None else None,
            usage_count=int(data.get("usage_count", 0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            reliability=float(data.get("reliability", 0.5)),
            active=bool(data.get("active", True)),
        )


class GeneratedExpansionRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.rules: Dict[str, GeneratedExpansionRule] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self.rules = {}
            return
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded: Dict[str, GeneratedExpansionRule] = {}
        if isinstance(data, dict):
            for item in data.get("rules", []):
                if not isinstance(item, dict):
                    continue
                rule = GeneratedExpansionRule.from_dict(item)
                if rule.rule_id:
                    loaded[rule.rule_id] = rule
        self.rules = loaded

    def save(self) -> None:
        payload = {
            "rules": [self.rules[key].to_dict() for key in sorted(self.rules)],
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def get(self, rule_id: str) -> Optional[GeneratedExpansionRule]:
        return self.rules.get(str(rule_id))

    def update(
        self,
        rule_id: str,
        relation_type: str,
        consequent: BrainInputV1,
        concept_link_id: Optional[int],
        success: bool,
    ) -> GeneratedExpansionRule:
        existing = self.rules.get(rule_id)
        if existing is None:
            usage_count = 1
            success_count = 1 if success else 0
            failure_count = 0 if success else 1
        else:
            usage_count = existing.usage_count + 1
            success_count = existing.success_count + (1 if success else 0)
            failure_count = existing.failure_count + (0 if success else 1)
        total = max(1, success_count + failure_count)
        reliability = round(success_count / total, 4)
        active = not (failure_count >= 3 and failure_count > (success_count * 2))
        updated = GeneratedExpansionRule(
            rule_id=rule_id,
            relation_type=relation_type,
            consequent=consequent,
            concept_link_id=concept_link_id,
            usage_count=usage_count,
            success_count=success_count,
            failure_count=failure_count,
            reliability=reliability,
            active=active,
        )
        self.rules[rule_id] = updated
        self.save()
        return updated

    def all_active_rules(self) -> List[GeneratedExpansionRule]:
        return [self.rules[key] for key in sorted(self.rules) if self.rules[key].active]


class BrainGenerativeExpansionV35:
    def __init__(
        self,
        state_dir: str | Path = DEFAULT_OUTPUT_DIR / "brain_generative_v35.state",
        minimum_candidate_paths: int = 3,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.v3_engine = BrainPlanningSimulationV3(state_dir=self.state_dir / "v3")
        self.generated_registry = GeneratedExpansionRegistry(self.state_dir / "brain_generated_expansions.json")
        self.minimum_candidate_paths = minimum_candidate_paths

    def plan(
        self,
        brain_input: BrainInputV1,
        max_depth: int = 5,
        beam_width: int = 4,
        current_sequence: int = 1,
    ) -> PlanningDecision:
        root = self.v3_engine.v2_engine.v1_engine.native_engine.normalize(brain_input, current_sequence=current_sequence)
        concept_paths = self._simulate_with_generation(
            root=root,
            max_depth=max_depth,
            beam_width=beam_width,
            current_sequence=current_sequence,
        )
        planning_paths = [
            self.v3_engine._score_concept_path(path, current_sequence=current_sequence)
            for path in concept_paths
        ]
        planning_paths = sorted(
            planning_paths,
            key=lambda item: (-item.decision_score, -item.confidence, -item.depth, self.v3_engine._planning_sort_key(item)),
        )
        highest_confidence_path = max(
            planning_paths,
            key=lambda item: (item.confidence, item.decision_score, item.depth, tuple(self.v3_engine._planning_sort_key(item))),
            default=None,
        )
        selected_path = planning_paths[0] if planning_paths else None
        return PlanningDecision(
            input_brain=root,
            candidate_paths=planning_paths,
            selected_path=selected_path,
            highest_confidence_path=highest_confidence_path,
            decision_score=selected_path.decision_score if selected_path is not None else 0.0,
            final_predicted_outcome=selected_path.final_outcome if selected_path is not None else None,
        )

    def validate_decision(
        self,
        input_brain: BrainInputV1,
        decision: PlanningDecision,
        preferred_chain: Sequence[BrainInputV1],
        preferred_relation_types: Sequence[str],
        alternative_chains: Sequence[Tuple[Sequence[BrainInputV1], Sequence[str], float, float]],
        current_sequence: int,
    ) -> Dict[str, object]:
        validation = self.v3_engine.validate_decision(
            input_brain=input_brain,
            decision=decision,
            preferred_chain=preferred_chain,
            preferred_relation_types=preferred_relation_types,
            alternative_chains=alternative_chains,
            current_sequence=current_sequence,
        )
        selected = decision.selected_path
        generated_rule_ids: List[str] = []
        promoted_link_ids: List[int] = []
        if selected is not None:
            success = bool(validation.get("success", False))
            prior_state = input_brain
            for step in selected.steps:
                if step.source_level != "generated":
                    prior_state = step.consequent
                    continue
                rule_id = self._generated_rule_id(
                    relation_type=step.relation_type,
                    consequent=step.consequent,
                    concept_link_id=step.concept_link_id,
                )
                self.generated_registry.update(
                    rule_id=rule_id,
                    relation_type=step.relation_type,
                    consequent=step.consequent,
                    concept_link_id=step.concept_link_id,
                    success=success,
                )
                generated_rule_ids.append(rule_id)
                if success:
                    link = self.v3_engine.v2_engine.learn_link(
                        antecedent=prior_state,
                        consequent=step.consequent,
                        relation_type=step.relation_type,
                        current_sequence=current_sequence,
                        notes=["promoted_from_generated_path"],
                    )
                    self.v3_engine.v2_engine.v1_engine.link_registry.update_usage(
                        link_id=link.link_id,
                        current_sequence=current_sequence,
                        success=True,
                    )
                    promoted_link_ids.append(link.link_id)
                prior_state = step.consequent
        self.v3_engine.v2_engine.rebuild_concepts(current_sequence=current_sequence)
        return {
            **validation,
            "generated_rule_ids": generated_rule_ids,
            "promoted_link_ids": promoted_link_ids,
            "used_generation": bool(generated_rule_ids),
        }

    def registry_summary(self) -> Dict[str, object]:
        active_rules = self.generated_registry.all_active_rules()
        average_generated_reliability = round(
            sum(rule.reliability for rule in active_rules) / max(1, len(active_rules)),
            4,
        ) if active_rules else 0.0
        return {
            **self.v3_engine.registry_summary(),
            "generated_rule_count": len(active_rules),
            "average_generated_rule_reliability": average_generated_reliability,
        }

    def _simulate_with_generation(
        self,
        root: BrainInputV1,
        max_depth: int,
        beam_width: int,
        current_sequence: int,
    ) -> List[ConceptReasoningPath]:
        self.v3_engine.v2_engine.rebuild_concepts(current_sequence=current_sequence)
        frontier: List[ConceptReasoningPath] = [
            ConceptReasoningPath(
                steps=[],
                final_outcome=root,
                confidence=1.0,
            )
        ]
        completed: List[ConceptReasoningPath] = []

        for _ in range(max_depth):
            next_frontier: List[ConceptReasoningPath] = []
            for path in frontier:
                normal_matches = self.v3_engine.v2_engine._find_matching_candidates(path.final_outcome, current_sequence=current_sequence)
                matches = list(normal_matches)
                if len(matches) < self.minimum_candidate_paths:
                    existing_keys = {
                        self._state_key(candidate.consequent)
                        for candidate in matches
                    }
                    generated_matches = self._find_generative_matches(
                        current=path.final_outcome,
                        current_sequence=current_sequence,
                        exclude_state_keys=existing_keys,
                    )
                    matches.extend(generated_matches)
                if not matches:
                    completed.append(path)
                    continue
                for candidate in matches[:beam_width]:
                    if self.v3_engine.v2_engine._would_cycle(path, candidate.consequent):
                        continue
                    step_confidence = round(path.confidence * candidate.confidence_factor, 4)
                    step = ConceptChainStep(
                        source_level=candidate.source_level,
                        relation_type=candidate.relation_type,
                        antecedent=path.final_outcome,
                        consequent=candidate.consequent,
                        confidence=step_confidence,
                        match_score=candidate.match_score,
                        link_id=candidate.link_id,
                        concept_link_id=candidate.concept_link_id,
                        concept_ids=list(candidate.concept_ids),
                    )
                    next_frontier.append(
                        ConceptReasoningPath(
                            steps=list(path.steps) + [step],
                            final_outcome=candidate.consequent,
                            confidence=step_confidence,
                        )
                    )
            if not next_frontier:
                break
            frontier = sorted(
                next_frontier,
                key=lambda item: (-item.confidence, -item.depth, self.v3_engine.v2_engine._path_sort_key(item)),
            )[: max(beam_width, self.minimum_candidate_paths)]

        completed.extend(frontier)
        candidate_paths = [path for path in completed if path.steps]
        candidate_paths = self.v3_engine.v2_engine._apply_support_bonus(candidate_paths)
        if len(candidate_paths) < self.minimum_candidate_paths:
            candidate_paths = self._ensure_minimum_paths(candidate_paths)
        return candidate_paths

    def _find_generative_matches(
        self,
        current: BrainInputV1,
        current_sequence: int,
        exclude_state_keys: Sequence[str],
    ) -> List[object]:
        current_memory = self.v3_engine.v2_engine._memory_from_brain(current)
        exclude = {str(item) for item in exclude_state_keys}
        generated: List[object] = []
        for concept_link in self.v3_engine.v2_engine.concept_registry.all_active_concept_links():
            template_match = self.v3_engine.v2_engine.v1_engine._match_score(current, concept_link.antecedent_template)
            if template_match < 0.25:
                continue
            template_memory = self.v3_engine.v2_engine._memory_from_brain(concept_link.antecedent_template)
            if str(template_memory.action or "") and str(template_memory.action or "") != str(current_memory.action or ""):
                continue
            matched_concept_ids: List[int] = []
            membership_signals: List[float] = []
            for slot_path, concept_id in sorted(concept_link.concept_bindings.items()):
                slot_value = self.v3_engine.v2_engine._get_slot_path_value(current_memory, slot_path)
                if not slot_value:
                    membership_signals = []
                    break
                concept = self.v3_engine.v2_engine.concept_registry.get_concept(concept_id)
                matched_concept_ids.append(concept_id)
                if slot_value in concept.members:
                    membership_signals.append(1.0)
                else:
                    membership_signals.append(0.5 if concept.support_count >= 2 else 0.0)
            if not membership_signals or min(membership_signals) <= 0.0:
                continue
            consequent_key = self._state_key(concept_link.consequent)
            if consequent_key in exclude:
                continue
            rule_id = self._generated_rule_id(
                relation_type=concept_link.relation_type,
                consequent=concept_link.consequent,
                concept_link_id=concept_link.concept_link_id,
            )
            rule = self.generated_registry.get(rule_id)
            if rule is not None and not rule.active:
                continue
            membership_signal = sum(membership_signals) / len(membership_signals)
            concept_signal = sum(
                self.v3_engine.v2_engine.concept_registry.get_concept(concept_id).confidence
                for concept_id in matched_concept_ids
            ) / max(1, len(matched_concept_ids))
            rule_bias = rule.reliability if rule is not None else 0.5
            match_score = round((template_match * 0.65) + (membership_signal * 0.35), 4)
            confidence_factor = round(
                max(
                    0.08,
                    min(
                        0.55,
                        0.08
                        + (template_match * 0.14)
                        + (membership_signal * 0.08)
                        + (concept_signal * 0.06)
                        + (concept_link.reliability * 0.05)
                        + (rule_bias * 0.04),
                    ),
                ),
                4,
            )
            generated.append(
                type(
                    "GeneratedCandidate",
                    (),
                    {
                        "source_level": "generated",
                        "relation_type": concept_link.relation_type,
                        "antecedent": current,
                        "consequent": concept_link.consequent,
                        "match_score": match_score,
                        "confidence_factor": confidence_factor,
                        "link_id": None,
                        "concept_link_id": concept_link.concept_link_id,
                        "concept_ids": matched_concept_ids,
                    },
                )()
            )

        return sorted(
            generated,
            key=lambda item: (
                -item.confidence_factor,
                -item.match_score,
                item.concept_link_id if item.concept_link_id is not None else 99_999,
            ),
        )

    def _ensure_minimum_paths(
        self,
        candidate_paths: Sequence[ConceptReasoningPath],
    ) -> List[ConceptReasoningPath]:
        unique: Dict[Tuple[str, int], ConceptReasoningPath] = {}
        for path in candidate_paths:
            key = (self._state_key(path.final_outcome), path.depth)
            if key not in unique:
                unique[key] = path
        paths = sorted(
            unique.values(),
            key=lambda item: (-item.confidence, -item.depth, self.v3_engine.v2_engine._path_sort_key(item)),
        )
        return paths[: max(len(paths), self.minimum_candidate_paths)]

    def _generated_rule_id(
        self,
        relation_type: str,
        consequent: BrainInputV1,
        concept_link_id: Optional[int],
    ) -> str:
        raw = f"{relation_type}|{concept_link_id or 0}|{self._state_key(consequent)}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _state_key(self, brain_input: BrainInputV1) -> str:
        bundle = self.v3_engine.v2_engine.v1_engine.runtime.codec.encode_memory_bundle(
            self.v3_engine.v2_engine.v1_engine.runtime.brain_input_codec.brain_input_to_memory(brain_input)
        )
        return json.dumps(bundle, sort_keys=True)
