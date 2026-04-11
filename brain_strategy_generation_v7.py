from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from brain_input_v1 import BrainInputV1


def _has_text(value: object) -> bool:
    return bool(str(value or "").strip())


def _slug(value: str) -> str:
    sanitized = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or "").strip())
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized.strip("-")


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


@dataclass(frozen=True)
class StrategyStep:
    step_id: str
    title: str
    action: str
    detail: str
    depends_on: List[str] = field(default_factory=list)
    risk_level: str = "low"
    inferred: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "action": self.action,
            "detail": self.detail,
            "depends_on": list(self.depends_on),
            "risk_level": self.risk_level,
            "inferred": self.inferred,
        }


@dataclass(frozen=True)
class StrategyObject:
    strategy_id: str
    name: str
    focus: str
    goal: str
    domain: str
    target: str
    actions: List[str]
    steps: List[StrategyStep]
    dependencies: List[Dict[str, object]]
    scores: Dict[str, float]
    tradeoffs: List[str]
    signature: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "focus": self.focus,
            "goal": self.goal,
            "domain": self.domain,
            "target": self.target,
            "actions": list(self.actions),
            "steps": [step.to_dict() for step in self.steps],
            "dependencies": list(self.dependencies),
            "scores": {str(key): round(float(value), 4) for key, value in sorted(self.scores.items())},
            "tradeoffs": list(self.tradeoffs),
            "signature": self.signature,
        }


@dataclass(frozen=True)
class StrategyPatternRecord:
    signature: str
    domain: str
    focus: str
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    incomplete_count: int = 0
    last_sequence: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "signature": self.signature,
            "domain": self.domain,
            "focus": self.focus,
            "use_count": self.use_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "incomplete_count": self.incomplete_count,
            "last_sequence": self.last_sequence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "StrategyPatternRecord":
        return cls(
            signature=str(data.get("signature", "")),
            domain=str(data.get("domain", "generic")),
            focus=str(data.get("focus", "balanced_completion")),
            use_count=int(data.get("use_count", 0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            incomplete_count=int(data.get("incomplete_count", 0)),
            last_sequence=int(data.get("last_sequence", 0)),
        )


@dataclass(frozen=True)
class StrategyPlanningResult:
    selected_strategy: Dict[str, object]
    alternative_strategies: List[Dict[str, object]]
    explanation: str
    strategy_feedback: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "selected_strategy": dict(self.selected_strategy),
            "alternative_strategies": list(self.alternative_strategies),
            "explanation": self.explanation,
            "strategy_feedback": dict(self.strategy_feedback),
        }


class BrainStrategyGenerationV7:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.patterns: Dict[str, StrategyPatternRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self.patterns = {}
            return
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        loaded = data.get("patterns", []) if isinstance(data, dict) else []
        self.patterns = {
            record.signature: record
            for record in (
                StrategyPatternRecord.from_dict(item)
                for item in loaded
                if isinstance(item, dict)
            )
            if _has_text(record.signature)
        }

    def save(self) -> None:
        payload = {
            "patterns": [
                record.to_dict()
                for record in sorted(
                    self.patterns.values(),
                    key=lambda item: (-item.last_sequence, item.signature),
                )
            ]
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def tracked_strategies_summary(self) -> List[Dict[str, object]]:
        return [
            record.to_dict()
            for record in sorted(
                self.patterns.values(),
                key=lambda item: (-item.last_sequence, -item.success_count, item.signature),
            )
        ]

    def process_turn(
        self,
        *,
        sequence: int,
        route_type: str,
        effective_prompt_text: str,
        parsed_brain_input: BrainInputV1,
        final_result: BrainInputV1,
        selected_path: Dict[str, object],
        goal_tracking: Dict[str, object],
        clarification_state: Dict[str, object],
        candidate_path_count: int,
    ) -> StrategyPlanningResult:
        progress_tracking = goal_tracking.get("progress_tracking", {}) if isinstance(goal_tracking, dict) else {}
        progress_status = str(progress_tracking.get("status", "unknown")) if isinstance(progress_tracking, dict) else "unknown"
        if clarification_state.get("active") and not clarification_state.get("resolved"):
            return StrategyPlanningResult(
                selected_strategy={},
                alternative_strategies=[],
                explanation="Strategy generation is deferred until the missing planning details are clarified.",
                strategy_feedback={
                    "status": "waiting_for_clarification",
                    "reason": clarification_state.get("clarification_question", ""),
                },
            )

        if route_type not in {"planning_request", "goal_request", "hybrid_request", "transform_request"}:
            return StrategyPlanningResult(
                selected_strategy={},
                alternative_strategies=[],
                explanation="No multi-step strategy was generated for this route.",
                strategy_feedback={"status": "route_not_strategy_driven"},
            )

        domain = self._detect_domain(effective_prompt_text, parsed_brain_input, final_result, candidate_path_count)
        if route_type == "goal_request" and domain != "failure_recovery":
            domain = "goal_tradeoff"
        target = self._target_text(parsed_brain_input, final_result, effective_prompt_text)
        goal = self._goal_text(route_type, final_result, target)
        primary_action = self._primary_action(parsed_brain_input, final_result, route_type)
        strategies: List[StrategyObject] = []

        for focus in self._strategy_focuses(domain, route_type, effective_prompt_text, candidate_path_count):
            steps = self._build_steps(
                domain=domain,
                focus=focus,
                goal=goal,
                target=target,
                primary_action=primary_action,
            )
            if not steps:
                continue
            signature = f"{domain}:{focus}:{_slug(primary_action or target or route_type)}"
            strategies.append(
                StrategyObject(
                    strategy_id=f"{_slug(domain)}-{_slug(focus)}",
                    name=self._strategy_name(domain, focus),
                    focus=focus,
                    goal=goal,
                    domain=domain,
                    target=target,
                    actions=[step.action for step in steps],
                    steps=steps,
                    dependencies=[
                        {"step_id": step.step_id, "depends_on": list(step.depends_on)}
                        for step in steps
                        if step.depends_on
                    ],
                    scores=self._score_strategy(
                        focus=focus,
                        domain=domain,
                        steps=steps,
                        selected_path=selected_path,
                        progress_status=progress_status,
                        signature=signature,
                    ),
                    tradeoffs=self._tradeoffs(domain, focus),
                    signature=signature,
                )
            )

        ranked = sorted(
            strategies,
            key=lambda item: (
                -item.scores.get("total_score", 0.0),
                -item.scores.get("goal_alignment", 0.0),
                item.focus,
            ),
        )
        if not ranked:
            return StrategyPlanningResult(
                selected_strategy={},
                alternative_strategies=[],
                explanation="No strategy template matched this turn strongly enough to produce a plan.",
                strategy_feedback={"status": "no_strategy_generated"},
            )

        selected = ranked[0]
        self._reinforce_strategy(
            signature=selected.signature,
            domain=selected.domain,
            focus=selected.focus,
            sequence=sequence,
            progress_status=progress_status,
        )

        record = self.patterns.get(selected.signature)
        return StrategyPlanningResult(
            selected_strategy=selected.to_dict(),
            alternative_strategies=[strategy.to_dict() for strategy in ranked[1:3]],
            explanation=self._strategy_explanation(selected, ranked[1:3], progress_status),
            strategy_feedback={
                "status": progress_status,
                "selected_signature": selected.signature,
                "selected_focus": selected.focus,
                "success_count": record.success_count if record else 0,
                "failure_count": record.failure_count if record else 0,
                "incomplete_count": record.incomplete_count if record else 0,
            },
        )

    def _reinforce_strategy(
        self,
        *,
        signature: str,
        domain: str,
        focus: str,
        sequence: int,
        progress_status: str,
    ) -> None:
        previous = self.patterns.get(signature)
        self.patterns[signature] = StrategyPatternRecord(
            signature=signature,
            domain=domain,
            focus=focus,
            use_count=(previous.use_count if previous else 0) + 1,
            success_count=(previous.success_count if previous else 0) + (1 if progress_status == "decision_ready" else 0),
            failure_count=(previous.failure_count if previous else 0) + (1 if progress_status == "stalled" else 0),
            incomplete_count=(previous.incomplete_count if previous else 0) + (1 if progress_status == "clarification_pending" else 0),
            last_sequence=sequence,
        )
        self.save()

    def _detect_domain(
        self,
        prompt_text: str,
        parsed_brain_input: BrainInputV1,
        final_result: BrainInputV1,
        candidate_path_count: int,
    ) -> str:
        text = " ".join(
            part.lower()
            for part in (
                prompt_text,
                parsed_brain_input.action,
                str(parsed_brain_input.slots.get("object", "")),
                str(parsed_brain_input.slots.get("topic", "")),
                final_result.action,
                str(final_result.slots.get("object", "")),
                str(final_result.slots.get("topic", "")),
            )
            if _has_text(part)
        )
        if any(token in text for token in ("option", "compare", "choice", "choose", "between")):
            return "goal_tradeoff"
        if any(token in text for token in ("fail", "failure", "stuck", "recover", "retry", "error")) or candidate_path_count == 0:
            return "failure_recovery"
        if any(token in text for token in ("charge", "charger", "battery", "laptop", "power", "plug")):
            return "device_charging"
        if any(token in text for token in ("rewrite", "transform", "reduce risk", "revise")):
            return "transform"
        return "task_completion"

    def _strategy_focuses(
        self,
        domain: str,
        route_type: str,
        prompt_text: str,
        candidate_path_count: int,
    ) -> List[str]:
        lowered = prompt_text.lower()
        if domain == "device_charging":
            return ["safe_charging", "fast_charging", "battery_longevity"]
        if domain == "goal_tradeoff" or route_type == "goal_request":
            return ["goal_fit", "risk_balanced", "decisive_choice"]
        if domain == "failure_recovery" or candidate_path_count == 0 or "fail" in lowered:
            return ["failure_recovery", "conservative_retry", "diagnostic_reset"]
        if domain == "transform":
            return ["risk_reduction", "minimal_change", "verification_first"]
        return ["balanced_completion", "safety_first", "verification_first"]

    def _goal_text(
        self,
        route_type: str,
        final_result: BrainInputV1,
        target: str,
    ) -> str:
        if route_type == "goal_request":
            return f"Choose the best option for {target or 'the active goal'}."
        if route_type == "transform_request":
            return f"Transform the current plan for {target or 'the requested change'} while preserving intent."
        if route_type == "hybrid_request":
            return f"Use recalled context to complete {target or 'the requested action'} safely."
        return f"Complete {target or final_result.action or 'the requested task'} with a reliable multi-step plan."

    def _target_text(
        self,
        parsed_brain_input: BrainInputV1,
        final_result: BrainInputV1,
        prompt_text: str,
    ) -> str:
        for value in (
            str(final_result.slots.get("object", "")),
            str(final_result.slots.get("topic", "")),
            str(parsed_brain_input.slots.get("object", "")),
            str(parsed_brain_input.slots.get("topic", "")),
        ):
            if _has_text(value):
                return value.strip()
        words = [word.strip(".,?!") for word in prompt_text.split() if word.strip(".,?!")]
        return " ".join(words[:6]).strip()

    def _primary_action(
        self,
        parsed_brain_input: BrainInputV1,
        final_result: BrainInputV1,
        route_type: str,
    ) -> str:
        for value in (final_result.action, parsed_brain_input.action):
            if _has_text(value):
                return str(value).strip()
        return {"goal_request": "compare", "transform_request": "transform"}.get(route_type, "complete")

    def _step(
        self,
        number: int,
        title: str,
        action: str,
        detail: str,
        depends_on: Optional[List[str]] = None,
        risk_level: str = "low",
    ) -> StrategyStep:
        return StrategyStep(
            step_id=f"step_{number}",
            title=title,
            action=action,
            detail=detail,
            depends_on=list(depends_on or []),
            risk_level=risk_level,
            inferred=True,
        )

    def _build_steps(
        self,
        *,
        domain: str,
        focus: str,
        goal: str,
        target: str,
        primary_action: str,
    ) -> List[StrategyStep]:
        if domain == "device_charging":
            return self._charging_steps(focus, target or "the device")
        if domain == "goal_tradeoff":
            return self._tradeoff_steps(focus, target or "the options")
        if domain == "failure_recovery":
            return self._recovery_steps(focus, target or primary_action or "the blocked task")
        if domain == "transform":
            return self._transform_steps(focus, target or "the current plan")
        return self._task_steps(focus, target or primary_action or goal)

    def _charging_steps(self, focus: str, target: str) -> List[StrategyStep]:
        if focus == "fast_charging":
            return [
                self._step(1, "Choose a capable charger", "select_power_source", f"Use a charger and cable that can reliably power {target}."),
                self._step(2, "Connect directly", "connect_charger", f"Plug {target} into a stable outlet before doing other work.", ["step_1"], "medium"),
                self._step(3, "Reduce power draw", "reduce_load", "Pause heavy workloads so more incoming power goes to charging.", ["step_2"], "medium"),
                self._step(4, "Check charge gain", "monitor_charge", "Verify the battery level is rising at the expected pace.", ["step_3"]),
                self._step(5, "Stop on heat or full charge", "disconnect_when_needed", "Disconnect or slow down if the device overheats or no longer needs rapid charging.", ["step_4"], "medium"),
            ]
        if focus == "battery_longevity":
            return [
                self._step(1, "Pick a target range", "set_charge_target", f"Decide what charge range is sufficient for {target} instead of defaulting to extended 100% charging."),
                self._step(2, "Use a stable power source", "select_power_source", "Use a known-good outlet and charger to avoid electrical stress.", ["step_1"]),
                self._step(3, "Control heat", "reduce_heat", "Keep the device ventilated and avoid heavy load while charging.", ["step_2"], "medium"),
                self._step(4, "Monitor the charge window", "monitor_charge", "Watch the battery level until it reaches the target range.", ["step_3"]),
                self._step(5, "Disconnect at the target", "disconnect_when_ready", "Unplug once the target range is reached to avoid unnecessary top-off time.", ["step_4"]),
            ]
        return [
            self._step(1, "Inspect the setup", "inspect_power_setup", f"Check the charger, cable, and outlet you will use for {target}."),
            self._step(2, "Connect safely", "connect_charger", "Attach the charger securely and confirm power is flowing normally.", ["step_1"]),
            self._step(3, "Stabilize conditions", "reduce_heat", "Keep ventilation clear and avoid risky power conditions while charging.", ["step_2"]),
            self._step(4, "Monitor charge and temperature", "monitor_charge", "Watch for normal charging progress and stop if there is heat or instability.", ["step_3"]),
            self._step(5, "Disconnect when appropriate", "disconnect_when_ready", "Unplug once the device has enough charge for the goal.", ["step_4"]),
        ]

    def _task_steps(self, focus: str, target: str) -> List[StrategyStep]:
        if focus == "verification_first":
            return [
                self._step(1, "Define the deliverable", "define_target", f"State what finished looks like for {target}."),
                self._step(2, "Pick a safe first move", "choose_safe_start", "Select the smallest useful action that will move the task forward.", ["step_1"]),
                self._step(3, "Execute one contained step", "execute_step", "Do the first concrete action without expanding scope.", ["step_2"], "medium"),
                self._step(4, "Verify progress", "verify_result", "Check whether that action reduced risk and advanced the task.", ["step_3"]),
                self._step(5, "Decide to continue or stop", "review_next_step", "Either repeat with the next contained step or close the task if enough is done.", ["step_4"]),
            ]
        if focus == "safety_first":
            return [
                self._step(1, "Clarify the task boundary", "define_target", f"Identify the exact task you need to finish for {target}."),
                self._step(2, "List constraints and hazards", "identify_risks", "Call out anything that could make the task unsafe or incomplete.", ["step_1"]),
                self._step(3, "Prepare the safest setup", "prepare_workspace", "Gather what you need and remove obvious risks before acting.", ["step_2"]),
                self._step(4, "Complete the main action", "execute_step", "Do the highest-value safe action for the task.", ["step_3"], "medium"),
                self._step(5, "Check and close", "verify_result", "Confirm the task is finished without creating a new problem.", ["step_4"]),
            ]
        return [
            self._step(1, "Define success", "define_target", f"State the smallest complete version of {target}."),
            self._step(2, "Prepare inputs", "prepare", "Gather the information or tools needed for the next action.", ["step_1"]),
            self._step(3, "Do the highest-value step", "execute_step", "Take the main action that most directly advances completion.", ["step_2"], "medium"),
            self._step(4, "Verify the result", "verify_result", "Check whether the action actually moved the task forward.", ["step_3"]),
            self._step(5, "Close or queue the next step", "close_task", "Finish the task if complete, or define the next contained action.", ["step_4"]),
        ]

    def _tradeoff_steps(self, focus: str, target: str) -> List[StrategyStep]:
        if focus == "decisive_choice":
            return [
                self._step(1, "Name the options", "list_options", f"Write down the options involved in {target}."),
                self._step(2, "Pick decisive criteria", "set_criteria", "Choose the few goal criteria that matter most for the decision.", ["step_1"]),
                self._step(3, "Eliminate poor fits", "eliminate_options", "Remove options that clearly fail the top criteria.", ["step_2"], "medium"),
                self._step(4, "Select the best remaining option", "choose_option", "Choose the option with the strongest overall fit.", ["step_3"]),
                self._step(5, "Take the first follow-through step", "commit_next_step", "Act on the chosen option so the decision becomes useful.", ["step_4"]),
            ]
        if focus == "risk_balanced":
            return [
                self._step(1, "List the options", "list_options", f"Enumerate the choices inside {target}."),
                self._step(2, "Define goals and risks", "set_criteria", "Separate what you want to optimize from what you want to avoid.", ["step_1"]),
                self._step(3, "Score risk and fit", "score_options", "Compare each option against both goals and downside risk.", ["step_2"], "medium"),
                self._step(4, "Choose the safest good fit", "choose_option", "Prefer the option that stays aligned without introducing avoidable risk.", ["step_3"]),
                self._step(5, "Review the tradeoff", "review_decision", "Check that the chosen option still reflects the user's priorities.", ["step_4"]),
            ]
        return [
            self._step(1, "List the options", "list_options", f"Write down the options involved in {target}."),
            self._step(2, "List the goals", "set_criteria", "Turn the user's goals into explicit comparison criteria.", ["step_1"]),
            self._step(3, "Score each option", "score_options", "Rate each option against the goals using the same frame.", ["step_2"]),
            self._step(4, "Choose the best fit", "choose_option", "Select the option with the strongest goal alignment.", ["step_3"]),
            self._step(5, "Plan the next move", "commit_next_step", "Define the first action that follows from the chosen option.", ["step_4"]),
        ]

    def _recovery_steps(self, focus: str, target: str) -> List[StrategyStep]:
        if focus == "diagnostic_reset":
            return [
                self._step(1, "Pause the failing action", "stabilize", f"Stop the current attempt so {target} does not get worse.", risk_level="medium"),
                self._step(2, "Capture the failure signal", "capture_failure", "Note what failed, when it failed, and what changed before it failed.", ["step_1"]),
                self._step(3, "Reset the unstable dependency", "reset_dependency", "Reset the most likely broken dependency before retrying.", ["step_2"], "medium"),
                self._step(4, "Retest one contained step", "retry_step", "Retry only the smallest useful action.", ["step_3"], "medium"),
                self._step(5, "Escalate if still blocked", "escalate", "If the reset failed, switch to a safer fallback instead of repeating the same attempt.", ["step_4"]),
            ]
        if focus == "conservative_retry":
            return [
                self._step(1, "Stop and stabilize", "stabilize", f"Pause work on {target} so the failure does not cascade.", risk_level="medium"),
                self._step(2, "Find the lowest-risk fix", "diagnose", "Identify the smallest change most likely to remove the block.", ["step_1"]),
                self._step(3, "Apply the contained fix", "apply_fix", "Make only that bounded fix before changing anything else.", ["step_2"], "medium"),
                self._step(4, "Retry one step", "retry_step", "Retest one contained action instead of rerunning the full task.", ["step_3"], "medium"),
                self._step(5, "Confirm recovery", "verify_result", "Check that the system recovered and remains stable.", ["step_4"]),
            ]
        return [
            self._step(1, "Record the failure", "capture_failure", f"Describe how {target} failed and what happened right before it."),
            self._step(2, "Identify the likely cause", "diagnose", "Pick the most plausible structural cause first.", ["step_1"]),
            self._step(3, "Apply a safe fix", "apply_fix", "Use the least risky fix that addresses that cause.", ["step_2"], "medium"),
            self._step(4, "Retry in a small scope", "retry_step", "Repeat only the smallest step needed to validate the fix.", ["step_3"], "medium"),
            self._step(5, "Verify and continue", "verify_result", "Continue only if the retry succeeded cleanly.", ["step_4"]),
        ]

    def _transform_steps(self, focus: str, target: str) -> List[StrategyStep]:
        if focus == "minimal_change":
            return [
                self._step(1, "Inspect the current plan", "inspect_plan", f"Review the existing plan for {target}."),
                self._step(2, "Identify the smallest risky segment", "identify_risk", "Find the specific portion that needs revision.", ["step_1"]),
                self._step(3, "Rewrite only that segment", "rewrite_segment", "Reduce risk without changing unrelated structure.", ["step_2"], "medium"),
                self._step(4, "Check intent preservation", "verify_intent", "Confirm the revised plan still serves the original goal.", ["step_3"]),
                self._step(5, "Apply the revision", "apply_revision", "Adopt the updated plan if the risk is lower and intent is preserved.", ["step_4"]),
            ]
        if focus == "verification_first":
            return [
                self._step(1, "Review the original plan", "inspect_plan", f"Read the existing plan for {target}."),
                self._step(2, "Define what must stay true", "define_constraints", "Preserve the key goals and constraints before editing.", ["step_1"]),
                self._step(3, "Rewrite the risky parts", "rewrite_segment", "Adjust the plan where risk is highest.", ["step_2"], "medium"),
                self._step(4, "Verify risk reduction", "verify_revision", "Check that the revised plan is safer than the original.", ["step_3"]),
                self._step(5, "Publish the revised plan", "apply_revision", "Use the safer version for the next action.", ["step_4"]),
            ]
        return [
            self._step(1, "Review the current plan", "inspect_plan", f"Read the plan for {target} and note the current structure."),
            self._step(2, "Locate the risk drivers", "identify_risk", "Find the steps most likely to create avoidable risk.", ["step_1"]),
            self._step(3, "Rewrite for safer sequencing", "rewrite_segment", "Adjust the order and wording so the plan becomes safer.", ["step_2"], "medium"),
            self._step(4, "Check preserved intent", "verify_intent", "Confirm the rewritten plan still serves the same objective.", ["step_3"]),
            self._step(5, "Adopt the safer plan", "apply_revision", "Use the revised plan for the next action.", ["step_4"]),
        ]

    def _score_strategy(
        self,
        *,
        focus: str,
        domain: str,
        steps: List[StrategyStep],
        selected_path: Dict[str, object],
        progress_status: str,
        signature: str,
    ) -> Dict[str, float]:
        goal_alignment = float(selected_path.get("goal_alignment_score", 0.0) or 0.0)
        if goal_alignment <= 0.0:
            goal_alignment = 0.72 if "goal" in focus or "safe" in focus else 0.62
        plausibility = float(selected_path.get("plausibility_score", 0.0) or selected_path.get("grounded_score", 0.0) or 0.0)
        if plausibility <= 0.0:
            plausibility = 0.58 if domain in {"device_charging", "goal_tradeoff"} else 0.46
        completeness = _clamp(len(steps) / 5.0)
        base_risk = {
            "safe_charging": 0.18,
            "battery_longevity": 0.2,
            "fast_charging": 0.48,
            "goal_fit": 0.22,
            "risk_balanced": 0.2,
            "decisive_choice": 0.35,
            "balanced_completion": 0.28,
            "safety_first": 0.18,
            "verification_first": 0.2,
            "failure_recovery": 0.34,
            "conservative_retry": 0.22,
            "diagnostic_reset": 0.3,
            "risk_reduction": 0.2,
            "minimal_change": 0.24,
        }.get(focus, 0.3)
        if progress_status == "stalled":
            base_risk = min(1.0, base_risk + 0.08)
        reinforcement = self._reinforcement_score(signature)
        total_score = (
            goal_alignment * 0.35
            + plausibility * 0.25
            + completeness * 0.15
            + (1.0 - base_risk) * 0.15
            + reinforcement * 0.10
        )
        return {
            "goal_alignment": round(goal_alignment, 4),
            "plausibility": round(plausibility, 4),
            "risk": round(base_risk, 4),
            "completeness": round(completeness, 4),
            "reinforcement": round(reinforcement, 4),
            "total_score": round(_clamp(total_score), 4),
        }

    def _reinforcement_score(self, signature: str) -> float:
        record = self.patterns.get(signature)
        if record is None:
            return 0.5
        total = record.success_count + record.failure_count + record.incomplete_count
        if total <= 0:
            return 0.5
        value = (record.success_count + 0.5) / float(total + 1)
        penalty = min(0.25, record.failure_count * 0.05 + record.incomplete_count * 0.03)
        return _clamp(value - penalty)

    def _strategy_name(self, domain: str, focus: str) -> str:
        labels = {
            "safe_charging": "Strategy A: Safe Charging",
            "fast_charging": "Strategy B: Fast Charging",
            "battery_longevity": "Strategy C: Battery Longevity",
            "goal_fit": "Strategy A: Goal Fit",
            "risk_balanced": "Strategy B: Risk Balanced",
            "decisive_choice": "Strategy C: Decisive Choice",
            "balanced_completion": "Strategy A: Balanced Completion",
            "safety_first": "Strategy B: Safety First",
            "verification_first": "Strategy C: Verification First",
            "failure_recovery": "Strategy A: Recovery Path",
            "conservative_retry": "Strategy B: Conservative Retry",
            "diagnostic_reset": "Strategy C: Diagnostic Reset",
            "risk_reduction": "Strategy A: Risk Reduction",
            "minimal_change": "Strategy B: Minimal Change",
        }
        return labels.get(focus, f"{domain}: {focus}")

    def _tradeoffs(self, domain: str, focus: str) -> List[str]:
        if domain == "device_charging":
            if focus == "fast_charging":
                return ["Faster battery gain", "Higher heat and wear risk than safety-first charging"]
            if focus == "battery_longevity":
                return ["Reduces prolonged high-charge stress", "May stop before a full charge"]
            return ["Lowest safety risk", "Usually slower than speed-focused charging"]
        if domain == "goal_tradeoff":
            if focus == "decisive_choice":
                return ["Fast decision closure", "May use fewer comparison criteria"]
            if focus == "risk_balanced":
                return ["Better downside control", "Can take longer to compare"]
            return ["Strongest goal alignment", "Requires explicit criteria and scoring"]
        if domain == "failure_recovery":
            if focus == "diagnostic_reset":
                return ["Thorough reset of likely causes", "More steps before retrying"]
            if focus == "conservative_retry":
                return ["Limits blast radius during retry", "Can feel slower than a full restart"]
            return ["Fastest path back to progress", "Needs accurate first-cause diagnosis"]
        return {
            "verification_first": ["Strong error detection", "Adds review overhead before closing"],
            "safety_first": ["Lower execution risk", "May move more slowly than the balanced plan"],
            "minimal_change": ["Preserves more of the original plan", "May miss deeper redesign opportunities"],
        }.get(focus, ["Balanced progression", "Not optimized for one single tradeoff"])

    def _strategy_explanation(
        self,
        selected: StrategyObject,
        alternatives: List[StrategyObject],
        progress_status: str,
    ) -> str:
        if not alternatives:
            return f"{selected.name} was selected because it provides the strongest mix of alignment, plausibility, and completeness for the current turn."
        alternative_names = ", ".join(item.name for item in alternatives)
        return (
            f"{selected.name} was selected because it scored highest on total strategy quality for the current turn. "
            f"It keeps risk at {selected.scores.get('risk', 0.0):.2f} while preserving a {selected.scores.get('goal_alignment', 0.0):.2f} goal alignment score. "
            f"Alternative options remained available: {alternative_names}. Current progress status: {progress_status}."
        )
