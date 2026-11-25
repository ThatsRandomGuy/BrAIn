from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import re

from brain_codec import BrainMemory


@dataclass
class ParseResult:
    memory: BrainMemory
    debug: Dict[str, List[str] | str | None]


class BrainParser:
    INTENT_PATTERNS = {
        "question": [
            r"\?$",
            r"\bwhat\b",
            r"\bwhy\b",
            r"\bhow\b",
            r"\bwhen\b",
            r"\bwhere\b",
            r"\bwho\b",
            r"\bcan you\b",
            r"\bcould you\b",
        ],
        "request": [
            r"\bi want you to\b",
            r"\bplease\b",
            r"\bmake\b",
            r"\bcreate\b",
            r"\bbuild\b",
            r"\bgenerate\b",
            r"\bwrite\b",
            r"\bdesign\b",
            r"\bhelp me\b",
        ],
        "inform": [
            r"\bi think\b",
            r"\bi believe\b",
            r"\bi am\b",
            r"\bthis is\b",
            r"\bhere is\b",
        ],
    }

    ACTION_PATTERNS = {
        "create": [r"\bcreate\b", r"\bmake\b", r"\bbuild\b", r"\bgenerate\b", r"\bdesign\b", r"\bwrite\b"],
        "modify": [r"\bchange\b", r"\bmodify\b", r"\bedit\b", r"\bupdate\b", r"\bimprove\b", r"\bfix\b"],
        "store": [r"\bsave\b", r"\bstore\b", r"\bremember\b", r"\bkeep\b"],
        "delete": [r"\bdelete\b", r"\bremove\b", r"\berase\b"],
        "compare": [r"\bcompare\b", r"\bversus\b", r"\bvs\b", r"\bdifference\b"],
        "eat": [r"\beat\b", r"\beating\b"],
        "drink": [r"\bdrink\b", r"\bdrinking\b"],
        "charge": [r"\bcharge\b", r"\bcharging\b"],
        "water": [r"\bwater\b", r"\bwatering\b"],
        "repair": [r"\brepair\b", r"\brepairing\b"],
        "study": [r"\bstudy\b", r"\bstudying\b"],
        "clean": [r"\bclean\b", r"\bcleaning\b"],
        "backup": [r"\bbackup\b", r"\bback up\b", r"\bbacking up\b"],
        "practice": [r"\bpractice\b", r"\bpracticing\b"],
        "cook": [r"\bcook\b", r"\bcooking\b"],
    }

    OBJECT_PATTERNS = {
        "game": [r"\bgame\b"],
        "app": [r"\bapp\b", r"\bapplication\b"],
        "website": [r"\bwebsite\b", r"\bsite\b", r"\bweb app\b", r"\bwebapp\b"],
        "file": [r"\bfile\b", r"\bdocument\b", r"\bdoc\b", r"\bscript\b", r"\bcodebase\b"],
    }

    TOPIC_PATTERNS = {
        "apples": [r"\bapple\b", r"\bapples\b"],
        "brain": [r"\bbrain\b", r"\bbrain system\b", r"\bmemory engine\b"],
        "compression": [r"\bcompression\b", r"\bcompress\b", r"\btoken reduction\b"],
    }

    MODIFIER_PATTERNS = {
        "horror": [r"\bhorror\b", r"\bscary\b", r"\bcreepy\b"],
        "funny": [r"\bfunny\b", r"\bcomedic\b", r"\bcomedy\b"],
        "educational": [r"\beducational\b", r"\blearning\b"],
    }

    PRIORITY_PATTERNS = {
        "high": [r"\burgent\b", r"\bimportant\b", r"\bhigh priority\b", r"\basap\b", r"\bright now\b"],
        "low": [r"\blow priority\b", r"\bwhenever\b", r"\bnot urgent\b", r"\blater\b"],
    }

    TIME_PATTERNS = {
        "now": [r"\bnow\b", r"\bright now\b", r"\btoday\b", r"\bimmediately\b"],
    }

    NEGATION_CUES = [r"\bnot\b", r"\bdo not\b", r"\bdon't\b", r"\bno\b", r"\bwithout\b", r"\bexclude\b"]

    CONDITION_PATTERNS = {
        "unless_requested": [r"\bunless i ask\b", r"\bunless requested\b", r"\bunless i request\b"],
    }

    CONSTRAINT_PATTERNS = {
        "low_hardware": [r"\blow hardware\b", r"\bminimal hardware\b", r"\bsmall hardware\b", r"\bweak hardware\b"],
    }

    STOPWORDS = {
        "a", "an", "the", "this", "that", "these", "those",
        "my", "your", "his", "her", "its", "our", "their",
        "i", "you", "we", "they", "he", "she", "it",
        "to", "for", "of", "on", "in", "at", "by", "with",
        "and", "or", "but", "if", "then", "than", "so",
        "is", "are", "was", "were", "be", "being", "been",
        "want", "need", "would", "like", "should", "could", "can",
        "please", "about", "make", "create", "build", "generate", "write", "design",
        "do", "not", "dont", "don't", "right", "now", "unless", "ask", "otherwise",
    }

    CREATE_VERBS = {"create", "make", "build", "generate", "design", "write"}
    GENERAL_ACTION_VERBS = {
        "eat", "drink", "charge", "water", "repair", "study", "clean", "backup", "practice", "cook",
        "feel", "keep", "finish", "earn", "rest", "get", "miss", "lose", "delay", "think", "solve",
        "complete", "fill", "use", "grow", "harvest", "stay", "save", "restore", "resume", "pass",
        "advance", "graduate", "prepare", "serve", "improve", "perform", "receive", "gain", "seek",
        "preserve", "continue", "find",
    }
    COMPARE_TRAILING_NOISE = (
        "right now",
        "now",
        "today",
        "immediately",
        "please",
    )

    def parse(self, text: str, actor: str = "user") -> ParseResult:
        raw = text.strip()
        normalized = self._normalize(raw)

        memory = BrainMemory(actor=actor, priority="normal")
        debug: Dict[str, List[str] | str | None] = {
            "raw": raw,
            "normalized": normalized,
            "matches": [],
            "fallback_notes": [],
        }

        self._set_confidence(memory, "actor", actor, "default", 1.0)

        memory.intent = self._detect_intent(normalized, debug) or "request"
        if memory.intent:
            if self._has_match(debug, "intent:"):
                self._set_confidence(memory, "intent", memory.intent, "pattern", 0.95)
            else:
                self._set_confidence(memory, "intent", memory.intent, "default", 0.55)

        detected_action = self._detect_label(normalized, self.ACTION_PATTERNS, debug, "action")
        if detected_action:
            memory.action = detected_action
            self._set_confidence(memory, "action", memory.action, "pattern", 0.95)

        detected_object = self._detect_label(normalized, self.OBJECT_PATTERNS, debug, "object")
        if detected_object:
            memory.object = detected_object
            self._set_confidence(memory, "object", memory.object, "pattern", 0.95)

        detected_topic = self._detect_label(normalized, self.TOPIC_PATTERNS, debug, "topic")
        if detected_topic:
            memory.topic = detected_topic
            self._set_confidence(memory, "topic", memory.topic, "pattern", 0.95)

        detected_modifiers = self._detect_multiple_labels(normalized, self.MODIFIER_PATTERNS, debug, "modifier")
        for modifier in detected_modifiers:
            memory.modifiers.append(modifier)
            self._append_confidence(memory, "modifiers", modifier, "pattern", 0.92)

        memory.priority = self._detect_priority(normalized, debug) or "normal"
        if memory.priority == "normal":
            self._set_confidence(memory, "priority", memory.priority, "default", 0.7)
        else:
            self._set_confidence(memory, "priority", memory.priority, "pattern", 0.93)

        detected_time = self._detect_label(normalized, self.TIME_PATTERNS, debug, "time")
        if detected_time:
            memory.time = detected_time
            self._set_confidence(memory, "time", memory.time, "pattern", 0.92)

        detected_negations = self._detect_negations(normalized, debug)
        for negation in detected_negations:
            memory.negations.append(negation)
            self._append_confidence(memory, "negations", negation, "pattern", 0.9)

        detected_conditions = self._detect_multiple_labels(normalized, self.CONDITION_PATTERNS, debug, "condition")
        for condition in detected_conditions:
            memory.conditions.append(condition)
            self._append_confidence(memory, "conditions", condition, "pattern", 0.9)

        detected_constraints = self._detect_multiple_labels(normalized, self.CONSTRAINT_PATTERNS, debug, "constraint")
        for constraint in detected_constraints:
            memory.constraints.append(constraint)
            self._append_confidence(memory, "constraints", constraint, "pattern", 0.9)

        self._apply_fallbacks(normalized, memory, debug)
        self._promote_unknown_candidates(normalized, memory, debug)

        memory.modifiers = self._unique_preserve_order(
            [m for m in memory.modifiers if m not in memory.negations]
        )
        memory.negations = self._unique_preserve_order(memory.negations)
        memory.constraints = self._unique_preserve_order(memory.constraints)
        memory.conditions = self._unique_preserve_order(memory.conditions)

        self._normalize_compare_memory(memory, normalized, debug)
        memory.relations = self._build_relations(memory, normalized, debug)

        return ParseResult(memory=memory, debug=debug)

    # -----------------------------
    # Detection helpers
    # -----------------------------

    def _detect_intent(self, text: str, debug: Dict[str, List[str] | str | None]) -> Optional[str]:
        for pattern in self.INTENT_PATTERNS["question"]:
            if re.search(pattern, text):
                self._log_match(debug, f"intent:question via {pattern}")
                return "question"

        for intent in ("request", "inform"):
            for pattern in self.INTENT_PATTERNS[intent]:
                if re.search(pattern, text):
                    self._log_match(debug, f"intent:{intent} via {pattern}")
                    return intent

        return None

    def _detect_label(
        self,
        text: str,
        pattern_map: Dict[str, List[str]],
        debug: Dict[str, List[str] | str | None],
        label_type: str,
    ) -> Optional[str]:
        best_label: Optional[str] = None
        best_pattern: Optional[str] = None
        best_start: Optional[int] = None
        for label, patterns in pattern_map.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if not match:
                    continue
                if best_start is None or match.start() < best_start:
                    best_label = label
                    best_pattern = pattern
                    best_start = match.start()
        if best_label is not None and best_pattern is not None:
            self._log_match(debug, f"{label_type}:{best_label} via {best_pattern}")
        return best_label

    def _detect_multiple_labels(
        self,
        text: str,
        pattern_map: Dict[str, List[str]],
        debug: Dict[str, List[str] | str | None],
        label_type: str,
    ) -> List[str]:
        found: List[str] = []
        for label, patterns in pattern_map.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    found.append(label)
                    self._log_match(debug, f"{label_type}:{label} via {pattern}")
                    break
        return found

    def _detect_priority(self, text: str, debug: Dict[str, List[str] | str | None]) -> Optional[str]:
        for priority, patterns in self.PRIORITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    self._log_match(debug, f"priority:{priority} via {pattern}")
                    return priority
        return None

    def _detect_negations(self, text: str, debug: Dict[str, List[str] | str | None]) -> List[str]:
        found: List[str] = []

        for cue in self.NEGATION_CUES:
            cue_match = re.search(cue, text)
            if not cue_match:
                continue

            tail = text[cue_match.end():].strip()
            if not tail:
                continue

            for label, patterns in self.MODIFIER_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, tail):
                        if label not in found:
                            found.append(label)
                            self._log_match(debug, f"negation:{label} via cue {cue}")
                        break

            for label, patterns in self.OBJECT_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, tail):
                        if label not in found:
                            found.append(label)
                            self._log_match(debug, f"negation:{label} via cue {cue}")
                        break

            for label, patterns in self.TOPIC_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, tail):
                        if label not in found:
                            found.append(label)
                            self._log_match(debug, f"negation:{label} via cue {cue}")
                        break

        return found

    # -----------------------------
    # Fallback logic
    # -----------------------------

    def _apply_fallbacks(self, text: str, memory: BrainMemory, debug: Dict[str, List[str] | str | None]) -> None:
        if not memory.action:
            inferred_action = self._infer_action_from_shape(text)
            if inferred_action:
                memory.action = inferred_action
                self._set_confidence(memory, "action", inferred_action, "fallback", 0.72)
                self._log_fallback(debug, f"inferred action={inferred_action}")

        if not memory.object:
            inferred_object = self._infer_object_from_action_context(text, memory.action)
            if inferred_object:
                memory.object = inferred_object
                self._set_confidence(memory, "object", inferred_object, "fallback", 0.72)
                self._log_fallback(debug, f"inferred object={inferred_object}")

        if not memory.topic:
            inferred_topic = self._infer_topic_after_about(text)
            if inferred_topic:
                memory.topic = inferred_topic
                self._set_confidence(memory, "topic", inferred_topic, "fallback", 0.74)
                self._log_fallback(debug, f"inferred topic={inferred_topic}")

    def _infer_action_from_shape(self, text: str) -> Optional[str]:
        if text.endswith("?"):
            return "compare" if re.search(r"\bvs\b|\bversus\b|\bcompare\b", text) else None
        if re.search(r"\bfix\b|\bchange\b|\bedit\b|\bupdate\b", text):
            return "modify"
        if re.search(r"\bmake\b|\bbuild\b|\bcreate\b|\bwrite\b|\bgenerate\b", text):
            return "create"
        generic_match = re.search(
            r"\b(eat|drink|charge|water|repair|study|clean|backup|practice|cook|feel|keep|finish|earn|rest|get|miss|lose|delay|think|solve|complete|fill|use|grow|harvest|stay|save|restore|resume|pass|advance|graduate|prepare|serve|improve|perform|receive|gain|seek|preserve|continue|find)\b",
            text,
        )
        if generic_match:
            return generic_match.group(1)
        return None

    def _infer_object_from_action_context(self, text: str, action: str) -> Optional[str]:
        if action == "create":
            if "game" in text:
                return "game"
            if "website" in text or "web app" in text or "webapp" in text:
                return "website"
            if "app" in text:
                return "app"
            if "file" in text or "script" in text:
                return "file"
        if action in self.GENERAL_ACTION_VERBS:
            pattern = rf"\b{re.escape(action)}\s+(?:a|an|the)?\s*([a-z0-9_\- ]{{1,50}})"
            match = re.search(pattern, text)
            if match:
                phrase = match.group(1).strip()
                phrase = re.split(r"\babout\b|\bfor\b|\bwith\b|\bbut\b|\bunless\b|\bif\b|\bto\b|\bso\b", phrase)[0].strip()
                if phrase:
                    tokens = [t for t in phrase.split() if t not in self.STOPWORDS]
                    if tokens:
                        candidate = self._clean_phrase(tokens[-1])
                        if candidate:
                            return candidate
        return None

    def _infer_topic_after_about(self, text: str) -> Optional[str]:
        match = re.search(r"\babout\s+([a-z0-9_\- ]{1,60})", text)
        if not match:
            return None

        candidate = match.group(1).strip()
        candidate = re.split(r"\bbut\b|\band\b|\bwith\b|\bunless\b|\bif\b|\bthat\b|\bfor\b", candidate)[0].strip()
        candidate = re.sub(r"[^a-z0-9_\- ]", "", candidate).strip()
        if not candidate:
            return None

        for known_topic, patterns in self.TOPIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, candidate):
                    return known_topic

        return self._clean_phrase(candidate)

    # -----------------------------
    # Unknown promotion
    # -----------------------------

    def _promote_unknown_candidates(
        self,
        text: str,
        memory: BrainMemory,
        debug: Dict[str, List[str] | str | None],
    ) -> None:
        if not memory.topic:
            promoted_topic = self._promote_unknown_topic(text)
            if promoted_topic:
                memory.topic = promoted_topic
                self._set_confidence(memory, "topic", promoted_topic, "promotion", 0.66)
                self._log_fallback(debug, f"promoted unknown topic={promoted_topic}")

        if not memory.object and memory.action in self.CREATE_VERBS:
            promoted_object = self._promote_unknown_object(text)
            if promoted_object:
                memory.object = promoted_object
                self._set_confidence(memory, "object", promoted_object, "promotion", 0.68)
                self._log_fallback(debug, f"promoted unknown object={promoted_object}")

        promoted_modifiers = self._promote_unknown_modifiers(text, memory.object)
        for modifier in promoted_modifiers:
            if modifier not in memory.modifiers:
                memory.modifiers.append(modifier)
                self._append_confidence(memory, "modifiers", modifier, "promotion", 0.62)
                self._log_fallback(debug, f"promoted unknown modifier={modifier}")

        promoted_negations = self._promote_unknown_negations(text)
        for negation in promoted_negations:
            if negation not in memory.negations:
                memory.negations.append(negation)
                self._append_confidence(memory, "negations", negation, "promotion", 0.64)
                self._log_fallback(debug, f"promoted unknown negation={negation}")

    def _promote_unknown_topic(self, text: str) -> Optional[str]:
        match = re.search(r"\babout\s+([a-z0-9_\- ]{1,60})", text)
        if not match:
            return None

        phrase = match.group(1).strip()
        phrase = re.split(r"\bbut\b|\band\b|\bwith\b|\bunless\b|\bif\b|\bthat\b|\bfor\b", phrase)[0].strip()
        cleaned = self._clean_phrase(phrase)
        if not cleaned:
            return None

        return cleaned

    def _promote_unknown_object(self, text: str) -> Optional[str]:
        match = re.search(
            r"\b(?:create|make|build|generate|design|write)\s+(?:a|an|the)?\s*([a-z0-9_\- ]{1,50})",
            text,
        )
        if not match:
            return None

        phrase = match.group(1).strip()
        phrase = re.split(r"\babout\b|\bfor\b|\bwith\b|\bbut\b|\bunless\b|\bif\b", phrase)[0].strip()
        if not phrase:
            return None

        tokens = [t for t in phrase.split() if t not in self.STOPWORDS]
        if not tokens:
            return None

        candidate = tokens[-1]
        candidate = self._clean_phrase(candidate)
        if not candidate:
            return None

        return candidate

    def _promote_unknown_modifiers(self, text: str, obj: str) -> List[str]:
        if not obj:
            return []

        matches: List[str] = []
        pattern = rf"\b(?:create|make|build|generate|design|write)\s+(?:a|an|the)?\s*([a-z0-9_\- ]+?)\s+{re.escape(obj)}\b"
        match = re.search(pattern, text)
        if not match:
            return []

        prefix = match.group(1).strip()
        if not prefix:
            return []

        raw_tokens = [t.strip() for t in prefix.split() if t.strip()]
        for token in raw_tokens:
            cleaned = self._clean_phrase(token)
            if not cleaned or cleaned in self.STOPWORDS:
                continue
            if cleaned == obj:
                continue
            matches.append(cleaned)

        return self._unique_preserve_order(matches)

    def _promote_unknown_negations(self, text: str) -> List[str]:
        results: List[str] = []

        for cue in self.NEGATION_CUES:
            cue_match = re.search(cue, text)
            if not cue_match:
                continue

            tail = text[cue_match.end():].strip()
            if not tail:
                continue

            tail = re.split(r"\bbut\b|\band\b|\bunless\b|\bif\b|\bthat\b", tail)[0].strip()
            if not tail:
                continue

            tokens = [t for t in tail.split() if t]
            filtered = [
                self._clean_phrase(t)
                for t in tokens[:3]
                if self._clean_phrase(t) and self._clean_phrase(t) not in self.STOPWORDS
            ]

            if not filtered:
                continue

            results.append(filtered[0])

        return self._unique_preserve_order(results)

    # -----------------------------
    # Relationship builder
    # -----------------------------

    def _normalize_compare_memory(
        self,
        memory: BrainMemory,
        text: str,
        debug: Dict[str, List[str] | str | None],
    ) -> None:
        if memory.action != "compare":
            return

        pair = self._extract_compare_pair(text)
        if not pair:
            return

        if memory.object:
            memory.object = ""
            memory.confidence.pop("object", None)
            self._log_fallback(debug, "cleared compare object to preserve relation-first compare semantics")

        if memory.topic:
            memory.topic = ""
            memory.confidence.pop("topic", None)
            self._log_fallback(debug, "cleared compare topic to preserve relation-first compare semantics")

    def _build_relations(
        self,
        memory: BrainMemory,
        text: str,
        debug: Dict[str, List[str] | str | None],
    ) -> List[Dict[str, str]]:
        relations: List[Dict[str, str]] = []

        if memory.object and memory.topic:
            relations.append({
                "type": "about",
                "source": memory.object,
                "target": memory.topic,
            })

        for modifier in memory.modifiers:
            if memory.object:
                relations.append({
                    "type": "has_modifier",
                    "source": memory.object,
                    "target": modifier,
                })

        for negation in memory.negations:
            anchor = memory.object or memory.action
            if anchor:
                relations.append({
                    "type": "negates",
                    "source": anchor,
                    "target": negation,
                })

        for constraint in memory.constraints:
            anchor = memory.object or memory.action
            if anchor:
                relations.append({
                    "type": "constraint",
                    "source": anchor,
                    "target": constraint,
                })

        for condition in memory.conditions:
            anchor = memory.object or memory.action
            if anchor:
                relations.append({
                    "type": "condition",
                    "source": anchor,
                    "target": condition,
                })

        if memory.time:
            anchor = memory.action or memory.object or "memory"
            relations.append({
                "type": "time",
                "source": anchor,
                "target": memory.time,
            })

        if memory.action == "compare":
            pair = self._extract_compare_pair(text)
            if pair:
                left, right = pair
                relations.append({
                    "type": "compare_left",
                    "source": "action",
                    "target": left,
                })
                relations.append({
                    "type": "compare_right",
                    "source": "action",
                    "target": right,
                })
                self._log_match(debug, f"relation:compare_left={left}")
                self._log_match(debug, f"relation:compare_right={right}")

        return relations

    def _extract_compare_pair(self, text: str) -> Optional[tuple[str, str]]:
        patterns = [
            r"\bcompare\s+([a-z0-9_\- ]+?)\s+(?:with|against|to|vs|versus)\s+([a-z0-9_\- ]+)",
            r"\b([a-z0-9_\- ]+?)\s+(?:vs|versus)\s+([a-z0-9_\- ]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue

            left = self._clean_compare_operand(match.group(1))
            right = self._clean_compare_operand(match.group(2))

            if left and right:
                return left, right

        return None

    def _clean_compare_operand(self, phrase: str) -> str:
        cleaned = phrase.lower().strip()
        cleaned = re.sub(r"^compare\s+", "", cleaned).strip()
        cleaned = re.sub(r"[^a-z0-9_\- ]", "", cleaned).strip()

        for noise in self.COMPARE_TRAILING_NOISE:
            if cleaned.endswith(f" {noise}"):
                cleaned = cleaned[: -len(noise)].strip()
            elif cleaned == noise:
                cleaned = ""

        cleaned = re.split(r"\bbut\b|\bunless\b|\bif\b", cleaned)[0].strip()
        if not cleaned:
            return ""

        return self._clean_phrase(cleaned)

    # -----------------------------
    # Confidence helpers
    # -----------------------------

    def _set_confidence(self, memory: BrainMemory, field: str, value: str, source: str, score: float) -> None:
        memory.confidence[field] = {
            "value": value,
            "source": source,
            "score": round(score, 2),
        }

    def _append_confidence(self, memory: BrainMemory, field: str, value: str, source: str, score: float) -> None:
        existing = memory.confidence.get(field)
        entry = {
            "value": value,
            "source": source,
            "score": round(score, 2),
        }

        if existing is None:
            memory.confidence[field] = [entry]
            return

        if isinstance(existing, list):
            if not any(isinstance(item, dict) and item.get("value") == value for item in existing):
                existing.append(entry)
            return

        memory.confidence[field] = [entry]

    @staticmethod
    def _has_match(debug: Dict[str, List[str] | str | None], prefix: str) -> bool:
        matches = debug.get("matches", [])
        return isinstance(matches, list) and any(isinstance(item, str) and item.startswith(prefix) for item in matches)

    # -----------------------------
    # Utility helpers
    # -----------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = text.replace("’", "'")
        text = re.sub(r"\s+", " ", text)
        return text

    def _clean_phrase(self, phrase: str) -> str:
        phrase = phrase.lower().strip()
        phrase = re.sub(r"[^a-z0-9_\- ]", "", phrase)
        phrase = re.sub(r"\s+", " ", phrase).strip()
        if not phrase:
            return ""

        tokens = [t for t in phrase.split() if t and t not in self.STOPWORDS]
        if not tokens:
            return ""

        return "_".join(tokens)

    @staticmethod
    def _unique_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        output: List[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                output.append(item)
        return output

    @staticmethod
    def _log_match(debug: Dict[str, List[str] | str | None], message: str) -> None:
        matches = debug.setdefault("matches", [])
        if isinstance(matches, list):
            matches.append(message)

    @staticmethod
    def _log_fallback(debug: Dict[str, List[str] | str | None], message: str) -> None:
        notes = debug.setdefault("fallback_notes", [])
        if isinstance(notes, list):
            notes.append(message)


def pretty_print_result(result: ParseResult) -> None:
    print("Semantic memory:")
    for key, value in result.memory.to_dict().items():
        print(f"  {key}: {value}")

    print("\nDebug:")
    for key, value in result.debug.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = BrainParser()

    samples = [
        "Create a cozy farming simulator about strawberries.",
        "Please build a surreal dashboard about food waste.",
        "Make a weird roguelike about mushrooms, but not grindy.",
        "Create a game about apples.",
        "Build a website about compression for low hardware right now.",
        "Compare compression_method_a with compression_method_b.",
    ]

    for idx, text in enumerate(samples, start=1):
        print(f"\n--- Sample {idx} ---")
        result = parser.parse(text)
        pretty_print_result(result)
