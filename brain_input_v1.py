from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from brain_binary_codec import BrainBinaryCodec
from brain_codec import BrainCodec, BrainMemory
from brain_hierarchical_pattern_codec import BrainHierarchicalPatternCodec, BrainHierarchicalPatternRegistry
from brain_pattern_codec import BrainPatternCodec, BrainPatternStore, PatternDefinition, PatternSlot
from brain_pattern_evolution import BrainPatternEvolution


BRAIN_INPUT_VERSION = "BrainInputV1"

SINGLE_FIELDS: Tuple[str, ...] = (
    "actor",
    "intent",
    "action",
    "object",
    "topic",
    "time",
    "priority",
)

LIST_FIELDS: Tuple[str, ...] = (
    "modifiers",
    "constraints",
    "negations",
    "conditions",
)

PRIMITIVE_TYPE_TO_SLOT_FIELD: Dict[str, str] = {
    "modifier": "modifiers",
    "constraint": "constraints",
    "negation": "negations",
    "condition": "conditions",
}

STRUCTURAL_TOKEN_STRINGS: Tuple[str, ...] = (
    BRAIN_INPUT_VERSION,
    "SEQ_HYBRID",
    "SEQ_SLOT",
    "SEQ_FLAT",
    "SEQ_GRAPH",
    "SEQ_PATTERN_FIRST",
    "REL",
    "NODE",
    "EDGE",
    "SLOTS_BEGIN",
    "RELATIONS_BEGIN",
    "PRIMITIVE_IDS_BEGIN",
    "SECTION_END",
)

ENCODING_TYPE_TOKENS: Dict[str, str] = {
    "bitpacked": "ENC_BITPACKED",
    "delta": "ENC_DELTA",
    "pattern": "ENC_PATTERN",
    "hierarchical_pattern": "ENC_HIERARCHICAL_PATTERN",
    "hierarchical_subpatterns": "ENC_HIERARCHICAL_SUBPATTERNS",
    "evolved_pattern": "ENC_EVOLVED_PATTERN",
    "evolved_subpatterns": "ENC_EVOLVED_SUBPATTERNS",
}

TOKEN_TO_ENCODING_TYPE = {value: key for key, value in ENCODING_TYPE_TOKENS.items()}


def _sanitize_token_fragment(value: str) -> str:
    characters: List[str] = []
    for char in value.upper():
        if "A" <= char <= "Z" or "0" <= char <= "9":
            characters.append(char)
        else:
            characters.append("_")
    collapsed = "".join(characters).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed or "EMPTY"


@dataclass(frozen=True)
class BrainInputRelation:
    relation_type: str
    source: str
    target: str

    def to_dict(self) -> Dict[str, str]:
        return {"type": self.relation_type, "source": self.source, "target": self.target}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "BrainInputRelation":
        return cls(
            relation_type=str(data.get("type", "")),
            source=str(data.get("source", "")),
            target=str(data.get("target", "")),
        )


@dataclass(frozen=True)
class BrainInputV1:
    encoding_type: str
    actor: str = ""
    intent: str = ""
    action: str = ""
    time: str = ""
    priority: str = "normal"
    pattern_id: Optional[int] = None
    reference_id: Optional[int] = None
    slots: Dict[str, object] = field(default_factory=dict)
    relations: List[BrainInputRelation] = field(default_factory=list)
    subpattern_refs: List[int] = field(default_factory=list)
    primitive_ids: List[int] = field(default_factory=list)
    temporal_score: Optional[float] = None
    version: str = BRAIN_INPUT_VERSION

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "version": self.version,
            "encoding_type": self.encoding_type,
            "actor": self.actor,
            "intent": self.intent,
            "action": self.action,
            "time": self.time,
            "priority": self.priority,
            "slots": {
                "object": str(self.slots.get("object", "")),
                "topic": str(self.slots.get("topic", "")),
                "modifiers": [str(item) for item in self.slots.get("modifiers", [])],
                "constraints": [str(item) for item in self.slots.get("constraints", [])],
                "negations": [str(item) for item in self.slots.get("negations", [])],
                "conditions": [str(item) for item in self.slots.get("conditions", [])],
            },
            "relations": [relation.to_dict() for relation in self.relations],
            "subpattern_refs": list(self.subpattern_refs),
            "primitive_ids": list(self.primitive_ids),
        }
        if self.pattern_id is not None:
            payload["pattern_id"] = self.pattern_id
        if self.reference_id is not None:
            payload["reference_id"] = self.reference_id
        if self.temporal_score is not None:
            payload["temporal_score"] = round(float(self.temporal_score), 4)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "BrainInputV1":
        slots_raw = data.get("slots", {})
        slots = slots_raw if isinstance(slots_raw, dict) else {}
        relations_raw = data.get("relations", [])
        return cls(
            version=str(data.get("version", BRAIN_INPUT_VERSION)),
            encoding_type=str(data.get("encoding_type", "bitpacked")),
            actor=str(data.get("actor", "")),
            intent=str(data.get("intent", "")),
            action=str(data.get("action", "")),
            time=str(data.get("time", "")),
            priority=str(data.get("priority", "normal")),
            pattern_id=int(data["pattern_id"]) if data.get("pattern_id") is not None else None,
            reference_id=int(data["reference_id"]) if data.get("reference_id") is not None else None,
            slots={
                "object": str(slots.get("object", "")),
                "topic": str(slots.get("topic", "")),
                "modifiers": [str(item) for item in slots.get("modifiers", [])] if isinstance(slots.get("modifiers", []), list) else [],
                "constraints": [str(item) for item in slots.get("constraints", [])] if isinstance(slots.get("constraints", []), list) else [],
                "negations": [str(item) for item in slots.get("negations", [])] if isinstance(slots.get("negations", []), list) else [],
                "conditions": [str(item) for item in slots.get("conditions", [])] if isinstance(slots.get("conditions", []), list) else [],
            },
            relations=[BrainInputRelation.from_dict(item) for item in relations_raw if isinstance(item, dict)],
            subpattern_refs=[int(item) for item in data.get("subpattern_refs", []) if isinstance(item, int)] if isinstance(data.get("subpattern_refs", []), list) else [],
            primitive_ids=[int(item) for item in data.get("primitive_ids", []) if isinstance(item, int)] if isinstance(data.get("primitive_ids", []), list) else [],
            temporal_score=round(float(data["temporal_score"]), 4) if data.get("temporal_score") is not None else None,
        )


class BrainSemanticTokenRegistry:
    """
    Persistent deterministic registry for semantic tokens.
    """

    def __init__(self, registry_path: str | Path = "brain_semantic_tokens.json") -> None:
        self.registry_path = Path(registry_path)
        self.primitive_to_token: Dict[str, str] = {}
        self.token_to_primitive: Dict[str, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._next_id = 1
        self._register_structural_tokens()
        self._load()

    def _register_structural_tokens(self) -> None:
        for token in STRUCTURAL_TOKEN_STRINGS + tuple(ENCODING_TYPE_TOKENS.values()):
            self._ensure_token_id(token)

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        primitive_to_token = data.get("primitive_to_token", {})
        token_to_id = data.get("token_to_id", {})
        if isinstance(primitive_to_token, dict):
            for primitive, token in primitive_to_token.items():
                primitive_key = str(primitive)
                token_value = str(token)
                self.primitive_to_token[primitive_key] = token_value
                self.token_to_primitive[token_value] = primitive_key
                self._ensure_token_id(token_value)
        if isinstance(token_to_id, dict):
            pairs: List[Tuple[str, int]] = []
            for token, token_id in token_to_id.items():
                try:
                    pairs.append((str(token), int(token_id)))
                except (TypeError, ValueError):
                    continue
            for token, token_id in sorted(pairs, key=lambda item: item[1]):
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            if self.id_to_token:
                self._next_id = max(self.id_to_token) + 1
            for token in list(self.primitive_to_token.values()) + list(STRUCTURAL_TOKEN_STRINGS) + list(ENCODING_TYPE_TOKENS.values()):
                self._ensure_token_id(token)

    def save(self) -> None:
        payload = {
            "primitive_to_token": {primitive: self.primitive_to_token[primitive] for primitive in sorted(self.primitive_to_token)},
            "token_to_id": {token: self.token_to_id[token] for token in sorted(self.token_to_id, key=lambda value: self.token_to_id[value])},
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _ensure_token_id(self, token: str) -> int:
        token = str(token)
        if token in self.token_to_id:
            return self.token_to_id[token]
        token_id = self._next_id
        self._next_id += 1
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return token_id

    def ensure_token(self, token: str) -> str:
        self._ensure_token_id(token)
        return token

    def ensure_primitive_token(self, primitive: str) -> str:
        if primitive in self.primitive_to_token:
            token = self.primitive_to_token[primitive]
            self._ensure_token_id(token)
            return token
        primitive_type, _, value = primitive.partition(":")
        prefix = _sanitize_token_fragment(primitive_type)
        fragment = _sanitize_token_fragment(value)
        candidate = f"{prefix}_{fragment}"
        if candidate in self.token_to_primitive and self.token_to_primitive[candidate] != primitive:
            candidate = f"{candidate}__{sum(ord(char) for char in primitive) % 100000:05d}"
        self.primitive_to_token[primitive] = candidate
        self.token_to_primitive[candidate] = primitive
        self._ensure_token_id(candidate)
        return candidate

    def lookup_primitive(self, token: str) -> Optional[str]:
        return self.token_to_primitive.get(token)

    def token_id(self, token: str) -> int:
        return self._ensure_token_id(token)


class BrainInputCodec:
    """
    Deterministic BrainInputV1 builder and reversible sequence codec.

    This layer sits above BrAIn's existing encodings. It turns structured
    semantic memory into a model-consumable semantic sequence without
    bypassing the current memory architecture.
    """

    def __init__(
        self,
        codec: BrainCodec,
        binary_codec: BrainBinaryCodec,
        token_registry: Optional[BrainSemanticTokenRegistry] = None,
        pattern_codec: Optional[BrainPatternCodec] = None,
        pattern_store: Optional[BrainPatternStore] = None,
        hierarchical_codec: Optional[BrainHierarchicalPatternCodec] = None,
        hierarchical_registry: Optional[BrainHierarchicalPatternRegistry] = None,
        pattern_evolution: Optional[BrainPatternEvolution] = None,
    ) -> None:
        self.codec = codec
        self.binary_codec = binary_codec
        self.token_registry = token_registry or BrainSemanticTokenRegistry()
        self.pattern_codec = pattern_codec
        self.pattern_store = pattern_store
        self.hierarchical_codec = hierarchical_codec
        self.hierarchical_registry = hierarchical_registry
        self.pattern_evolution = pattern_evolution

    def build_brain_input(
        self,
        memory: BrainMemory,
        encoding_type: str,
        pattern_id: Optional[int] = None,
        reference_id: Optional[int] = None,
        subpattern_refs: Optional[Sequence[int]] = None,
        primitive_ids: Optional[Sequence[int]] = None,
        temporal_score: Optional[float] = None,
    ) -> BrainInputV1:
        resolved = BrainCodec.resolve_memory(memory)
        if primitive_ids is None:
            primitive_ids = self._collect_primitive_ids(resolved)
        slots = {
            "object": resolved.object,
            "topic": resolved.topic,
            "modifiers": list(resolved.modifiers),
            "constraints": list(resolved.constraints),
            "negations": list(resolved.negations),
            "conditions": list(resolved.conditions),
        }
        relations = [
            BrainInputRelation(
                relation_type=str(relation.get("type", "")),
                source=str(relation.get("source", "")),
                target=str(relation.get("target", "")),
            )
            for relation in resolved.relations
        ]
        return BrainInputV1(
            encoding_type=encoding_type,
            actor=resolved.actor,
            intent=resolved.intent,
            action=resolved.action,
            time=resolved.time or "",
            priority=resolved.priority,
            pattern_id=pattern_id,
            reference_id=reference_id,
            slots=slots,
            relations=relations,
            subpattern_refs=sorted(int(item) for item in (subpattern_refs or [])),
            primitive_ids=sorted(set(int(item) for item in primitive_ids)),
            temporal_score=round(float(temporal_score), 4) if temporal_score is not None else None,
        )

    def memory_to_brain_input(
        self,
        memory: BrainMemory,
        encoding_type: str,
        pattern_id: Optional[int] = None,
        reference_id: Optional[int] = None,
        subpattern_refs: Optional[Sequence[int]] = None,
        temporal_score: Optional[float] = None,
    ) -> BrainInputV1:
        return self.build_brain_input(
            memory=memory,
            encoding_type=encoding_type,
            pattern_id=pattern_id,
            reference_id=reference_id,
            subpattern_refs=subpattern_refs,
            temporal_score=temporal_score,
        )

    def brain_input_to_sequence(
        self,
        brain_input: BrainInputV1,
        strategy: str = "hybrid",
        current_sequence: Optional[int] = None,
    ) -> List[str]:
        strategy_key = strategy.strip().lower().replace("-", "_") or "hybrid"
        if strategy_key not in {"hybrid", "slot", "flat", "graph", "pattern_first"}:
            raise ValueError(f"Unsupported BrainInput sequence strategy: {strategy}")
        active_input = (
            self.brain_input_to_pattern_first(brain_input, current_sequence=current_sequence)
            if strategy_key == "pattern_first"
            else brain_input
        )
        header = [BRAIN_INPUT_VERSION, f"SEQ_{strategy_key.upper()}"]
        header.append(self._encoding_type_token(active_input.encoding_type))
        if active_input.pattern_id is not None:
            header.append(f"PATTERN_{active_input.pattern_id}")
        if active_input.reference_id is not None:
            header.append(f"REF_{active_input.reference_id}")
        if active_input.pattern_id is None:
            header.extend(f"SUBP_{ref}" for ref in active_input.subpattern_refs)

        compact_tokens = self._compact_pattern_tokens(active_input)
        if compact_tokens is not None:
            if active_input.temporal_score is not None:
                compact_tokens.append(f"TEMP_{active_input.temporal_score:.4f}")
            return header + compact_tokens

        if strategy_key == "pattern_first":
            return self._encode_hybrid(active_input, header)
        if strategy_key == "graph":
            return self._encode_graph(active_input, header)
        if strategy_key == "slot":
            return self._encode_slot(active_input, header)
        if strategy_key == "flat":
            return self._encode_flat(active_input, header)
        return self._encode_hybrid(active_input, header)

    def sequence_to_brain_input(
        self,
        sequence: Sequence[str],
        strategy: str = "hybrid",
    ) -> BrainInputV1:
        tokens = [str(token) for token in sequence]
        if not tokens or tokens[0] != BRAIN_INPUT_VERSION:
            raise ValueError("BrainInput sequence must begin with BrainInputV1 marker.")
        strategy_key = strategy.strip().lower().replace("-", "_") or "hybrid"
        if strategy_key not in {"hybrid", "slot", "flat", "graph", "pattern_first"}:
            raise ValueError(f"Unsupported BrainInput sequence strategy: {strategy}")
        if len(tokens) < 3:
            raise ValueError("BrainInput sequence is missing required header tokens.")

        encoding_type = TOKEN_TO_ENCODING_TYPE.get(tokens[2], "bitpacked")
        payload_tokens = tokens[3:]
        pattern_id: Optional[int] = None
        reference_id: Optional[int] = None
        subpattern_refs: List[int] = []
        primitive_ids: List[int] = []
        temporal_score: Optional[float] = None
        compact_slot_values: Dict[str, str] = {}
        scalars: Dict[str, str] = {
            "actor": "",
            "intent": "",
            "action": "",
            "object": "",
            "topic": "",
            "time": "",
            "priority": "normal",
        }
        list_slots: Dict[str, List[str]] = {
            "modifiers": [],
            "constraints": [],
            "negations": [],
            "conditions": [],
        }
        relations: List[BrainInputRelation] = []

        index = 0
        section_mode = "fields"
        while index < len(payload_tokens):
            token = payload_tokens[index]

            if token.startswith("PATTERN_"):
                pattern_id = int(token.split("_", 1)[1])
                index += 1
                continue
            if token.startswith("REF_"):
                reference_id = int(token.split("_", 1)[1])
                index += 1
                continue
            if token.startswith("SUBP_"):
                subpattern_refs.append(int(token.split("_", 1)[1]))
                index += 1
                continue
            if token.startswith("PID_"):
                primitive_ids.append(int(token.split("_", 1)[1]))
                index += 1
                continue
            if token.startswith("TEMP_"):
                temporal_score = round(float(token.split("_", 1)[1]), 4)
                index += 1
                continue
            if token.startswith("SLOT_"):
                slot_name = token.split("_", 1)[1].lower()
                if index + 1 >= len(payload_tokens):
                    raise ValueError("Compact BrainInput slot token is missing a value token.")
                slot_token = payload_tokens[index + 1]
                primitive = self.token_registry.lookup_primitive(slot_token)
                if primitive is not None and ":" in primitive:
                    _, value = primitive.split(":", 1)
                    compact_slot_values[slot_name] = value
                index += 2
                continue

            if token in {"SLOTS_BEGIN", "RELATIONS_BEGIN", "PRIMITIVE_IDS_BEGIN"}:
                section_mode = token
                index += 1
                continue
            if token == "SECTION_END":
                section_mode = "fields"
                index += 1
                continue

            if strategy_key == "graph" and token == "EDGE":
                relation, index = self._decode_relation_triplet(payload_tokens, index + 1)
                relations.append(relation)
                continue
            if token == "REL":
                relation, index = self._decode_relation_triplet(payload_tokens, index + 1)
                relations.append(relation)
                continue
            if strategy_key == "graph" and token == "NODE":
                if index + 1 >= len(payload_tokens):
                    raise ValueError("NODE token must be followed by a primitive token.")
                self._apply_primitive_token(payload_tokens[index + 1], scalars, list_slots)
                index += 2
                continue

            if section_mode == "PRIMITIVE_IDS_BEGIN":
                index += 1
                continue

            self._apply_primitive_token(token, scalars, list_slots)
            index += 1

        compact_reconstructed = self._reconstruct_from_pattern_reference(
            encoding_type=encoding_type,
            pattern_id=pattern_id,
            subpattern_refs=subpattern_refs,
            slot_values=compact_slot_values,
            reference_id=reference_id,
            temporal_score=temporal_score,
        )
        if compact_reconstructed is not None:
            return compact_reconstructed

        return BrainInputV1(
            encoding_type=encoding_type,
            actor=scalars["actor"],
            intent=scalars["intent"],
            action=scalars["action"],
            time=scalars["time"],
            priority=scalars["priority"],
            pattern_id=pattern_id,
            reference_id=reference_id,
            slots={
                "object": scalars["object"],
                "topic": scalars["topic"],
                "modifiers": list_slots["modifiers"],
                "constraints": list_slots["constraints"],
                "negations": list_slots["negations"],
                "conditions": list_slots["conditions"],
            },
            relations=relations,
            subpattern_refs=sorted(subpattern_refs),
            primitive_ids=sorted(set(primitive_ids)),
            temporal_score=temporal_score,
        )

    def brain_input_to_grid(
        self,
        brain_input: BrainInputV1,
        strategy: str = "hybrid",
        width: int = 8,
        current_sequence: Optional[int] = None,
    ) -> Dict[str, object]:
        sequence = self.brain_input_to_sequence(
            brain_input,
            strategy=strategy,
            current_sequence=current_sequence,
        )
        token_ids = [self.token_registry.token_id(token) for token in sequence]
        rows = (len(token_ids) + width - 1) // width
        grid = [[0 for _ in range(width)] for _ in range(rows)]
        for index, token_id in enumerate(token_ids):
            row = index // width
            col = index % width
            grid[row][col] = token_id
        return {
            "width": width,
            "height": rows,
            "length": len(token_ids),
            "strategy": strategy,
            "grid": grid,
        }

    def grid_to_brain_input(self, payload: Dict[str, object]) -> BrainInputV1:
        grid_raw = payload.get("grid", [])
        if not isinstance(grid_raw, list):
            raise ValueError("BrainInput grid payload must include a grid list.")
        length = int(payload.get("length", 0))
        strategy = str(payload.get("strategy", "hybrid"))
        token_ids: List[int] = []
        for row in grid_raw:
            if not isinstance(row, list):
                continue
            for item in row:
                try:
                    token_ids.append(int(item))
                except (TypeError, ValueError):
                    token_ids.append(0)
        sequence = [
            self.token_registry.id_to_token[token_id]
            for token_id in token_ids[:length]
            if token_id in self.token_registry.id_to_token and token_id > 0
        ]
        return self.sequence_to_brain_input(sequence, strategy=strategy)

    def _collect_primitive_ids(self, memory: BrainMemory) -> List[int]:
        self.binary_codec.sync_from_codec()
        bundle = self.codec.encode_memory_bundle(memory)
        encoded = self.binary_codec.encode_jsonable_bundle(bundle)
        primitive_ids = [
            int(item)
            for item in list(encoded.core_ids) + list(encoded.relation_ids)
            if int(item) > 0
        ]
        return sorted(set(primitive_ids))

    def brain_input_to_memory(self, brain_input: BrainInputV1) -> BrainMemory:
        return BrainMemory(
            actor=brain_input.actor,
            intent=brain_input.intent,
            action=brain_input.action,
            object=str(brain_input.slots.get("object", "")),
            topic=str(brain_input.slots.get("topic", "")),
            modifiers=[str(item) for item in brain_input.slots.get("modifiers", [])],
            negations=[str(item) for item in brain_input.slots.get("negations", [])],
            constraints=[str(item) for item in brain_input.slots.get("constraints", [])],
            conditions=[str(item) for item in brain_input.slots.get("conditions", [])],
            relations=[relation.to_dict() for relation in brain_input.relations],
            time=brain_input.time or None,
            priority=brain_input.priority,
        )

    def brain_input_to_pattern_first(
        self,
        brain_input: BrainInputV1,
        current_sequence: Optional[int] = None,
    ) -> BrainInputV1:
        memory = self.brain_input_to_memory(brain_input)
        selection = self._select_pattern_first_reference(
            memory=memory,
            current_sequence=current_sequence,
        )
        if selection is None:
            return self.build_brain_input(
                memory=memory,
                encoding_type=brain_input.encoding_type,
                primitive_ids=brain_input.primitive_ids,
                temporal_score=brain_input.temporal_score,
            )
        return self.build_brain_input(
            memory=memory,
            encoding_type=str(selection["encoding_type"]),
            pattern_id=(
                int(selection["pattern_id"])
                if selection.get("pattern_id") is not None
                else None
            ),
            subpattern_refs=[
                int(item)
                for item in selection.get("subpattern_refs", [])
                if isinstance(item, int)
            ],
            primitive_ids=brain_input.primitive_ids,
            temporal_score=brain_input.temporal_score,
        )

    def _collect_primitives(self, memory: BrainMemory) -> List[str]:
        primitives: List[str] = []
        for field_name in ("actor", "intent", "action", "object", "topic", "time", "priority"):
            value = getattr(memory, field_name)
            if value:
                primitives.append(self.codec.make_primitive(field_name, value))
        for modifier in memory.modifiers:
            primitives.append(self.codec.make_primitive("modifier", modifier))
        for constraint in memory.constraints:
            primitives.append(self.codec.make_primitive("constraint", constraint))
        for negation in memory.negations:
            primitives.append(self.codec.make_primitive("negation", negation))
        for condition in memory.conditions:
            primitives.append(self.codec.make_primitive("condition", condition))
        for relation in memory.relations:
            relation_type = str(relation.get("type", ""))
            source = str(relation.get("source", ""))
            target = str(relation.get("target", ""))
            if relation_type:
                primitives.append(self.codec.make_primitive("relation_type", relation_type))
            if source:
                primitives.append(self._infer_endpoint_primitive(source, memory, relation_type, "source"))
            if target:
                primitives.append(self._infer_endpoint_primitive(target, memory, relation_type, "target"))
        return primitives

    def _compact_pattern_tokens(self, brain_input: BrainInputV1) -> Optional[List[str]]:
        reference = self._resolve_pattern_reference(
            encoding_type=brain_input.encoding_type,
            pattern_id=brain_input.pattern_id,
            subpattern_refs=brain_input.subpattern_refs,
        )
        if reference is None:
            return None

        memory = self.brain_input_to_memory(brain_input)
        slot_values = reference["extractor"](memory)
        ordered_slots: Sequence[PatternSlot] = reference["variable_slots"]
        tokens: List[str] = []
        for slot in sorted(ordered_slots, key=lambda item: item.slot_index):
            value = str(slot_values.get(slot.slot_name, "") or "")
            if not value and slot.optional:
                continue
            if not value:
                return None
            slot_marker = self.token_registry.ensure_token(f"SLOT_{slot.slot_name.upper()}")
            tokens.append(slot_marker)
            tokens.append(self._primitive_token(self.codec.make_primitive(slot.primitive_type, value)))
        return tokens

    def _reconstruct_from_pattern_reference(
        self,
        encoding_type: str,
        pattern_id: Optional[int],
        subpattern_refs: Sequence[int],
        slot_values: Dict[str, str],
        reference_id: Optional[int],
        temporal_score: Optional[float],
    ) -> Optional[BrainInputV1]:
        if not slot_values and pattern_id is None and not subpattern_refs:
            return None
        reference = self._resolve_pattern_reference(
            encoding_type=encoding_type,
            pattern_id=pattern_id,
            subpattern_refs=subpattern_refs,
        )
        if reference is None:
            return None
        memory = reference["reconstructor"](slot_values)
        return self.build_brain_input(
            memory=memory,
            encoding_type=encoding_type,
            pattern_id=pattern_id,
            reference_id=reference_id,
            subpattern_refs=reference.get("subpattern_refs", subpattern_refs),
            temporal_score=temporal_score,
        )

    def _resolve_pattern_reference(
        self,
        encoding_type: str,
        pattern_id: Optional[int],
        subpattern_refs: Sequence[int],
    ) -> Optional[Dict[str, object]]:
        if encoding_type == "pattern" and pattern_id is not None and self.pattern_codec is not None and self.pattern_store is not None:
            pattern = self.pattern_store.get(pattern_id)
            return {
                "variable_slots": pattern.variable_slots,
                "subpattern_refs": [],
                "extractor": lambda memory: self.pattern_codec._extract_slot_values(memory, pattern.variable_slots),
                "reconstructor": lambda slot_values: self.pattern_codec._reconstruct_memory(pattern, slot_values),
            }

        if encoding_type in {"hierarchical_pattern", "evolved_pattern"} and pattern_id is not None and self.hierarchical_codec is not None and self.hierarchical_registry is not None:
            pattern = self.hierarchical_registry.get_pattern(pattern_id)
            composed = self.hierarchical_codec.compose_pattern(pattern, self.hierarchical_registry)
            return {
                "variable_slots": composed.variable_slots,
                "subpattern_refs": list(pattern.subpattern_ids),
                "extractor": lambda memory: self.hierarchical_codec._extract_slot_values(memory, composed.variable_slots),
                "reconstructor": lambda slot_values: self.hierarchical_codec._reconstruct_from_composed(composed, slot_values),
            }

        if encoding_type in {"hierarchical_subpatterns", "evolved_subpatterns"} and subpattern_refs and self.hierarchical_codec is not None and self.hierarchical_registry is not None:
            subpatterns = [self.hierarchical_registry.get_subpattern(subpattern_id) for subpattern_id in subpattern_refs]
            composed = self.hierarchical_codec._compose_subpatterns(subpatterns)
            return {
                "variable_slots": composed.variable_slots,
                "subpattern_refs": list(subpattern_refs),
                "extractor": lambda memory: self.hierarchical_codec._extract_slot_values(memory, composed.variable_slots),
                "reconstructor": lambda slot_values: self.hierarchical_codec._reconstruct_from_composed(composed, slot_values),
            }
        return None

    def _select_pattern_first_reference(
        self,
        memory: BrainMemory,
        current_sequence: Optional[int] = None,
    ) -> Optional[Dict[str, object]]:
        memory_bundle = self.codec.encode_memory_bundle(memory)
        for selector in (
            lambda: self._select_hierarchical_pattern_reference(memory_bundle),
            lambda: self._select_evolved_pattern_reference(memory_bundle, current_sequence=current_sequence),
            lambda: self._select_flat_pattern_reference(memory_bundle),
        ):
            reference = selector()
            if reference is None:
                continue
            if self._is_lossless_pattern_reference(
                memory=memory,
                encoding_type=str(reference["encoding_type"]),
                pattern_id=(
                    int(reference["pattern_id"])
                    if reference.get("pattern_id") is not None
                    else None
                ),
                subpattern_refs=[
                    int(item)
                    for item in reference.get("subpattern_refs", [])
                    if isinstance(item, int)
                ],
            ):
                return reference
        return None

    def _select_hierarchical_pattern_reference(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
    ) -> Optional[Dict[str, object]]:
        if self.hierarchical_codec is None or self.hierarchical_registry is None:
            return None
        match = self.hierarchical_codec.find_best_pattern(memory_bundle, self.hierarchical_registry)
        if match.pattern_id is not None:
            pattern = self.hierarchical_registry.get_pattern(match.pattern_id)
            return {
                "encoding_type": "hierarchical_pattern",
                "pattern_id": pattern.pattern_id,
                "subpattern_refs": list(pattern.subpattern_ids),
                "reasons": list(match.reasons),
            }

        candidate = self.hierarchical_codec.extract_subpatterns(memory_bundle)
        subpattern_refs: List[int] = []
        for subpattern in candidate.subpatterns:
            existing = self.hierarchical_registry.find_subpattern_by_fingerprint(
                subpattern.fingerprint,
                active_only=True,
            )
            if existing is None:
                return None
            subpattern_refs.append(existing.subpattern_id)
        if not subpattern_refs:
            return None
        return {
            "encoding_type": "hierarchical_subpatterns",
            "pattern_id": None,
            "subpattern_refs": subpattern_refs,
            "reasons": list(match.reasons),
        }

    def _select_evolved_pattern_reference(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        current_sequence: Optional[int] = None,
    ) -> Optional[Dict[str, object]]:
        if self.pattern_evolution is None:
            return None
        decision = self.pattern_evolution.recommend_pattern(
            memory_bundle=memory_bundle,
            current_sequence=current_sequence,
            include_subpatterns=True,
            update_usage=False,
        )
        if decision is None or decision.mode == "full":
            return None
        if decision.mode == "pattern" and decision.pattern_id is not None:
            pattern = self.hierarchical_registry.get_pattern(decision.pattern_id) if self.hierarchical_registry is not None else None
            return {
                "encoding_type": "evolved_pattern",
                "pattern_id": decision.pattern_id,
                "subpattern_refs": list(pattern.subpattern_ids) if pattern is not None else [],
                "reasons": list(decision.reasons),
            }
        if decision.mode == "subpatterns" and decision.subpattern_ids:
            return {
                "encoding_type": "evolved_subpatterns",
                "pattern_id": None,
                "subpattern_refs": [int(item) for item in decision.subpattern_ids],
                "reasons": list(decision.reasons),
            }
        return None

    def _select_flat_pattern_reference(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
    ) -> Optional[Dict[str, object]]:
        if self.pattern_codec is None or self.pattern_store is None:
            return None
        match = self.pattern_codec.find_best_pattern(
            memory_bundle=memory_bundle,
            known_patterns=self.pattern_store.all_patterns(),
        )
        if match is None:
            return None
        return {
            "encoding_type": "pattern",
            "pattern_id": match.pattern_id,
            "subpattern_refs": [],
            "reasons": list(match.reasons),
        }

    def _is_lossless_pattern_reference(
        self,
        memory: BrainMemory,
        encoding_type: str,
        pattern_id: Optional[int],
        subpattern_refs: Sequence[int],
    ) -> bool:
        reference = self._resolve_pattern_reference(
            encoding_type=encoding_type,
            pattern_id=pattern_id,
            subpattern_refs=subpattern_refs,
        )
        if reference is None:
            return False
        try:
            slot_values = reference["extractor"](memory)
            reconstructed = reference["reconstructor"](slot_values)
        except Exception:
            return False
        return self.codec.encode_memory_bundle(reconstructed) == self.codec.encode_memory_bundle(memory)

    def _encode_hybrid(self, brain_input: BrainInputV1, header: List[str]) -> List[str]:
        tokens = list(header)
        tokens.extend(self._field_tokens(brain_input))
        for relation in brain_input.relations:
            tokens.append("REL")
            tokens.extend(self._relation_tokens(relation, brain_input))
        if brain_input.temporal_score is not None:
            tokens.append(f"TEMP_{brain_input.temporal_score:.4f}")
        return tokens

    def _encode_flat(self, brain_input: BrainInputV1, header: List[str]) -> List[str]:
        tokens = list(header)
        tokens.extend(self._field_tokens(brain_input))
        for relation in brain_input.relations:
            tokens.append("REL")
            tokens.extend(self._relation_tokens(relation, brain_input))
        if brain_input.temporal_score is not None:
            tokens.append(f"TEMP_{brain_input.temporal_score:.4f}")
        return tokens

    def _encode_slot(self, brain_input: BrainInputV1, header: List[str]) -> List[str]:
        tokens = list(header)
        tokens.append("SLOTS_BEGIN")
        tokens.extend(self._field_tokens(brain_input))
        tokens.append("SECTION_END")
        tokens.append("RELATIONS_BEGIN")
        for relation in brain_input.relations:
            tokens.append("REL")
            tokens.extend(self._relation_tokens(relation, brain_input))
        tokens.append("SECTION_END")
        if brain_input.temporal_score is not None:
            tokens.append(f"TEMP_{brain_input.temporal_score:.4f}")
        if brain_input.primitive_ids:
            tokens.append("PRIMITIVE_IDS_BEGIN")
            tokens.extend(f"PID_{primitive_id}" for primitive_id in brain_input.primitive_ids)
            tokens.append("SECTION_END")
        return tokens

    def _encode_graph(self, brain_input: BrainInputV1, header: List[str]) -> List[str]:
        tokens = list(header)
        for token in self._field_tokens(brain_input):
            tokens.extend(["NODE", token])
        for relation in brain_input.relations:
            tokens.append("EDGE")
            tokens.extend(self._relation_tokens(relation, brain_input))
        if brain_input.temporal_score is not None:
            tokens.append(f"TEMP_{brain_input.temporal_score:.4f}")
        return tokens

    def _field_tokens(self, brain_input: BrainInputV1) -> List[str]:
        tokens: List[str] = []
        scalar_fields = {
            "actor": brain_input.actor,
            "intent": brain_input.intent,
            "action": brain_input.action,
            "priority": "" if brain_input.priority == "normal" else brain_input.priority,
            "time": brain_input.time,
            "object": str(brain_input.slots.get("object", "")),
            "topic": str(brain_input.slots.get("topic", "")),
        }
        for field_name in ("actor", "intent", "action", "priority", "time", "object", "topic"):
            value = scalar_fields[field_name]
            if value:
                tokens.append(self._primitive_token(self.codec.make_primitive(field_name, value)))
        for field_name, primitive_type in (
            ("modifiers", "modifier"),
            ("constraints", "constraint"),
            ("negations", "negation"),
            ("conditions", "condition"),
        ):
            for value in brain_input.slots.get(field_name, []):
                tokens.append(self._primitive_token(self.codec.make_primitive(primitive_type, str(value))))
        return tokens

    def _relation_tokens(
        self,
        relation: BrainInputRelation,
        brain_input: BrainInputV1,
    ) -> List[str]:
        proxy_memory = BrainMemory(
            actor=brain_input.actor,
            intent=brain_input.intent,
            action=brain_input.action,
            object=str(brain_input.slots.get("object", "")),
            topic=str(brain_input.slots.get("topic", "")),
            modifiers=[str(item) for item in brain_input.slots.get("modifiers", [])],
            constraints=[str(item) for item in brain_input.slots.get("constraints", [])],
            negations=[str(item) for item in brain_input.slots.get("negations", [])],
            conditions=[str(item) for item in brain_input.slots.get("conditions", [])],
            time=brain_input.time or None,
            priority=brain_input.priority,
            relations=[],
        )
        relation_type_token = self._primitive_token(self.codec.make_primitive("relation_type", relation.relation_type))
        source_token = self._primitive_token(
            self._infer_endpoint_primitive(relation.source, proxy_memory, relation.relation_type, "source")
        )
        target_token = self._primitive_token(
            self._infer_endpoint_primitive(relation.target, proxy_memory, relation.relation_type, "target")
        )
        return [relation_type_token, source_token, target_token]

    def _decode_relation_triplet(
        self,
        tokens: Sequence[str],
        index: int,
    ) -> Tuple[BrainInputRelation, int]:
        if index + 2 >= len(tokens):
            raise ValueError("Relation token triplet is incomplete.")
        relation_type_token = tokens[index]
        source_token = tokens[index + 1]
        target_token = tokens[index + 2]
        relation_primitive = self.token_registry.lookup_primitive(relation_type_token)
        source_primitive = self.token_registry.lookup_primitive(source_token)
        target_primitive = self.token_registry.lookup_primitive(target_token)
        if relation_primitive is None or not relation_primitive.startswith("relation_type:"):
            raise ValueError(f"Invalid relation token: {relation_type_token}")
        relation_type = relation_primitive.split(":", 1)[1]
        source = source_primitive.split(":", 1)[1] if source_primitive else ""
        target = target_primitive.split(":", 1)[1] if target_primitive else ""
        return BrainInputRelation(relation_type=relation_type, source=source, target=target), index + 3

    def _apply_primitive_token(
        self,
        token: str,
        scalars: Dict[str, str],
        list_slots: Dict[str, List[str]],
    ) -> None:
        primitive = self.token_registry.lookup_primitive(token)
        if primitive is None or ":" not in primitive:
            return
        primitive_type, value = primitive.split(":", 1)
        if primitive_type in scalars:
            scalars[primitive_type] = value
            return
        slot_field = PRIMITIVE_TYPE_TO_SLOT_FIELD.get(primitive_type)
        if slot_field is not None:
            list_slots[slot_field].append(value)

    def _primitive_token(self, primitive: str) -> str:
        token = self.token_registry.ensure_primitive_token(primitive)
        return self.token_registry.ensure_token(token)

    def _encoding_type_token(self, encoding_type: str) -> str:
        token = ENCODING_TYPE_TOKENS.get(encoding_type, "ENC_BITPACKED")
        return self.token_registry.ensure_token(token)

    def _infer_endpoint_primitive(
        self,
        value: str,
        memory: BrainMemory,
        relation_type: str,
        endpoint_role: str,
    ) -> str:
        if not value:
            return self.codec.make_primitive("topic", "unknown")
        if value == memory.actor:
            return self.codec.make_primitive("actor", value)
        if value == memory.intent:
            return self.codec.make_primitive("intent", value)
        if value == memory.action:
            return self.codec.make_primitive("action", value)
        if value == memory.object:
            return self.codec.make_primitive("object", value)
        if value == memory.topic:
            return self.codec.make_primitive("topic", value)
        if value == memory.time:
            return self.codec.make_primitive("time", value)
        if value == memory.priority:
            return self.codec.make_primitive("priority", value)
        if value in memory.modifiers:
            return self.codec.make_primitive("modifier", value)
        if value in memory.constraints:
            return self.codec.make_primitive("constraint", value)
        if value in memory.negations:
            return self.codec.make_primitive("negation", value)
        if value in memory.conditions:
            return self.codec.make_primitive("condition", value)
        if relation_type in {"compare_left", "compare_right"}:
            return self.codec.make_primitive("compare_entity", value)
        if endpoint_role == "source" and memory.object:
            return self.codec.make_primitive("object", value)
        return self.codec.make_primitive("topic", value)
