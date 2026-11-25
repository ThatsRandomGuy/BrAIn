from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from brain_bitpacked_codec import BrainBitPackedCodec
from brain_binary_codec import BrainBinaryCodec, EMPTY_ID
from brain_codec import BrainCodec, BrainMemory


PATTERN_MAGIC = b"PT"
PATTERN_VERSION = 1

SINGLE_FIELDS = ("actor", "intent", "action", "object", "topic", "time", "priority")
LIST_FIELDS = ("modifiers", "negations", "constraints", "conditions")

VARIABLE_SINGLE_FIELD_TYPES = {
    "topic": "topic",
    "time": "time",
}

VARIABLE_LIST_FIELD_TYPES = {
    "modifiers": "modifier",
    "negations": "negation",
    "constraints": "constraint",
    "conditions": "condition",
}

VARIABLE_RELATION_PRIMITIVE_TYPES = {
    "topic",
    "modifier",
    "negation",
    "constraint",
    "condition",
    "time",
    "compare_entity",
}


@dataclass(frozen=True)
class PatternSlot:
    slot_name: str
    slot_index: int
    source_kind: str
    path: str
    primitive_type: str
    description: str
    optional: bool = False

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "slot_name": self.slot_name,
            "slot_index": self.slot_index,
            "source_kind": self.source_kind,
            "path": self.path,
            "primitive_type": self.primitive_type,
            "description": self.description,
        }
        if self.optional:
            payload["optional"] = True
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PatternSlot":
        return cls(
            slot_name=str(data.get("slot_name", "")),
            slot_index=int(data.get("slot_index", 0)),
            source_kind=str(data.get("source_kind", "field")),
            path=str(data.get("path", "")),
            primitive_type=str(data.get("primitive_type", "")),
            description=str(data.get("description", "")),
            optional=bool(data.get("optional", False)),
        )


@dataclass(frozen=True)
class PatternRelationTemplate:
    relation_index: int
    relation_type: str
    source_mode: str
    source_value: str
    target_mode: str
    target_value: str
    optional: bool = False

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "relation_index": self.relation_index,
            "relation_type": self.relation_type,
            "source_mode": self.source_mode,
            "source_value": self.source_value,
            "target_mode": self.target_mode,
            "target_value": self.target_value,
        }
        if self.optional:
            payload["optional"] = True
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PatternRelationTemplate":
        return cls(
            relation_index=int(data.get("relation_index", 0)),
            relation_type=str(data.get("relation_type", "")),
            source_mode=str(data.get("source_mode", "literal")),
            source_value=str(data.get("source_value", "")),
            target_mode=str(data.get("target_mode", "literal")),
            target_value=str(data.get("target_value", "")),
            optional=bool(data.get("optional", False)),
        )


@dataclass(frozen=True)
class PatternDefinition:
    pattern_id: int
    name: str
    fingerprint: str
    created_sequence: int
    usage_count: int
    stable_fields: Dict[str, str]
    single_field_modes: Dict[str, str]
    list_field_lengths: Dict[str, int]
    variable_slots: List[PatternSlot]
    relation_templates: List[PatternRelationTemplate]
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "fingerprint": self.fingerprint,
            "created_sequence": self.created_sequence,
            "usage_count": self.usage_count,
            "stable_fields": dict(self.stable_fields),
            "single_field_modes": dict(self.single_field_modes),
            "list_field_lengths": dict(self.list_field_lengths),
            "variable_slots": [slot.to_dict() for slot in self.variable_slots],
            "relation_templates": [template.to_dict() for template in self.relation_templates],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PatternDefinition":
        stable_fields_raw = data.get("stable_fields", {})
        stable_fields = (
            {str(key): str(value) for key, value in stable_fields_raw.items()}
            if isinstance(stable_fields_raw, dict)
            else {}
        )

        single_field_modes_raw = data.get("single_field_modes", {})
        single_field_modes = (
            {str(key): str(value) for key, value in single_field_modes_raw.items()}
            if isinstance(single_field_modes_raw, dict)
            else {}
        )

        list_field_lengths_raw = data.get("list_field_lengths", {})
        list_field_lengths = (
            {str(key): int(value) for key, value in list_field_lengths_raw.items()}
            if isinstance(list_field_lengths_raw, dict)
            else {}
        )

        variable_slots_raw = data.get("variable_slots", [])
        relation_templates_raw = data.get("relation_templates", [])
        notes_raw = data.get("notes", [])

        return cls(
            pattern_id=int(data.get("pattern_id", 0)),
            name=str(data.get("name", "")),
            fingerprint=str(data.get("fingerprint", "")),
            created_sequence=int(data.get("created_sequence", 0)),
            usage_count=int(data.get("usage_count", 0)),
            stable_fields=stable_fields,
            single_field_modes=single_field_modes,
            list_field_lengths=list_field_lengths,
            variable_slots=[
                PatternSlot.from_dict(slot)
                for slot in variable_slots_raw
                if isinstance(slot, dict)
            ],
            relation_templates=[
                PatternRelationTemplate.from_dict(template)
                for template in relation_templates_raw
                if isinstance(template, dict)
            ],
            notes=[str(note) for note in notes_raw] if isinstance(notes_raw, list) else [],
        )


@dataclass(frozen=True)
class PatternMatch:
    pattern_id: int
    pattern_name: str
    fingerprint: str
    score: float
    stable_fields: List[str]
    variable_slots: Dict[str, str]
    reasons: List[str]


@dataclass(frozen=True)
class PatternEncodedBundle:
    pattern_id: int
    pattern_name: str
    slot_values: Dict[str, str]
    new_primitives: List[str]
    payload: bytes

    @property
    def total_bytes(self) -> int:
        return len(self.payload)

    @property
    def variable_slot_count(self) -> int:
        return len(self.slot_values)


@dataclass
class StoredPatternMemory:
    memory_id: int
    encoding_type: str
    payload: bytes
    pattern_id: Optional[int]
    resolved_bundle: Optional[Dict[str, List[List[List[int]]]]] = None
    match: Optional[PatternMatch] = None
    encoded_bundle: Optional[PatternEncodedBundle] = None
    full_size_bytes: int = 0
    stored_size_bytes: int = 0
    variable_slot_count: int = 0
    new_primitives: List[str] = field(default_factory=list)


class BrainPatternCodecError(Exception):
    pass


class BrainPatternCodec:
    """
    Deterministic pattern extraction and slot encoding for BrAIn memories.
    """

    def __init__(
        self,
        codec: BrainCodec,
        binary_codec: Optional[BrainBinaryCodec] = None,
        bitpacked_codec: Optional[BrainBitPackedCodec] = None,
    ) -> None:
        self.codec = codec
        self.binary_codec = binary_codec or BrainBinaryCodec(codec=codec)
        self.bitpacked_codec = bitpacked_codec or BrainBitPackedCodec(binary_codec=self.binary_codec)

    def extract_pattern(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        pattern_id: int = 0,
        created_sequence: int = 0,
        usage_count: int = 0,
        notes: Optional[Sequence[str]] = None,
    ) -> PatternDefinition:
        memory = self.codec.decode_memory_bundle(memory_bundle)
        stable_fields: Dict[str, str] = {}
        single_field_modes: Dict[str, str] = {}
        list_field_lengths: Dict[str, int] = {}
        variable_slots: List[PatternSlot] = []
        slot_name_by_value: Dict[Tuple[str, str], str] = {}
        slot_index = 1

        for field in SINGLE_FIELDS:
            value = str(getattr(memory, field) or "")
            if field in VARIABLE_SINGLE_FIELD_TYPES and value:
                slot = self._build_slot(
                    slot_index=slot_index,
                    source_kind="field",
                    path=field,
                    primitive_type=VARIABLE_SINGLE_FIELD_TYPES[field],
                    description=f"{field} variable slot",
                )
                variable_slots.append(slot)
                single_field_modes[field] = "variable"
                slot_name_by_value[(slot.primitive_type, value)] = slot.slot_name
                slot_index += 1
            elif value:
                stable_fields[field] = value
                single_field_modes[field] = "stable"
            else:
                single_field_modes[field] = "empty"

        for field in LIST_FIELDS:
            values = list(getattr(memory, field))
            list_field_lengths[field] = len(values)
            primitive_type = VARIABLE_LIST_FIELD_TYPES[field]
            for value_index, value in enumerate(values):
                slot = self._build_slot(
                    slot_index=slot_index,
                    source_kind="field",
                    path=f"{field}[{value_index}]",
                    primitive_type=primitive_type,
                    description=f"{field}[{value_index}] variable slot",
                )
                variable_slots.append(slot)
                slot_name_by_value.setdefault((primitive_type, str(value)), slot.slot_name)
                slot_index += 1

        relation_templates: List[PatternRelationTemplate] = []
        for relation_index, relation in enumerate(memory.relations):
            relation_type = str(relation.get("type", "") or "")
            source_mode, source_value, slot_index = self._resolve_relation_component(
                relation_type=relation_type,
                endpoint_role="source",
                value=str(relation.get("source", "") or ""),
                relation_index=relation_index,
                slot_index=slot_index,
                variable_slots=variable_slots,
                slot_name_by_value=slot_name_by_value,
            )
            target_mode, target_value, slot_index = self._resolve_relation_component(
                relation_type=relation_type,
                endpoint_role="target",
                value=str(relation.get("target", "") or ""),
                relation_index=relation_index,
                slot_index=slot_index,
                variable_slots=variable_slots,
                slot_name_by_value=slot_name_by_value,
            )

            relation_templates.append(
                PatternRelationTemplate(
                    relation_index=relation_index,
                    relation_type=relation_type,
                    source_mode=source_mode,
                    source_value=source_value,
                    target_mode=target_mode,
                    target_value=target_value,
                )
            )

        resolved_notes = list(notes) if notes is not None else self._default_notes(variable_slots)
        candidate = PatternDefinition(
            pattern_id=pattern_id,
            name=f"pattern_{pattern_id:03d}" if pattern_id > 0 else "pattern_candidate",
            fingerprint="",
            created_sequence=created_sequence,
            usage_count=usage_count,
            stable_fields=stable_fields,
            single_field_modes=single_field_modes,
            list_field_lengths=list_field_lengths,
            variable_slots=variable_slots,
            relation_templates=relation_templates,
            notes=resolved_notes,
        )
        fingerprint = self._fingerprint_pattern(candidate)
        return PatternDefinition(
            pattern_id=candidate.pattern_id,
            name=candidate.name,
            fingerprint=fingerprint,
            created_sequence=candidate.created_sequence,
            usage_count=candidate.usage_count,
            stable_fields=candidate.stable_fields,
            single_field_modes=candidate.single_field_modes,
            list_field_lengths=candidate.list_field_lengths,
            variable_slots=candidate.variable_slots,
            relation_templates=candidate.relation_templates,
            notes=candidate.notes,
        )

    def find_best_pattern(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        known_patterns: Sequence[PatternDefinition | Dict[str, object]],
    ) -> Optional[PatternMatch]:
        candidate = self.extract_pattern(memory_bundle)
        memory = self.codec.decode_memory_bundle(memory_bundle)
        patterns = self._coerce_patterns(known_patterns)

        matches = [
            pattern
            for pattern in patterns
            if pattern.fingerprint == candidate.fingerprint
        ]
        if not matches:
            return None

        pattern = min(matches, key=lambda item: item.pattern_id)
        slot_values = self._extract_slot_values(memory, pattern.variable_slots)
        reasons = [
            "fingerprint_match",
            f"pattern_id={pattern.pattern_id}",
            f"stable_fields={','.join(sorted(pattern.stable_fields.keys())) or 'none'}",
            f"variable_slots={','.join(slot.slot_name for slot in pattern.variable_slots) or 'none'}",
            f"relation_count={len(pattern.relation_templates)}",
        ]
        return PatternMatch(
            pattern_id=pattern.pattern_id,
            pattern_name=pattern.name,
            fingerprint=pattern.fingerprint,
            score=1.0,
            stable_fields=sorted(pattern.stable_fields.keys()),
            variable_slots=slot_values,
            reasons=reasons,
        )

    def encode_memory_bundle_pattern(
        self,
        current_bundle: Dict[str, List[List[List[int]]]],
        pattern: PatternDefinition | Dict[str, object],
    ) -> PatternEncodedBundle:
        resolved_pattern = self._coerce_pattern(pattern)
        candidate = self.extract_pattern(current_bundle)
        if candidate.fingerprint != resolved_pattern.fingerprint:
            raise BrainPatternCodecError(
                f"Memory bundle does not match pattern {resolved_pattern.pattern_id}."
            )

        memory = self.codec.decode_memory_bundle(current_bundle)
        slot_values = self._extract_slot_values(memory, resolved_pattern.variable_slots)
        primitive_ids = [
            self._slot_value_to_id(slot, slot_values.get(slot.slot_name, ""))
            for slot in resolved_pattern.variable_slots
        ]
        primitive_id_bytes = self._primitive_id_width(primitive_ids)

        if resolved_pattern.pattern_id < 0 or resolved_pattern.pattern_id > 0xFFFF:
            raise BrainPatternCodecError("Pattern IDs must fit in uint16.")

        parts = [
            PATTERN_MAGIC,
            struct.pack("<BHB", PATTERN_VERSION, resolved_pattern.pattern_id, primitive_id_bytes),
        ]
        for primitive_id in primitive_ids:
            parts.append(self._pack_primitive_id(primitive_id, primitive_id_bytes))

        payload = b"".join(parts)
        new_primitives = self._diff_pattern_primitives(
            current_bundle=current_bundle,
            pattern=resolved_pattern,
            slot_values=slot_values,
        )
        return PatternEncodedBundle(
            pattern_id=resolved_pattern.pattern_id,
            pattern_name=resolved_pattern.name,
            slot_values=slot_values,
            new_primitives=new_primitives,
            payload=payload,
        )

    def decode_memory_bundle_pattern(
        self,
        pattern_bundle: PatternEncodedBundle | bytes,
        pattern: PatternDefinition | Dict[str, object],
    ) -> Dict[str, List[List[List[int]]]]:
        resolved_pattern = self._coerce_pattern(pattern)
        payload = pattern_bundle.payload if isinstance(pattern_bundle, PatternEncodedBundle) else pattern_bundle
        slot_values = self._deserialize_slot_values(payload, resolved_pattern)
        memory = self._reconstruct_memory(resolved_pattern, slot_values)
        return self.codec.encode_memory_bundle(memory)

    def full_bundle_bytes(self, bundle: Dict[str, List[List[List[int]]]]) -> bytes:
        self.binary_codec.sync_from_codec()
        return self.bitpacked_codec.encode_jsonable_bundle_bitpacked(bundle).payload

    def _resolve_relation_component(
        self,
        relation_type: str,
        endpoint_role: str,
        value: str,
        relation_index: int,
        slot_index: int,
        variable_slots: List[PatternSlot],
        slot_name_by_value: Dict[Tuple[str, str], str],
    ) -> Tuple[str, str, int]:
        if not value:
            return "literal", "", slot_index

        if relation_type in {"compare_left", "compare_right"} and endpoint_role == "target":
            primitive_type = "compare_entity"
            primitive_value = value
        else:
            primitive = self.codec._guess_relation_endpoint_primitive(
                value=value,
                relation_type=relation_type,
                endpoint_role=endpoint_role,
            )
            primitive_type, primitive_value = primitive.split(":", 1)
        reusable_slot = slot_name_by_value.get((primitive_type, primitive_value))
        if reusable_slot is not None:
            return "slot", reusable_slot, slot_index

        if (
            primitive_type in VARIABLE_RELATION_PRIMITIVE_TYPES
        ):
            path = f"relations[{relation_index}].{endpoint_role}"
            slot = self._build_slot(
                slot_index=slot_index,
                source_kind="relation",
                path=path,
                primitive_type=primitive_type,
                description=f"{relation_type}.{endpoint_role} variable slot",
            )
            variable_slots.append(slot)
            slot_name_by_value[(primitive_type, primitive_value)] = slot.slot_name
            return "slot", slot.slot_name, slot_index + 1

        return "literal", value, slot_index

    @staticmethod
    def _build_slot(
        slot_index: int,
        source_kind: str,
        path: str,
        primitive_type: str,
        description: str,
        optional: bool = False,
    ) -> PatternSlot:
        sanitized = (
            path.replace("[", "_")
            .replace("]", "")
            .replace(".", "_")
        )
        return PatternSlot(
            slot_name=f"slot_{sanitized}",
            slot_index=slot_index,
            source_kind=source_kind,
            path=path,
            primitive_type=primitive_type,
            description=description,
            optional=optional,
        )

    @staticmethod
    def _default_notes(variable_slots: Sequence[PatternSlot]) -> List[str]:
        if not variable_slots:
            return ["no_variable_slots"]
        return [
            f"variable_slots={','.join(slot.path for slot in variable_slots)}",
        ]

    def _extract_slot_values(
        self,
        memory: BrainMemory,
        variable_slots: Sequence[PatternSlot],
    ) -> Dict[str, str]:
        slot_values: Dict[str, str] = {}
        for slot in sorted(variable_slots, key=lambda item: item.slot_index):
            value = self._read_path(memory, slot.path)
            if value == "" and not slot.optional:
                raise BrainPatternCodecError(f"Pattern slot {slot.slot_name} is missing value at {slot.path}.")
            slot_values[slot.slot_name] = value
        return slot_values

    def _read_path(self, memory: BrainMemory, path: str) -> str:
        if path.startswith("relations["):
            relation_index, field_name = self._parse_relation_path(path)
            if relation_index >= len(memory.relations):
                return ""
            return str(memory.relations[relation_index].get(field_name, "") or "")

        if "[" in path and path.endswith("]"):
            field_name, index = self._parse_list_path(path)
            values = list(getattr(memory, field_name))
            return str(values[index]) if index < len(values) else ""

        return str(getattr(memory, path) or "")

    def _write_path(self, memory: BrainMemory, path: str, value: str) -> None:
        if path.startswith("relations["):
            raise BrainPatternCodecError("Relation paths are reconstructed from templates, not written directly.")

        if "[" in path and path.endswith("]"):
            field_name, index = self._parse_list_path(path)
            values = list(getattr(memory, field_name))
            while len(values) <= index:
                values.append("")
            values[index] = value
            setattr(memory, field_name, values)
            return

        setattr(memory, path, value)

    def _reconstruct_memory(
        self,
        pattern: PatternDefinition,
        slot_values: Dict[str, str],
    ) -> BrainMemory:
        memory = BrainMemory()

        for field, value in pattern.stable_fields.items():
            setattr(memory, field, value)

        for field in LIST_FIELDS:
            expected_length = int(pattern.list_field_lengths.get(field, 0))
            setattr(memory, field, [])
            if expected_length > 0:
                setattr(memory, field, [""] * expected_length)

        for slot in sorted(pattern.variable_slots, key=lambda item: item.slot_index):
            if slot.source_kind != "field":
                continue
            value = slot_values.get(slot.slot_name, "")
            self._write_path(memory, slot.path, value)

        for field in LIST_FIELDS:
            values = [value for value in list(getattr(memory, field)) if value]
            setattr(memory, field, values)

        relations: List[Dict[str, str]] = []
        for template in sorted(pattern.relation_templates, key=lambda item: item.relation_index):
            source = (
                template.source_value
                if template.source_mode == "literal"
                else slot_values.get(template.source_value, "")
            )
            target = (
                template.target_value
                if template.target_mode == "literal"
                else slot_values.get(template.target_value, "")
            )
            if template.optional and (not source or not target):
                continue
            relations.append(
                {
                    "type": template.relation_type,
                    "source": source,
                    "target": target,
                }
            )

        memory.relations = relations
        return memory

    def _deserialize_slot_values(
        self,
        payload: bytes,
        pattern: PatternDefinition,
    ) -> Dict[str, str]:
        if len(payload) < 6:
            raise BrainPatternCodecError("Pattern payload is too short.")
        if payload[:2] != PATTERN_MAGIC:
            raise BrainPatternCodecError("Pattern payload magic header is invalid.")

        version, pattern_id, primitive_id_bytes = struct.unpack("<BHB", payload[2:6])
        if version != PATTERN_VERSION:
            raise BrainPatternCodecError(f"Unsupported pattern payload version: {version}")
        if pattern_id != pattern.pattern_id:
            raise BrainPatternCodecError(
                f"Pattern payload references pattern {pattern_id}, expected {pattern.pattern_id}."
            )
        if primitive_id_bytes not in (1, 2, 4):
            raise BrainPatternCodecError("Pattern primitive ID width must be 1, 2, or 4 bytes.")

        expected_length = 6 + (len(pattern.variable_slots) * primitive_id_bytes)
        if len(payload) != expected_length:
            raise BrainPatternCodecError(
                f"Pattern payload length {len(payload)} does not match expected {expected_length}."
            )

        offset = 6
        slot_values: Dict[str, str] = {}
        for slot in sorted(pattern.variable_slots, key=lambda item: item.slot_index):
            primitive_id, offset = self._unpack_primitive_id(payload, offset, primitive_id_bytes)
            slot_values[slot.slot_name] = self._primitive_id_to_slot_value(slot, primitive_id)

        return slot_values

    def _slot_value_to_id(self, slot: PatternSlot, value: str) -> int:
        if not value:
            return EMPTY_ID
        primitive = self.codec.make_primitive(slot.primitive_type, value)
        return self.binary_codec.primitive_id(primitive)

    def _primitive_id_to_slot_value(self, slot: PatternSlot, primitive_id: int) -> str:
        if primitive_id == EMPTY_ID:
            return ""
        primitive = self.binary_codec.id_to_primitive.get(primitive_id)
        if primitive in (None, "UNKNOWN"):
            raise BrainPatternCodecError(f"Unknown pattern primitive ID: {primitive_id}")
        primitive_text = str(primitive)
        prefix, value = primitive_text.split(":", 1)
        if prefix != slot.primitive_type:
            raise BrainPatternCodecError(
                f"Pattern slot {slot.slot_name} expected primitive type {slot.primitive_type}, got {prefix}."
            )
        return value

    def _diff_pattern_primitives(
        self,
        current_bundle: Dict[str, List[List[List[int]]]],
        pattern: PatternDefinition,
        slot_values: Dict[str, str],
    ) -> List[str]:
        base_ids = self._pattern_base_primitive_ids(pattern)
        current_ids = self._bundle_primitive_ids(current_bundle)
        slot_ids = set()
        for slot in pattern.variable_slots:
            slot_ids.add(self._slot_value_to_id(slot, slot_values.get(slot.slot_name, "")))
        relevant_ids = sorted(
            primitive_id
            for primitive_id in slot_ids
            if primitive_id not in base_ids and primitive_id != EMPTY_ID
        )

        primitives: List[str] = []
        for primitive_id in relevant_ids:
            primitive = self.binary_codec.id_to_primitive.get(primitive_id)
            if primitive in (None, "UNKNOWN"):
                continue
            primitive_text = str(primitive)
            if primitive_id in current_ids or primitive_text.startswith("compare_entity:"):
                primitives.append(primitive_text)
        return primitives

    def _pattern_base_primitive_ids(self, pattern: PatternDefinition) -> set[int]:
        base_ids: set[int] = set()
        for field, value in pattern.stable_fields.items():
            primitive = self.codec.make_primitive(field, value)
            base_ids.add(self.binary_codec.primitive_id(primitive))

        for template in pattern.relation_templates:
            relation_primitive = self.codec.make_primitive("relation_type", template.relation_type)
            base_ids.add(self.binary_codec.primitive_id(relation_primitive))

            if template.source_mode == "literal" and template.source_value:
                source_primitive = self.codec._guess_relation_endpoint_primitive(
                    value=template.source_value,
                    relation_type=template.relation_type,
                    endpoint_role="source",
                )
                base_ids.add(self.binary_codec.primitive_id(source_primitive))

            if template.target_mode == "literal" and template.target_value:
                target_primitive = self.codec._guess_relation_endpoint_primitive(
                    value=template.target_value,
                    relation_type=template.relation_type,
                    endpoint_role="target",
                )
                base_ids.add(self.binary_codec.primitive_id(target_primitive))

        return {primitive_id for primitive_id in base_ids if primitive_id != EMPTY_ID}

    def _bundle_primitive_ids(self, bundle: Dict[str, List[List[List[int]]]]) -> set[int]:
        fixed_bundle = self.binary_codec.encode_jsonable_bundle(bundle)
        return {
            primitive_id
            for primitive_id in list(fixed_bundle.core_ids) + list(fixed_bundle.relation_ids)
            if primitive_id != EMPTY_ID
        }

    def _fingerprint_pattern(self, pattern: PatternDefinition) -> str:
        fingerprint_source = {
            "stable_fields": {key: pattern.stable_fields[key] for key in sorted(pattern.stable_fields)},
            "single_field_modes": {
                key: pattern.single_field_modes.get(key, "empty")
                for key in SINGLE_FIELDS
            },
            "list_field_lengths": {
                key: int(pattern.list_field_lengths.get(key, 0))
                for key in LIST_FIELDS
            },
            "variable_slots": [
                {
                    "slot_name": slot.slot_name,
                    "slot_index": slot.slot_index,
                    "source_kind": slot.source_kind,
                    "path": slot.path,
                    "primitive_type": slot.primitive_type,
                    **({"optional": True} if slot.optional else {}),
                }
                for slot in sorted(pattern.variable_slots, key=lambda item: item.slot_index)
            ],
            "relation_templates": [
                {
                    "relation_index": template.relation_index,
                    "relation_type": template.relation_type,
                    "source_mode": template.source_mode,
                    "source_value": template.source_value,
                    "target_mode": template.target_mode,
                    "target_value": template.target_value,
                    **({"optional": True} if template.optional else {}),
                }
                for template in sorted(pattern.relation_templates, key=lambda item: item.relation_index)
            ],
        }
        canonical = json.dumps(fingerprint_source, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _pack_primitive_id(primitive_id: int, width: int) -> bytes:
        if width == 1:
            return struct.pack("<B", primitive_id)
        if width == 2:
            return struct.pack("<H", primitive_id)
        if width == 4:
            return struct.pack("<I", primitive_id)
        raise BrainPatternCodecError(f"Unsupported primitive ID width: {width}")

    @staticmethod
    def _unpack_primitive_id(payload: bytes, offset: int, width: int) -> Tuple[int, int]:
        end = offset + width
        if end > len(payload):
            raise BrainPatternCodecError("Pattern payload ended while reading primitive ID.")
        if width == 1:
            return payload[offset], end
        if width == 2:
            return struct.unpack("<H", payload[offset:end])[0], end
        if width == 4:
            return struct.unpack("<I", payload[offset:end])[0], end
        raise BrainPatternCodecError(f"Unsupported primitive ID width: {width}")

    @staticmethod
    def _primitive_id_width(primitive_ids: Sequence[int]) -> int:
        max_id = max([0] + [primitive_id for primitive_id in primitive_ids if primitive_id >= 0])
        if max_id <= 0xFF:
            return 1
        if max_id <= 0xFFFF:
            return 2
        return 4

    @staticmethod
    def _parse_list_path(path: str) -> Tuple[str, int]:
        field_name, index_text = path[:-1].split("[", 1)
        return field_name, int(index_text)

    @staticmethod
    def _parse_relation_path(path: str) -> Tuple[int, str]:
        prefix, field_name = path.split("].", 1)
        relation_index = int(prefix.split("[", 1)[1])
        return relation_index, field_name

    @staticmethod
    def _coerce_pattern(pattern: PatternDefinition | Dict[str, object]) -> PatternDefinition:
        if isinstance(pattern, PatternDefinition):
            return pattern
        if isinstance(pattern, dict):
            return PatternDefinition.from_dict(pattern)
        raise BrainPatternCodecError("Pattern must be a PatternDefinition or mapping.")

    def _coerce_patterns(
        self,
        patterns: Sequence[PatternDefinition | Dict[str, object]],
    ) -> List[PatternDefinition]:
        return [self._coerce_pattern(pattern) for pattern in patterns]


class BrainPatternStore:
    """
    Persistent learned-pattern registry for BrAIn.
    """

    def __init__(
        self,
        pattern_codec: BrainPatternCodec,
        pattern_path: str | Path = "brain_patterns.json",
    ) -> None:
        self.pattern_codec = pattern_codec
        self.pattern_path = Path(pattern_path)
        self.patterns: Dict[int, PatternDefinition] = {}
        self.next_pattern_id = 1
        self._load()

    def _load(self) -> None:
        if not self.pattern_path.exists():
            self.patterns = {}
            self.next_pattern_id = 1
            return

        with open(self.pattern_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_patterns: Dict[int, PatternDefinition] = {}
        next_pattern_id = 1
        if isinstance(data, dict):
            next_pattern_id = int(data.get("next_pattern_id", 1))
            raw_patterns = data.get("patterns", [])
            if isinstance(raw_patterns, list):
                for item in raw_patterns:
                    if not isinstance(item, dict):
                        continue
                    pattern = PatternDefinition.from_dict(item)
                    if pattern.pattern_id <= 0:
                        continue
                    loaded_patterns[pattern.pattern_id] = pattern

        self.patterns = loaded_patterns
        self.next_pattern_id = max([next_pattern_id] + [pattern_id + 1 for pattern_id in self.patterns], default=1)

    def save(self) -> None:
        payload = {
            "next_pattern_id": self.next_pattern_id,
            "patterns": [
                self.patterns[pattern_id].to_dict()
                for pattern_id in sorted(self.patterns)
            ],
        }
        with open(self.pattern_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def all_patterns(self) -> List[PatternDefinition]:
        return [self.patterns[pattern_id] for pattern_id in sorted(self.patterns)]

    def get(self, pattern_id: int) -> PatternDefinition:
        pattern = self.patterns.get(pattern_id)
        if pattern is None:
            raise BrainPatternCodecError(f"Unknown pattern ID: {pattern_id}")
        return pattern

    def find_by_fingerprint(self, fingerprint: str) -> Optional[PatternDefinition]:
        for pattern_id in sorted(self.patterns):
            pattern = self.patterns[pattern_id]
            if pattern.fingerprint == fingerprint:
                return pattern
        return None

    def ensure_pattern(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        notes: Optional[Sequence[str]] = None,
    ) -> Tuple[PatternDefinition, bool]:
        candidate = self.pattern_codec.extract_pattern(memory_bundle, notes=notes)
        existing = self.find_by_fingerprint(candidate.fingerprint)
        if existing is not None:
            return existing, False

        pattern_id = self.next_pattern_id
        created = self.pattern_codec.extract_pattern(
            memory_bundle=memory_bundle,
            pattern_id=pattern_id,
            created_sequence=pattern_id,
            usage_count=1,
            notes=notes,
        )
        self.patterns[pattern_id] = created
        self.next_pattern_id += 1
        self.save()
        return created, True

    def record_usage(self, pattern_id: int, increment: int = 1) -> PatternDefinition:
        pattern = self.get(pattern_id)
        updated = PatternDefinition(
            pattern_id=pattern.pattern_id,
            name=pattern.name,
            fingerprint=pattern.fingerprint,
            created_sequence=pattern.created_sequence,
            usage_count=pattern.usage_count + increment,
            stable_fields=pattern.stable_fields,
            single_field_modes=pattern.single_field_modes,
            list_field_lengths=pattern.list_field_lengths,
            variable_slots=pattern.variable_slots,
            relation_templates=pattern.relation_templates,
            notes=pattern.notes,
        )
        self.patterns[pattern_id] = updated
        self.save()
        return updated


class BrainPatternMemoryStore:
    """
    In-memory storage for full and pattern-encoded BrAIn memories.

    The pattern registry persists across runs. Stored memory instances are kept
    in memory for the active benchmark/session.
    """

    def __init__(
        self,
        pattern_codec: BrainPatternCodec,
        pattern_store: BrainPatternStore,
    ) -> None:
        self.pattern_codec = pattern_codec
        self.pattern_store = pattern_store
        self.entries: Dict[int, StoredPatternMemory] = {}
        self.order: List[int] = []

    def add_memory(
        self,
        memory_id: int,
        memory_bundle: Dict[str, List[List[List[int]]]],
    ) -> StoredPatternMemory:
        if memory_id in self.entries:
            raise BrainPatternCodecError(f"Memory ID already exists in pattern store: {memory_id}")

        full_payload = self.pattern_codec.full_bundle_bytes(memory_bundle)
        match = self.pattern_codec.find_best_pattern(
            memory_bundle=memory_bundle,
            known_patterns=self.pattern_store.all_patterns(),
        )

        if match is not None:
            pattern = self.pattern_store.get(match.pattern_id)
            encoded_bundle = self.pattern_codec.encode_memory_bundle_pattern(memory_bundle, pattern)
            self.pattern_store.record_usage(pattern.pattern_id)
            if encoded_bundle.total_bytes <= len(full_payload):
                entry = StoredPatternMemory(
                    memory_id=memory_id,
                    encoding_type="pattern",
                    payload=encoded_bundle.payload,
                    pattern_id=pattern.pattern_id,
                    resolved_bundle=memory_bundle,
                    match=match,
                    encoded_bundle=encoded_bundle,
                    full_size_bytes=len(full_payload),
                    stored_size_bytes=encoded_bundle.total_bytes,
                    variable_slot_count=encoded_bundle.variable_slot_count,
                    new_primitives=list(encoded_bundle.new_primitives),
                )
            else:
                entry = StoredPatternMemory(
                    memory_id=memory_id,
                    encoding_type="full",
                    payload=full_payload,
                    pattern_id=pattern.pattern_id,
                    resolved_bundle=memory_bundle,
                    match=match,
                    encoded_bundle=encoded_bundle,
                    full_size_bytes=len(full_payload),
                    stored_size_bytes=len(full_payload),
                    variable_slot_count=encoded_bundle.variable_slot_count,
                    new_primitives=list(encoded_bundle.new_primitives),
                )
        else:
            self.pattern_store.ensure_pattern(memory_bundle)
            entry = StoredPatternMemory(
                memory_id=memory_id,
                encoding_type="full",
                payload=full_payload,
                pattern_id=None,
                resolved_bundle=memory_bundle,
                match=None,
                encoded_bundle=None,
                full_size_bytes=len(full_payload),
                stored_size_bytes=len(full_payload),
                variable_slot_count=0,
                new_primitives=[],
            )

        self.entries[memory_id] = entry
        self.order.append(memory_id)
        return entry

    def resolve_bundle(self, memory_id: int) -> Dict[str, List[List[List[int]]]]:
        entry = self.entries.get(memory_id)
        if entry is None:
            raise BrainPatternCodecError(f"Unknown memory ID: {memory_id}")

        if entry.resolved_bundle is not None:
            return entry.resolved_bundle

        if entry.encoding_type == "full":
            bundle = self.pattern_codec.bitpacked_codec.decode_memory_bundle_bitpacked(entry.payload)
            entry.resolved_bundle = bundle
            return bundle

        if entry.pattern_id is None:
            raise BrainPatternCodecError(f"Pattern-encoded memory {memory_id} is missing a pattern ID.")

        pattern = self.pattern_store.get(entry.pattern_id)
        bundle = self.pattern_codec.decode_memory_bundle_pattern(entry.payload, pattern)
        entry.resolved_bundle = bundle
        return bundle


def extract_pattern(
    memory_bundle: Dict[str, List[List[List[int]]]],
    codec: Optional[BrainCodec] = None,
) -> PatternDefinition:
    active_codec = codec or BrainCodec(auto_register=False)
    pattern_codec = BrainPatternCodec(codec=active_codec)
    return pattern_codec.extract_pattern(memory_bundle)


def find_best_pattern(
    memory_bundle: Dict[str, List[List[List[int]]]],
    known_patterns: Sequence[PatternDefinition | Dict[str, object]],
    codec: Optional[BrainCodec] = None,
) -> Optional[PatternMatch]:
    active_codec = codec or BrainCodec(auto_register=False)
    pattern_codec = BrainPatternCodec(codec=active_codec)
    return pattern_codec.find_best_pattern(memory_bundle, known_patterns)


def encode_memory_bundle_pattern(
    current_bundle: Dict[str, List[List[List[int]]]],
    pattern: PatternDefinition | Dict[str, object],
    codec: Optional[BrainCodec] = None,
) -> PatternEncodedBundle:
    active_codec = codec or BrainCodec(auto_register=False)
    pattern_codec = BrainPatternCodec(codec=active_codec)
    return pattern_codec.encode_memory_bundle_pattern(current_bundle, pattern)


def decode_memory_bundle_pattern(
    pattern_bundle: PatternEncodedBundle | bytes,
    pattern: PatternDefinition | Dict[str, object],
    codec: Optional[BrainCodec] = None,
) -> Dict[str, List[List[List[int]]]]:
    active_codec = codec or BrainCodec(auto_register=False)
    pattern_codec = BrainPatternCodec(codec=active_codec)
    return pattern_codec.decode_memory_bundle_pattern(pattern_bundle, pattern)
