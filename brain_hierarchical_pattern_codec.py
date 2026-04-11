from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from brain_codec import BrainCodec, BrainMemory
from brain_pattern_codec import (
    LIST_FIELDS,
    PATTERN_MAGIC,
    PatternRelationTemplate,
    PatternSlot,
    SINGLE_FIELDS,
    VARIABLE_LIST_FIELD_TYPES,
    VARIABLE_RELATION_PRIMITIVE_TYPES,
    VARIABLE_SINGLE_FIELD_TYPES,
    BrainPatternCodec,
    BrainPatternCodecError,
)


HIERARCHICAL_MAGIC = b"HP"
HIERARCHICAL_VERSION = 1
MODE_PATTERN = 1
MODE_SUBPATTERNS = 2

FIELD_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("actor", "intent"),
    ("priority",),
    ("action", "object"),
)


@dataclass(frozen=True)
class HierarchicalSubpatternDefinition:
    subpattern_id: int
    kind: str
    name: str
    fingerprint: str
    created_sequence: int
    usage_count: int
    stable_fields: Dict[str, str]
    variable_slots: List[PatternSlot]
    relation_templates: List[PatternRelationTemplate]
    notes: List[str]
    active: bool = True
    generalization_level: int = 0
    merged_from_subpattern_ids: List[int] = field(default_factory=list)
    last_used_sequence: int = 0
    reuse_frequency: float = 0.0
    cooccurrence_count: int = 0
    temporal_score: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "subpattern_id": self.subpattern_id,
            "kind": self.kind,
            "name": self.name,
            "fingerprint": self.fingerprint,
            "created_sequence": self.created_sequence,
            "usage_count": self.usage_count,
            "stable_fields": dict(self.stable_fields),
            "variable_slots": [slot.to_dict() for slot in self.variable_slots],
            "relation_templates": [template.to_dict() for template in self.relation_templates],
            "notes": list(self.notes),
        }
        if not self.active:
            payload["active"] = False
        if self.generalization_level > 0:
            payload["generalization_level"] = self.generalization_level
        if self.merged_from_subpattern_ids:
            payload["merged_from_subpattern_ids"] = list(self.merged_from_subpattern_ids)
        if self.last_used_sequence > 0:
            payload["last_used_sequence"] = self.last_used_sequence
        if self.reuse_frequency > 0.0:
            payload["reuse_frequency"] = self.reuse_frequency
        if self.cooccurrence_count > 0:
            payload["cooccurrence_count"] = self.cooccurrence_count
        if self.temporal_score > 0.0:
            payload["temporal_score"] = self.temporal_score
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "HierarchicalSubpatternDefinition":
        stable_fields_raw = data.get("stable_fields", {})
        stable_fields = (
            {str(key): str(value) for key, value in stable_fields_raw.items()}
            if isinstance(stable_fields_raw, dict)
            else {}
        )
        variable_slots_raw = data.get("variable_slots", [])
        relation_templates_raw = data.get("relation_templates", [])
        notes_raw = data.get("notes", [])
        return cls(
            subpattern_id=int(data.get("subpattern_id", 0)),
            kind=str(data.get("kind", "")),
            name=str(data.get("name", "")),
            fingerprint=str(data.get("fingerprint", "")),
            created_sequence=int(data.get("created_sequence", 0)),
            usage_count=int(data.get("usage_count", 0)),
            stable_fields=stable_fields,
            variable_slots=[
                PatternSlot.from_dict(item)
                for item in variable_slots_raw
                if isinstance(item, dict)
            ],
            relation_templates=[
                PatternRelationTemplate.from_dict(item)
                for item in relation_templates_raw
                if isinstance(item, dict)
            ],
            notes=[str(note) for note in notes_raw] if isinstance(notes_raw, list) else [],
            active=bool(data.get("active", True)),
            generalization_level=int(data.get("generalization_level", 0)),
            merged_from_subpattern_ids=[
                int(item)
                for item in data.get("merged_from_subpattern_ids", [])
                if isinstance(item, int)
            ] if isinstance(data.get("merged_from_subpattern_ids", []), list) else [],
            last_used_sequence=int(data.get("last_used_sequence", 0)),
            reuse_frequency=float(data.get("reuse_frequency", 0.0)),
            cooccurrence_count=int(data.get("cooccurrence_count", 0)),
            temporal_score=float(data.get("temporal_score", 0.0)),
        )


@dataclass(frozen=True)
class HierarchicalPatternDefinition:
    pattern_id: int
    name: str
    fingerprint: str
    created_sequence: int
    usage_count: int
    subpattern_ids: List[int]
    variable_slots: List[PatternSlot]
    notes: List[str]
    active: bool = True
    generalization_level: int = 0
    generalized_slot_count: int = 0
    merged_from_pattern_ids: List[int] = field(default_factory=list)
    superseded_by_pattern_id: Optional[int] = None
    last_used_sequence: int = 0
    reuse_frequency: float = 0.0
    total_encoded_size_bytes: int = 0
    total_full_size_bytes: int = 0
    average_compression_ratio: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    compression_efficiency: float = 0.0
    temporal_score: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "fingerprint": self.fingerprint,
            "created_sequence": self.created_sequence,
            "usage_count": self.usage_count,
            "subpattern_ids": list(self.subpattern_ids),
            "variable_slots": [slot.to_dict() for slot in self.variable_slots],
            "notes": list(self.notes),
        }
        if not self.active:
            payload["active"] = False
        if self.generalization_level > 0:
            payload["generalization_level"] = self.generalization_level
        if self.generalized_slot_count > 0:
            payload["generalized_slot_count"] = self.generalized_slot_count
        if self.merged_from_pattern_ids:
            payload["merged_from_pattern_ids"] = list(self.merged_from_pattern_ids)
        if self.superseded_by_pattern_id is not None:
            payload["superseded_by_pattern_id"] = self.superseded_by_pattern_id
        if self.last_used_sequence > 0:
            payload["last_used_sequence"] = self.last_used_sequence
        if self.reuse_frequency > 0.0:
            payload["reuse_frequency"] = self.reuse_frequency
        if self.total_encoded_size_bytes > 0:
            payload["total_encoded_size_bytes"] = self.total_encoded_size_bytes
        if self.total_full_size_bytes > 0:
            payload["total_full_size_bytes"] = self.total_full_size_bytes
        if self.average_compression_ratio > 0.0:
            payload["average_compression_ratio"] = self.average_compression_ratio
        if self.success_count > 0:
            payload["success_count"] = self.success_count
        if self.failure_count > 0:
            payload["failure_count"] = self.failure_count
        if self.success_rate > 0.0:
            payload["success_rate"] = self.success_rate
        if self.compression_efficiency > 0.0:
            payload["compression_efficiency"] = self.compression_efficiency
        if self.temporal_score > 0.0:
            payload["temporal_score"] = self.temporal_score
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "HierarchicalPatternDefinition":
        subpattern_ids_raw = data.get("subpattern_ids", [])
        variable_slots_raw = data.get("variable_slots", [])
        notes_raw = data.get("notes", [])
        return cls(
            pattern_id=int(data.get("pattern_id", 0)),
            name=str(data.get("name", "")),
            fingerprint=str(data.get("fingerprint", "")),
            created_sequence=int(data.get("created_sequence", 0)),
            usage_count=int(data.get("usage_count", 0)),
            subpattern_ids=[
                int(item)
                for item in subpattern_ids_raw
                if isinstance(item, int)
            ],
            variable_slots=[
                PatternSlot.from_dict(item)
                for item in variable_slots_raw
                if isinstance(item, dict)
            ],
            notes=[str(note) for note in notes_raw] if isinstance(notes_raw, list) else [],
            active=bool(data.get("active", True)),
            generalization_level=int(data.get("generalization_level", 0)),
            generalized_slot_count=int(data.get("generalized_slot_count", 0)),
            merged_from_pattern_ids=[
                int(item)
                for item in data.get("merged_from_pattern_ids", [])
                if isinstance(item, int)
            ] if isinstance(data.get("merged_from_pattern_ids", []), list) else [],
            superseded_by_pattern_id=(
                int(data["superseded_by_pattern_id"])
                if data.get("superseded_by_pattern_id") is not None
                else None
            ),
            last_used_sequence=int(data.get("last_used_sequence", 0)),
            reuse_frequency=float(data.get("reuse_frequency", 0.0)),
            total_encoded_size_bytes=int(data.get("total_encoded_size_bytes", 0)),
            total_full_size_bytes=int(data.get("total_full_size_bytes", 0)),
            average_compression_ratio=float(data.get("average_compression_ratio", 0.0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            success_rate=float(data.get("success_rate", 0.0)),
            compression_efficiency=float(data.get("compression_efficiency", 0.0)),
            temporal_score=float(data.get("temporal_score", 0.0)),
        )


@dataclass(frozen=True)
class HierarchicalPatternCandidate:
    fingerprint: str
    subpatterns: List[HierarchicalSubpatternDefinition]
    variable_slots: List[PatternSlot]
    notes: List[str]


@dataclass(frozen=True)
class HierarchicalPatternMatch:
    pattern_id: Optional[int]
    pattern_name: str
    fingerprint: str
    reuse_mode: str
    score: float
    matched_subpattern_ids: List[int]
    missing_subpattern_fingerprints: List[str]
    variable_slots: Dict[str, str]
    reasons: List[str]

    @property
    def matched_subpattern_count(self) -> int:
        return len(self.matched_subpattern_ids)


@dataclass(frozen=True)
class HierarchicalEncodedBundle:
    mode: str
    pattern_id: Optional[int]
    subpattern_ids: List[int]
    slot_values: Dict[str, str]
    new_primitives: List[str]
    payload: bytes

    @property
    def total_bytes(self) -> int:
        return len(self.payload)

    @property
    def variable_slot_count(self) -> int:
        return len(self.slot_values)

    @property
    def subpattern_count(self) -> int:
        return len(self.subpattern_ids)


@dataclass
class StoredHierarchicalMemory:
    memory_id: int
    encoding_type: str
    payload: bytes
    pattern_id: Optional[int]
    subpattern_ids: List[int]
    resolved_bundle: Optional[Dict[str, List[List[List[int]]]]] = None
    match: Optional[HierarchicalPatternMatch] = None
    encoded_bundle: Optional[HierarchicalEncodedBundle] = None
    full_size_bytes: int = 0
    stored_size_bytes: int = 0
    variable_slot_count: int = 0
    new_primitives: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ComposedHierarchicalStructure:
    stable_fields: Dict[str, str]
    variable_slots: List[PatternSlot]
    relation_templates: List[PatternRelationTemplate]


class BrainHierarchicalPatternCodecError(BrainPatternCodecError):
    pass


class BrainHierarchicalPatternCodec(BrainPatternCodec):
    """
    Hierarchical pattern codec for BrAIn.

    This layer breaks flat patterns into reusable subpatterns such as:
    - stable field groups (`actor+intent`, `priority`, `action+object`)
    - slot attachments (`topic` + `about(object -> topic)`)
    - relation-only fragments when needed
    """

    def extract_subpatterns(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
    ) -> HierarchicalPatternCandidate:
        memory = self.codec.decode_memory_bundle(memory_bundle)
        stable_fields = self._collect_stable_fields(memory)
        variable_slots = self._collect_field_slots(memory)
        slot_by_name = {slot.slot_name: slot for slot in variable_slots}
        slot_name_by_key = {
            (slot.primitive_type, self._read_path(memory, slot.path)): slot.slot_name
            for slot in variable_slots
        }

        relation_templates = self._extract_relation_templates(
            memory=memory,
            stable_fields=stable_fields,
            slot_by_name=slot_by_name,
            slot_name_by_key=slot_name_by_key,
            variable_slots=variable_slots,
        )

        subpatterns: List[HierarchicalSubpatternDefinition] = []
        remaining_stable_fields = dict(stable_fields)

        for group in FIELD_GROUPS:
            grouped_fields = {
                field: remaining_stable_fields[field]
                for field in group
                if field in remaining_stable_fields
            }
            if not grouped_fields:
                continue
            for field in grouped_fields:
                remaining_stable_fields.pop(field, None)
            subpatterns.append(
                self._make_subpattern_candidate(
                    kind="field_group",
                    name=f"fields_{'_'.join(group)}",
                    stable_fields=grouped_fields,
                    variable_slots=[],
                    relation_templates=[],
                    notes=[f"stable_fields={','.join(sorted(grouped_fields))}"],
                )
            )

        for field in sorted(remaining_stable_fields):
            subpatterns.append(
                self._make_subpattern_candidate(
                    kind="field_group",
                    name=f"field_{field}",
                    stable_fields={field: remaining_stable_fields[field]},
                    variable_slots=[],
                    relation_templates=[],
                    notes=[f"stable_field={field}"],
                )
            )

        assigned_relation_indexes: set[int] = set()
        for slot in sorted(variable_slots, key=self._slot_sort_key):
            related_relations = [
                template
                for template in relation_templates
                if template.relation_index not in assigned_relation_indexes
                and self._relation_references_slot(template, slot.slot_name)
            ]
            for template in related_relations:
                assigned_relation_indexes.add(template.relation_index)
            subpatterns.append(
                self._make_subpattern_candidate(
                    kind="attachment",
                    name=f"attachment_{slot.slot_name}",
                    stable_fields={},
                    variable_slots=[slot],
                    relation_templates=related_relations,
                    notes=[f"slot_path={slot.path}"],
                )
            )

        for template in relation_templates:
            if template.relation_index in assigned_relation_indexes:
                continue
            subpatterns.append(
                self._make_subpattern_candidate(
                    kind="relation",
                    name=f"relation_{template.relation_type}_{template.relation_index:02d}",
                    stable_fields={},
                    variable_slots=[],
                    relation_templates=[template],
                    notes=[f"relation_type={template.relation_type}"],
                )
            )

        composed = self._compose_subpatterns(subpatterns)
        fingerprint = self._fingerprint_hierarchical_candidate(subpatterns)
        notes = [
            f"subpattern_count={len(subpatterns)}",
            f"variable_slots={','.join(slot.path for slot in composed.variable_slots) or 'none'}",
        ]
        return HierarchicalPatternCandidate(
            fingerprint=fingerprint,
            subpatterns=subpatterns,
            variable_slots=composed.variable_slots,
            notes=notes,
        )

    def find_best_pattern(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        registry: "BrainHierarchicalPatternRegistry",
    ) -> HierarchicalPatternMatch:
        candidate = self.extract_subpatterns(memory_bundle)
        memory = self.codec.decode_memory_bundle(memory_bundle)
        exact_pattern = registry.find_pattern_by_fingerprint(candidate.fingerprint, active_only=True)

        matched_subpattern_ids: List[int] = []
        missing_subpattern_fingerprints: List[str] = []
        for subpattern in candidate.subpatterns:
            existing = registry.find_subpattern_by_fingerprint(subpattern.fingerprint, active_only=True)
            if existing is None:
                missing_subpattern_fingerprints.append(subpattern.fingerprint)
            else:
                matched_subpattern_ids.append(existing.subpattern_id)

        composed = self._compose_subpatterns(candidate.subpatterns)
        slot_values = self._extract_slot_values(memory, composed.variable_slots)

        if exact_pattern is not None:
            return HierarchicalPatternMatch(
                pattern_id=exact_pattern.pattern_id,
                pattern_name=exact_pattern.name,
                fingerprint=exact_pattern.fingerprint,
                reuse_mode="pattern",
                score=1.0,
                matched_subpattern_ids=matched_subpattern_ids,
                missing_subpattern_fingerprints=missing_subpattern_fingerprints,
                variable_slots=slot_values,
                reasons=[
                    "exact_hierarchical_pattern_match",
                    f"pattern_id={exact_pattern.pattern_id}",
                    f"matched_subpatterns={len(matched_subpattern_ids)}",
                ],
            )

        if matched_subpattern_ids:
            return HierarchicalPatternMatch(
                pattern_id=None,
                pattern_name="",
                fingerprint=candidate.fingerprint,
                reuse_mode="subpatterns",
                score=round(len(matched_subpattern_ids) / max(1, len(candidate.subpatterns)), 4),
                matched_subpattern_ids=matched_subpattern_ids,
                missing_subpattern_fingerprints=missing_subpattern_fingerprints,
                variable_slots=slot_values,
                reasons=[
                    "partial_subpattern_reuse",
                    f"matched_subpatterns={len(matched_subpattern_ids)}",
                    f"missing_subpatterns={len(missing_subpattern_fingerprints)}",
                ],
            )

        return HierarchicalPatternMatch(
            pattern_id=None,
            pattern_name="",
            fingerprint=candidate.fingerprint,
            reuse_mode="none",
            score=0.0,
            matched_subpattern_ids=[],
            missing_subpattern_fingerprints=missing_subpattern_fingerprints,
            variable_slots=slot_values,
            reasons=["no_hierarchical_reuse_match"],
        )

    def encode_memory_bundle_hierarchical(
        self,
        current_bundle: Dict[str, List[List[List[int]]]],
        registry: "BrainHierarchicalPatternRegistry",
        pattern: Optional[HierarchicalPatternDefinition] = None,
        subpattern_ids: Optional[Sequence[int]] = None,
    ) -> HierarchicalEncodedBundle:
        memory = self.codec.decode_memory_bundle(current_bundle)

        if pattern is not None:
            composed = self.compose_pattern(pattern, registry)
            slot_values = self._extract_slot_values(memory, composed.variable_slots)
            pattern_id_width = self._primitive_id_width([pattern.pattern_id])
            primitive_ids = [
                self._slot_value_to_id(slot, slot_values.get(slot.slot_name, ""))
                for slot in composed.variable_slots
            ]
            primitive_width = self._primitive_id_width(primitive_ids)
            flags = self._make_flags(
                mode=MODE_PATTERN,
                ref_width=pattern_id_width,
                primitive_width=primitive_width,
            )
            parts = [
                HIERARCHICAL_MAGIC,
                bytes([flags]),
                self._pack_primitive_id(pattern.pattern_id, pattern_id_width),
            ]
            for primitive_id in primitive_ids:
                parts.append(self._pack_primitive_id(primitive_id, primitive_width))
            payload = b"".join(parts)
            return HierarchicalEncodedBundle(
                mode="pattern",
                pattern_id=pattern.pattern_id,
                subpattern_ids=list(pattern.subpattern_ids),
                slot_values=slot_values,
                new_primitives=self._diff_composed_primitives(current_bundle, composed, slot_values),
                payload=payload,
            )

        if not subpattern_ids:
            raise BrainHierarchicalPatternCodecError("Hierarchical encoding requires a pattern or subpattern IDs.")

        ordered_ids = [int(item) for item in subpattern_ids]
        ordered_subpatterns = [registry.get_subpattern(subpattern_id) for subpattern_id in ordered_ids]
        composed = self._compose_subpatterns(ordered_subpatterns)
        slot_values = self._extract_slot_values(memory, composed.variable_slots)
        subpattern_id_width = self._primitive_id_width(ordered_ids)
        primitive_ids = [
            self._slot_value_to_id(slot, slot_values.get(slot.slot_name, ""))
            for slot in composed.variable_slots
        ]
        primitive_width = self._primitive_id_width(primitive_ids)
        if len(ordered_ids) > 0xFF:
            raise BrainHierarchicalPatternCodecError("Hierarchical subpattern payload supports at most 255 subpatterns.")
        flags = self._make_flags(
            mode=MODE_SUBPATTERNS,
            ref_width=subpattern_id_width,
            primitive_width=primitive_width,
        )
        parts = [
            HIERARCHICAL_MAGIC,
            bytes([flags, len(ordered_ids)]),
        ]
        for subpattern_id in ordered_ids:
            parts.append(self._pack_primitive_id(subpattern_id, subpattern_id_width))
        for primitive_id in primitive_ids:
            parts.append(self._pack_primitive_id(primitive_id, primitive_width))
        payload = b"".join(parts)
        return HierarchicalEncodedBundle(
            mode="subpatterns",
            pattern_id=None,
            subpattern_ids=ordered_ids,
            slot_values=slot_values,
            new_primitives=self._diff_composed_primitives(current_bundle, composed, slot_values),
            payload=payload,
        )

    def decode_memory_bundle_hierarchical(
        self,
        payload: bytes | HierarchicalEncodedBundle,
        registry: "BrainHierarchicalPatternRegistry",
    ) -> Dict[str, List[List[List[int]]]]:
        raw_payload = payload.payload if isinstance(payload, HierarchicalEncodedBundle) else payload
        if len(raw_payload) < 4:
            raise BrainHierarchicalPatternCodecError("Hierarchical payload is too short.")
        if raw_payload[:2] != HIERARCHICAL_MAGIC:
            raise BrainHierarchicalPatternCodecError("Hierarchical payload magic header is invalid.")

        flags = raw_payload[2]
        version, mode, ref_width, primitive_width = self._parse_flags(flags)
        if version != HIERARCHICAL_VERSION:
            raise BrainHierarchicalPatternCodecError(f"Unsupported hierarchical payload version: {version}")

        offset = 3
        if mode == MODE_PATTERN:
            pattern_id, offset = self._unpack_primitive_id(raw_payload, offset, ref_width)
            pattern = registry.get_pattern(pattern_id)
            composed = self.compose_pattern(pattern, registry)
        elif mode == MODE_SUBPATTERNS:
            if offset >= len(raw_payload):
                raise BrainHierarchicalPatternCodecError("Hierarchical payload ended while reading subpattern count.")
            subpattern_count = raw_payload[offset]
            offset += 1
            subpattern_ids: List[int] = []
            for _ in range(subpattern_count):
                subpattern_id, offset = self._unpack_primitive_id(raw_payload, offset, ref_width)
                subpattern_ids.append(subpattern_id)
            composed = self._compose_subpatterns(
                [registry.get_subpattern(subpattern_id) for subpattern_id in subpattern_ids]
            )
        else:
            raise BrainHierarchicalPatternCodecError(f"Unsupported hierarchical payload mode: {mode}")

        slot_values: Dict[str, str] = {}
        for slot in composed.variable_slots:
            primitive_id, offset = self._unpack_primitive_id(raw_payload, offset, primitive_width)
            slot_values[slot.slot_name] = self._primitive_id_to_slot_value(slot, primitive_id)

        if offset != len(raw_payload):
            raise BrainHierarchicalPatternCodecError("Hierarchical payload contains trailing bytes.")

        memory = self._reconstruct_from_composed(composed, slot_values)
        return self.codec.encode_memory_bundle(memory)

    def compose_pattern(
        self,
        pattern: HierarchicalPatternDefinition,
        registry: "BrainHierarchicalPatternRegistry",
    ) -> _ComposedHierarchicalStructure:
        subpatterns = [registry.get_subpattern(subpattern_id) for subpattern_id in pattern.subpattern_ids]
        return self._compose_subpatterns(subpatterns)

    def _collect_stable_fields(self, memory: object) -> Dict[str, str]:
        stable_fields: Dict[str, str] = {}
        for field in SINGLE_FIELDS:
            if field in VARIABLE_SINGLE_FIELD_TYPES:
                continue
            value = str(getattr(memory, field) or "")
            if value:
                stable_fields[field] = value
        return stable_fields

    def _collect_field_slots(self, memory: object) -> List[PatternSlot]:
        slots: List[PatternSlot] = []
        slot_index = 1

        for field in SINGLE_FIELDS:
            if field not in VARIABLE_SINGLE_FIELD_TYPES:
                continue
            value = str(getattr(memory, field) or "")
            if not value:
                continue
            slots.append(
                self._build_slot(
                    slot_index=slot_index,
                    source_kind="field",
                    path=field,
                    primitive_type=VARIABLE_SINGLE_FIELD_TYPES[field],
                    description=f"{field} variable slot",
                )
            )
            slot_index += 1

        for field in LIST_FIELDS:
            primitive_type = VARIABLE_LIST_FIELD_TYPES[field]
            for value_index, value in enumerate(list(getattr(memory, field))):
                if not value:
                    continue
                slots.append(
                    self._build_slot(
                        slot_index=slot_index,
                        source_kind="field",
                        path=f"{field}[{value_index}]",
                        primitive_type=primitive_type,
                        description=f"{field}[{value_index}] variable slot",
                    )
                )
                slot_index += 1

        return slots

    def _extract_relation_templates(
        self,
        memory: object,
        stable_fields: Dict[str, str],
        slot_by_name: Dict[str, PatternSlot],
        slot_name_by_key: Dict[Tuple[str, str], str],
        variable_slots: List[PatternSlot],
    ) -> List[PatternRelationTemplate]:
        relation_templates: List[PatternRelationTemplate] = []
        next_slot_index = len(variable_slots) + 1

        for relation_index, relation in enumerate(list(getattr(memory, "relations"))):
            relation_type = str(relation.get("type", "") or "")
            source_mode, source_value, next_slot_index = self._canonicalize_relation_endpoint(
                value=str(relation.get("source", "") or ""),
                relation_type=relation_type,
                endpoint_role="source",
                relation_index=relation_index,
                stable_fields=stable_fields,
                slot_by_name=slot_by_name,
                slot_name_by_key=slot_name_by_key,
                variable_slots=variable_slots,
                next_slot_index=next_slot_index,
            )
            target_mode, target_value, next_slot_index = self._canonicalize_relation_endpoint(
                value=str(relation.get("target", "") or ""),
                relation_type=relation_type,
                endpoint_role="target",
                relation_index=relation_index,
                stable_fields=stable_fields,
                slot_by_name=slot_by_name,
                slot_name_by_key=slot_name_by_key,
                variable_slots=variable_slots,
                next_slot_index=next_slot_index,
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

        return relation_templates

    def _canonicalize_relation_endpoint(
        self,
        value: str,
        relation_type: str,
        endpoint_role: str,
        relation_index: int,
        stable_fields: Dict[str, str],
        slot_by_name: Dict[str, PatternSlot],
        slot_name_by_key: Dict[Tuple[str, str], str],
        variable_slots: List[PatternSlot],
        next_slot_index: int,
    ) -> Tuple[str, str, int]:
        if not value:
            return "literal", "", next_slot_index

        stable_field = self._stable_field_reference(value, stable_fields)
        if stable_field is not None:
            return "field_ref", stable_field, next_slot_index

        primitive_type = self._infer_relation_primitive_type(value, relation_type, endpoint_role)
        existing_slot_name = slot_name_by_key.get((primitive_type, value))
        if existing_slot_name is not None:
            return "slot", existing_slot_name, next_slot_index

        if primitive_type in VARIABLE_RELATION_PRIMITIVE_TYPES:
            slot = self._build_slot(
                slot_index=next_slot_index,
                source_kind="relation",
                path=f"relations[{relation_index}].{endpoint_role}",
                primitive_type=primitive_type,
                description=f"{relation_type}.{endpoint_role} variable slot",
            )
            variable_slots.append(slot)
            slot_by_name[slot.slot_name] = slot
            slot_name_by_key[(primitive_type, value)] = slot.slot_name
            return "slot", slot.slot_name, next_slot_index + 1

        return "literal", value, next_slot_index

    def _stable_field_reference(
        self,
        value: str,
        stable_fields: Dict[str, str],
    ) -> Optional[str]:
        for field in SINGLE_FIELDS:
            if stable_fields.get(field) == value:
                return field
        return None

    def _infer_relation_primitive_type(
        self,
        value: str,
        relation_type: str,
        endpoint_role: str,
    ) -> str:
        if relation_type in {"compare_left", "compare_right"} and endpoint_role == "target":
            return "compare_entity"
        primitive = self.codec._guess_relation_endpoint_primitive(
            value=value,
            relation_type=relation_type,
            endpoint_role=endpoint_role,
        )
        return primitive.split(":", 1)[0]

    def _make_subpattern_candidate(
        self,
        kind: str,
        name: str,
        stable_fields: Dict[str, str],
        variable_slots: Sequence[PatternSlot],
        relation_templates: Sequence[PatternRelationTemplate],
        notes: Sequence[str],
    ) -> HierarchicalSubpatternDefinition:
        normalized_slots = self._normalize_slots(variable_slots)
        normalized_relations = list(relation_templates)
        fingerprint = self._fingerprint_subpattern(
            kind=kind,
            stable_fields=stable_fields,
            variable_slots=normalized_slots,
            relation_templates=normalized_relations,
        )
        return HierarchicalSubpatternDefinition(
            subpattern_id=0,
            kind=kind,
            name=name,
            fingerprint=fingerprint,
            created_sequence=0,
            usage_count=0,
            stable_fields=dict(stable_fields),
            variable_slots=normalized_slots,
            relation_templates=normalized_relations,
            notes=list(notes),
        )

    def _compose_subpatterns(
        self,
        subpatterns: Sequence[HierarchicalSubpatternDefinition],
    ) -> _ComposedHierarchicalStructure:
        stable_fields: Dict[str, str] = {}
        slots_by_name: Dict[str, PatternSlot] = {}
        relation_templates: List[PatternRelationTemplate] = []

        for subpattern in subpatterns:
            for field, value in subpattern.stable_fields.items():
                existing = stable_fields.get(field)
                if existing is not None and existing != value:
                    raise BrainHierarchicalPatternCodecError(
                        f"Conflicting stable field values for {field}: {existing} vs {value}"
                    )
                stable_fields[field] = value

            for slot in subpattern.variable_slots:
                existing_slot = slots_by_name.get(slot.slot_name)
                if existing_slot is not None:
                    if (
                        existing_slot.path != slot.path
                        or existing_slot.primitive_type != slot.primitive_type
                        or existing_slot.source_kind != slot.source_kind
                    ):
                        raise BrainHierarchicalPatternCodecError(
                            f"Conflicting slot definitions for {slot.slot_name}."
                        )
                    if existing_slot.optional != slot.optional:
                        slots_by_name[slot.slot_name] = PatternSlot(
                            slot_name=existing_slot.slot_name,
                            slot_index=existing_slot.slot_index,
                            source_kind=existing_slot.source_kind,
                            path=existing_slot.path,
                            primitive_type=existing_slot.primitive_type,
                            description=existing_slot.description,
                            optional=existing_slot.optional and slot.optional,
                        )
                    continue
                slots_by_name[slot.slot_name] = slot

            relation_templates.extend(subpattern.relation_templates)

        normalized_slots = self._normalize_slots(slots_by_name.values())
        return _ComposedHierarchicalStructure(
            stable_fields=stable_fields,
            variable_slots=normalized_slots,
            relation_templates=relation_templates,
        )

    def _normalize_slots(
        self,
        slots: Sequence[PatternSlot],
    ) -> List[PatternSlot]:
        normalized: List[PatternSlot] = []
        for index, slot in enumerate(sorted(slots, key=self._slot_sort_key), start=1):
            normalized.append(
                PatternSlot(
                    slot_name=slot.slot_name,
                    slot_index=index,
                    source_kind=slot.source_kind,
                    path=slot.path,
                    primitive_type=slot.primitive_type,
                    description=slot.description,
                    optional=slot.optional,
                )
            )
        return normalized

    @staticmethod
    def _slot_sort_key(slot: PatternSlot) -> Tuple[int, str]:
        return (
            0 if slot.source_kind == "field" else 1,
            slot.path,
            slot.primitive_type,
        )

    @staticmethod
    def _relation_references_slot(template: PatternRelationTemplate, slot_name: str) -> bool:
        return (
            (template.source_mode == "slot" and template.source_value == slot_name)
            or (template.target_mode == "slot" and template.target_value == slot_name)
        )

    def _reconstruct_from_composed(
        self,
        composed: _ComposedHierarchicalStructure,
        slot_values: Dict[str, str],
    ) -> BrainMemory:
        memory = BrainMemory()

        for field, value in composed.stable_fields.items():
            setattr(memory, field, value)

        for field in LIST_FIELDS:
            setattr(memory, field, [])

        for slot in composed.variable_slots:
            if slot.source_kind != "field":
                continue
            value = slot_values.get(slot.slot_name, "")
            self._write_path(memory, slot.path, value)

        for field in LIST_FIELDS:
            values = [value for value in list(getattr(memory, field)) if value]
            setattr(memory, field, values)

        relations: List[Dict[str, str]] = []
        for template in sorted(composed.relation_templates, key=lambda item: item.relation_index):
            source = self._resolve_relation_endpoint_value(memory, template.source_mode, template.source_value, slot_values)
            target = self._resolve_relation_endpoint_value(memory, template.target_mode, template.target_value, slot_values)
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

    def _resolve_relation_endpoint_value(
        self,
        memory: object,
        mode: str,
        value: str,
        slot_values: Dict[str, str],
    ) -> str:
        if mode == "literal":
            return value
        if mode == "field_ref":
            return str(getattr(memory, value, "") or "")
        if mode == "slot":
            return str(slot_values.get(value, "") or "")
        raise BrainHierarchicalPatternCodecError(f"Unsupported relation endpoint mode: {mode}")

    def _fingerprint_subpattern(
        self,
        kind: str,
        stable_fields: Dict[str, str],
        variable_slots: Sequence[PatternSlot],
        relation_templates: Sequence[PatternRelationTemplate],
    ) -> str:
        payload = {
            "kind": kind,
            "stable_fields": {key: stable_fields[key] for key in sorted(stable_fields)},
            "variable_slots": [
                {
                    "source_kind": slot.source_kind,
                    "path": slot.path,
                    "primitive_type": slot.primitive_type,
                    **({"optional": True} if slot.optional else {}),
                }
                for slot in self._normalize_slots(variable_slots)
            ],
            "relation_templates": [
                {
                    "relation_type": template.relation_type,
                    "source_mode": template.source_mode,
                    "source_value": template.source_value,
                    "target_mode": template.target_mode,
                    "target_value": template.target_value,
                    **({"optional": True} if template.optional else {}),
                }
                for template in relation_templates
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _fingerprint_hierarchical_candidate(
        self,
        subpatterns: Sequence[HierarchicalSubpatternDefinition],
    ) -> str:
        composed = self._compose_subpatterns(subpatterns)
        payload = {
            "subpatterns": [subpattern.fingerprint for subpattern in subpatterns],
            "variable_slots": [
                {
                    "slot_name": slot.slot_name,
                    "path": slot.path,
                    "primitive_type": slot.primitive_type,
                    "source_kind": slot.source_kind,
                    **({"optional": True} if slot.optional else {}),
                }
                for slot in composed.variable_slots
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _diff_composed_primitives(
        self,
        current_bundle: Dict[str, List[List[List[int]]]],
        composed: _ComposedHierarchicalStructure,
        slot_values: Dict[str, str],
    ) -> List[str]:
        base_ids: set[int] = set()
        for field, value in composed.stable_fields.items():
            base_ids.add(self.binary_codec.primitive_id(self.codec.make_primitive(field, value)))

        for template in composed.relation_templates:
            base_ids.add(self.binary_codec.primitive_id(self.codec.make_primitive("relation_type", template.relation_type)))
            if template.source_mode == "literal" and template.source_value:
                primitive = self.codec._guess_relation_endpoint_primitive(
                    value=template.source_value,
                    relation_type=template.relation_type,
                    endpoint_role="source",
                )
                base_ids.add(self.binary_codec.primitive_id(primitive))
            if template.target_mode == "literal" and template.target_value:
                primitive = self.codec._guess_relation_endpoint_primitive(
                    value=template.target_value,
                    relation_type=template.relation_type,
                    endpoint_role="target",
                )
                base_ids.add(self.binary_codec.primitive_id(primitive))

        current_ids = self._bundle_primitive_ids(current_bundle)
        slot_ids = {
            self._slot_value_to_id(slot, slot_values.get(slot.slot_name, ""))
            for slot in composed.variable_slots
        }
        relevant_ids = sorted(
            primitive_id
            for primitive_id in slot_ids
            if primitive_id not in base_ids and primitive_id != 0
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

    def _make_flags(self, mode: int, ref_width: int, primitive_width: int) -> int:
        return (
            ((HIERARCHICAL_VERSION & 0x03) << 6)
            | ((self._width_code(ref_width) & 0x03) << 4)
            | ((self._width_code(primitive_width) & 0x03) << 2)
            | (mode & 0x03)
        )

    def _parse_flags(self, flags: int) -> Tuple[int, int, int, int]:
        version = (flags >> 6) & 0x03
        ref_width = self._code_width((flags >> 4) & 0x03)
        primitive_width = self._code_width((flags >> 2) & 0x03)
        mode = flags & 0x03
        return version, mode, ref_width, primitive_width

    @staticmethod
    def _width_code(width: int) -> int:
        if width == 1:
            return 0
        if width == 2:
            return 1
        if width == 4:
            return 2
        raise BrainHierarchicalPatternCodecError(f"Unsupported hierarchical width: {width}")

    @staticmethod
    def _code_width(code: int) -> int:
        if code == 0:
            return 1
        if code == 1:
            return 2
        if code == 2:
            return 4
        raise BrainHierarchicalPatternCodecError(f"Unsupported hierarchical width code: {code}")


class BrainHierarchicalPatternRegistry:
    """
    Persistent registry for hierarchical BrAIn subpatterns and composed patterns.
    """

    def __init__(
        self,
        codec: BrainHierarchicalPatternCodec,
        registry_path: str | Path = "brain_hierarchical_patterns.json",
    ) -> None:
        self.codec = codec
        self.registry_path = Path(registry_path)
        self.subpatterns: Dict[int, HierarchicalSubpatternDefinition] = {}
        self.patterns: Dict[int, HierarchicalPatternDefinition] = {}
        self.subpattern_cooccurrence: Dict[str, int] = {}
        self.next_subpattern_id = 1
        self.next_pattern_id = 1
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            self.subpatterns = {}
            self.patterns = {}
            self.subpattern_cooccurrence = {}
            self.next_subpattern_id = 1
            self.next_pattern_id = 1
            return

        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_subpatterns: Dict[int, HierarchicalSubpatternDefinition] = {}
        loaded_patterns: Dict[int, HierarchicalPatternDefinition] = {}
        loaded_cooccurrence: Dict[str, int] = {}
        if isinstance(data, dict):
            self.next_subpattern_id = int(data.get("next_subpattern_id", 1))
            self.next_pattern_id = int(data.get("next_pattern_id", 1))

            raw_subpatterns = data.get("subpatterns", [])
            if isinstance(raw_subpatterns, list):
                for item in raw_subpatterns:
                    if not isinstance(item, dict):
                        continue
                    subpattern = HierarchicalSubpatternDefinition.from_dict(item)
                    if subpattern.subpattern_id > 0:
                        loaded_subpatterns[subpattern.subpattern_id] = subpattern

            raw_patterns = data.get("patterns", [])
            if isinstance(raw_patterns, list):
                for item in raw_patterns:
                    if not isinstance(item, dict):
                        continue
                    pattern = HierarchicalPatternDefinition.from_dict(item)
                    if pattern.pattern_id > 0:
                        loaded_patterns[pattern.pattern_id] = pattern

            raw_cooccurrence = data.get("subpattern_cooccurrence", {})
            if isinstance(raw_cooccurrence, dict):
                loaded_cooccurrence = {
                    str(key): int(value)
                    for key, value in raw_cooccurrence.items()
                    if isinstance(value, int) and int(value) > 0
                }

        self.subpatterns = loaded_subpatterns
        self.patterns = loaded_patterns
        self.subpattern_cooccurrence = loaded_cooccurrence
        self.next_subpattern_id = max([self.next_subpattern_id] + [item + 1 for item in self.subpatterns], default=1)
        self.next_pattern_id = max([self.next_pattern_id] + [item + 1 for item in self.patterns], default=1)

    def save(self) -> None:
        payload = {
            "next_subpattern_id": self.next_subpattern_id,
            "next_pattern_id": self.next_pattern_id,
            "subpatterns": [
                self.subpatterns[subpattern_id].to_dict()
                for subpattern_id in sorted(self.subpatterns)
            ],
            "patterns": [
                self.patterns[pattern_id].to_dict()
                for pattern_id in sorted(self.patterns)
            ],
            "subpattern_cooccurrence": {
                key: self.subpattern_cooccurrence[key]
                for key in sorted(self.subpattern_cooccurrence)
            },
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def all_subpatterns(self) -> List[HierarchicalSubpatternDefinition]:
        return [self.subpatterns[subpattern_id] for subpattern_id in sorted(self.subpatterns)]

    def all_patterns(self) -> List[HierarchicalPatternDefinition]:
        return [self.patterns[pattern_id] for pattern_id in sorted(self.patterns)]

    def all_active_subpatterns(self) -> List[HierarchicalSubpatternDefinition]:
        return [
            self.subpatterns[subpattern_id]
            for subpattern_id in sorted(self.subpatterns)
            if self.subpatterns[subpattern_id].active
        ]

    def all_active_patterns(self) -> List[HierarchicalPatternDefinition]:
        return [
            self.patterns[pattern_id]
            for pattern_id in sorted(self.patterns)
            if self.patterns[pattern_id].active
        ]

    def get_subpattern(self, subpattern_id: int) -> HierarchicalSubpatternDefinition:
        subpattern = self.subpatterns.get(subpattern_id)
        if subpattern is None:
            raise BrainHierarchicalPatternCodecError(f"Unknown subpattern ID: {subpattern_id}")
        return subpattern

    def get_pattern(self, pattern_id: int) -> HierarchicalPatternDefinition:
        pattern = self.patterns.get(pattern_id)
        if pattern is None:
            raise BrainHierarchicalPatternCodecError(f"Unknown hierarchical pattern ID: {pattern_id}")
        return pattern

    def find_subpattern_by_fingerprint(
        self,
        fingerprint: str,
        active_only: bool = False,
    ) -> Optional[HierarchicalSubpatternDefinition]:
        for subpattern_id in sorted(self.subpatterns):
            subpattern = self.subpatterns[subpattern_id]
            if active_only and not subpattern.active:
                continue
            if subpattern.fingerprint == fingerprint:
                return subpattern
        return None

    def find_pattern_by_fingerprint(
        self,
        fingerprint: str,
        active_only: bool = False,
    ) -> Optional[HierarchicalPatternDefinition]:
        for pattern_id in sorted(self.patterns):
            pattern = self.patterns[pattern_id]
            if active_only and not pattern.active:
                continue
            if pattern.fingerprint == fingerprint:
                return pattern
        return None

    def ensure_subpattern(
        self,
        candidate: HierarchicalSubpatternDefinition,
    ) -> Tuple[HierarchicalSubpatternDefinition, bool]:
        existing = self.find_subpattern_by_fingerprint(candidate.fingerprint)
        if existing is not None:
            return existing, False

        subpattern_id = self.next_subpattern_id
        created = HierarchicalSubpatternDefinition(
            subpattern_id=subpattern_id,
            kind=candidate.kind,
            name=f"subpattern_{subpattern_id:03d}_{candidate.kind}",
            fingerprint=candidate.fingerprint,
            created_sequence=subpattern_id,
            usage_count=1,
            stable_fields=candidate.stable_fields,
            variable_slots=candidate.variable_slots,
            relation_templates=candidate.relation_templates,
            notes=candidate.notes,
            active=candidate.active,
            generalization_level=candidate.generalization_level,
            merged_from_subpattern_ids=list(candidate.merged_from_subpattern_ids),
            last_used_sequence=max(candidate.last_used_sequence, subpattern_id),
            reuse_frequency=candidate.reuse_frequency,
            cooccurrence_count=candidate.cooccurrence_count,
            temporal_score=candidate.temporal_score,
        )
        self.subpatterns[subpattern_id] = created
        self.next_subpattern_id += 1
        self.save()
        return created, True

    def ensure_pattern(
        self,
        candidate: HierarchicalPatternCandidate,
        subpattern_ids: Sequence[int],
    ) -> Tuple[HierarchicalPatternDefinition, bool]:
        existing = self.find_pattern_by_fingerprint(candidate.fingerprint)
        if existing is not None:
            return existing, False

        pattern_id = self.next_pattern_id
        created = HierarchicalPatternDefinition(
            pattern_id=pattern_id,
            name=f"hier_pattern_{pattern_id:03d}",
            fingerprint=candidate.fingerprint,
            created_sequence=pattern_id,
            usage_count=1,
            subpattern_ids=[int(item) for item in subpattern_ids],
            variable_slots=list(candidate.variable_slots),
            notes=list(candidate.notes),
            active=True,
            generalization_level=0,
            generalized_slot_count=0,
            merged_from_pattern_ids=[],
            superseded_by_pattern_id=None,
            last_used_sequence=pattern_id,
            reuse_frequency=1.0,
            total_encoded_size_bytes=0,
            total_full_size_bytes=0,
            average_compression_ratio=0.0,
            success_count=0,
            failure_count=0,
            success_rate=0.0,
            compression_efficiency=0.0,
            temporal_score=0.0,
        )
        self.patterns[pattern_id] = created
        self.next_pattern_id += 1
        self.save()
        return created, True

    def record_subpattern_usage(
        self,
        subpattern_ids: Sequence[int],
        increment: int = 1,
        sequence: Optional[int] = None,
        temporal_score: Optional[float] = None,
    ) -> None:
        updated = False
        for subpattern_id in subpattern_ids:
            subpattern = self.subpatterns.get(int(subpattern_id))
            if subpattern is None:
                continue
            next_usage_count = subpattern.usage_count + increment
            last_used_sequence = max(subpattern.last_used_sequence, sequence or subpattern.last_used_sequence)
            reuse_frequency = round(
                next_usage_count / max(1, last_used_sequence or subpattern.created_sequence or next_usage_count),
                4,
            )
            self.subpatterns[int(subpattern_id)] = HierarchicalSubpatternDefinition(
                subpattern_id=subpattern.subpattern_id,
                kind=subpattern.kind,
                name=subpattern.name,
                fingerprint=subpattern.fingerprint,
                created_sequence=subpattern.created_sequence,
                usage_count=next_usage_count,
                stable_fields=subpattern.stable_fields,
                variable_slots=subpattern.variable_slots,
                relation_templates=subpattern.relation_templates,
                notes=subpattern.notes,
                active=subpattern.active,
                generalization_level=subpattern.generalization_level,
                merged_from_subpattern_ids=list(subpattern.merged_from_subpattern_ids),
                last_used_sequence=last_used_sequence,
                reuse_frequency=reuse_frequency,
                cooccurrence_count=subpattern.cooccurrence_count,
                temporal_score=subpattern.temporal_score if temporal_score is None else max(0.0, temporal_score),
            )
            updated = True
        if updated:
            self.save()

    def record_pattern_usage(
        self,
        pattern_id: int,
        increment: int = 1,
        sequence: Optional[int] = None,
        encoded_size_bytes: int = 0,
        full_size_bytes: int = 0,
        temporal_score: Optional[float] = None,
    ) -> None:
        pattern = self.patterns.get(pattern_id)
        if pattern is None:
            return
        next_usage_count = pattern.usage_count + increment
        last_used_sequence = max(pattern.last_used_sequence, sequence or pattern.last_used_sequence)
        total_encoded_size_bytes = pattern.total_encoded_size_bytes + max(0, encoded_size_bytes)
        total_full_size_bytes = pattern.total_full_size_bytes + max(0, full_size_bytes)
        average_compression_ratio = round(
            total_encoded_size_bytes / total_full_size_bytes,
            4,
        ) if total_full_size_bytes > 0 else pattern.average_compression_ratio
        compression_efficiency = round(
            max(0.0, min(1.0, 1.0 - average_compression_ratio)),
            4,
        ) if total_full_size_bytes > 0 else pattern.compression_efficiency
        reuse_frequency = round(
            next_usage_count / max(1, last_used_sequence or pattern.created_sequence or next_usage_count),
            4,
        )
        self.patterns[pattern_id] = HierarchicalPatternDefinition(
            pattern_id=pattern.pattern_id,
            name=pattern.name,
            fingerprint=pattern.fingerprint,
            created_sequence=pattern.created_sequence,
            usage_count=next_usage_count,
            subpattern_ids=pattern.subpattern_ids,
            variable_slots=pattern.variable_slots,
            notes=pattern.notes,
            active=pattern.active,
            generalization_level=pattern.generalization_level,
            generalized_slot_count=pattern.generalized_slot_count,
            merged_from_pattern_ids=list(pattern.merged_from_pattern_ids),
            superseded_by_pattern_id=pattern.superseded_by_pattern_id,
            last_used_sequence=last_used_sequence,
            reuse_frequency=reuse_frequency,
            total_encoded_size_bytes=total_encoded_size_bytes,
            total_full_size_bytes=total_full_size_bytes,
            average_compression_ratio=average_compression_ratio,
            success_count=pattern.success_count,
            failure_count=pattern.failure_count,
            success_rate=pattern.success_rate,
            compression_efficiency=compression_efficiency,
            temporal_score=pattern.temporal_score if temporal_score is None else max(0.0, temporal_score),
        )
        self.save()

    def record_pattern_outcome(
        self,
        pattern_id: int,
        success: bool,
        increment: int = 1,
        temporal_score: Optional[float] = None,
    ) -> None:
        pattern = self.patterns.get(pattern_id)
        if pattern is None:
            return
        success_count = pattern.success_count + (increment if success else 0)
        failure_count = pattern.failure_count + (increment if not success else 0)
        total_outcomes = success_count + failure_count
        success_rate = round(success_count / max(1, total_outcomes), 4)
        self.patterns[pattern_id] = HierarchicalPatternDefinition(
            pattern_id=pattern.pattern_id,
            name=pattern.name,
            fingerprint=pattern.fingerprint,
            created_sequence=pattern.created_sequence,
            usage_count=pattern.usage_count,
            subpattern_ids=pattern.subpattern_ids,
            variable_slots=pattern.variable_slots,
            notes=pattern.notes,
            active=pattern.active,
            generalization_level=pattern.generalization_level,
            generalized_slot_count=pattern.generalized_slot_count,
            merged_from_pattern_ids=list(pattern.merged_from_pattern_ids),
            superseded_by_pattern_id=pattern.superseded_by_pattern_id,
            last_used_sequence=pattern.last_used_sequence,
            reuse_frequency=pattern.reuse_frequency,
            total_encoded_size_bytes=pattern.total_encoded_size_bytes,
            total_full_size_bytes=pattern.total_full_size_bytes,
            average_compression_ratio=pattern.average_compression_ratio,
            success_count=success_count,
            failure_count=failure_count,
            success_rate=success_rate,
            compression_efficiency=pattern.compression_efficiency,
            temporal_score=pattern.temporal_score if temporal_score is None else max(0.0, temporal_score),
        )
        self.save()

    def record_subpattern_cooccurrence(
        self,
        subpattern_ids: Sequence[int],
        increment: int = 1,
        sequence: Optional[int] = None,
    ) -> None:
        ordered_ids = sorted({int(item) for item in subpattern_ids if int(item) in self.subpatterns})
        if len(ordered_ids) < 2:
            return
        for index, left_id in enumerate(ordered_ids):
            for right_id in ordered_ids[index + 1:]:
                key = self._cooccurrence_key(left_id, right_id)
                self.subpattern_cooccurrence[key] = self.subpattern_cooccurrence.get(key, 0) + increment
        for subpattern_id in ordered_ids:
            subpattern = self.subpatterns.get(subpattern_id)
            if subpattern is None:
                continue
            self.subpatterns[subpattern_id] = HierarchicalSubpatternDefinition(
                subpattern_id=subpattern.subpattern_id,
                kind=subpattern.kind,
                name=subpattern.name,
                fingerprint=subpattern.fingerprint,
                created_sequence=subpattern.created_sequence,
                usage_count=subpattern.usage_count,
                stable_fields=subpattern.stable_fields,
                variable_slots=subpattern.variable_slots,
                relation_templates=subpattern.relation_templates,
                notes=subpattern.notes,
                active=subpattern.active,
                generalization_level=subpattern.generalization_level,
                merged_from_subpattern_ids=list(subpattern.merged_from_subpattern_ids),
                last_used_sequence=max(subpattern.last_used_sequence, sequence or subpattern.last_used_sequence),
                reuse_frequency=subpattern.reuse_frequency,
                cooccurrence_count=subpattern.cooccurrence_count + increment,
                temporal_score=subpattern.temporal_score,
            )
        self.save()

    def update_subpattern(self, subpattern: HierarchicalSubpatternDefinition) -> None:
        self.subpatterns[subpattern.subpattern_id] = subpattern
        self.save()

    def update_pattern(self, pattern: HierarchicalPatternDefinition) -> None:
        self.patterns[pattern.pattern_id] = pattern
        self.save()

    @staticmethod
    def _cooccurrence_key(left_id: int, right_id: int) -> str:
        first, second = sorted((int(left_id), int(right_id)))
        return f"{first}:{second}"


class BrainHierarchicalPatternMemoryStore:
    """
    In-memory storage for full and hierarchical-pattern-encoded BrAIn memories.
    """

    def __init__(
        self,
        codec: BrainHierarchicalPatternCodec,
        registry: BrainHierarchicalPatternRegistry,
    ) -> None:
        self.codec = codec
        self.registry = registry
        self.entries: Dict[int, StoredHierarchicalMemory] = {}
        self.order: List[int] = []

    def add_memory(
        self,
        memory_id: int,
        memory_bundle: Dict[str, List[List[List[int]]]],
    ) -> StoredHierarchicalMemory:
        if memory_id in self.entries:
            raise BrainHierarchicalPatternCodecError(f"Memory ID already exists in hierarchical store: {memory_id}")

        full_payload = self.codec.full_bundle_bytes(memory_bundle)
        candidate = self.codec.extract_subpatterns(memory_bundle)
        match = self.codec.find_best_pattern(memory_bundle, self.registry)

        ensured_subpattern_ids: List[int] = []
        for subpattern in candidate.subpatterns:
            ensured, _ = self.registry.ensure_subpattern(subpattern)
            ensured_subpattern_ids.append(ensured.subpattern_id)

        pattern_definition: Optional[HierarchicalPatternDefinition] = None
        encoded_bundle: Optional[HierarchicalEncodedBundle] = None
        preferred_encoding = "full"

        if match.pattern_id is not None:
            pattern_definition = self.registry.get_pattern(match.pattern_id)
            self.registry.record_pattern_usage(pattern_definition.pattern_id)
            self.registry.record_subpattern_usage(pattern_definition.subpattern_ids)
            self.registry.record_subpattern_cooccurrence(pattern_definition.subpattern_ids)
            encoded_bundle = self.codec.encode_memory_bundle_hierarchical(
                current_bundle=memory_bundle,
                registry=self.registry,
                pattern=pattern_definition,
            )
            preferred_encoding = "pattern"
        elif match.matched_subpattern_ids:
            pattern_definition, _ = self.registry.ensure_pattern(candidate, ensured_subpattern_ids)
            self.registry.record_subpattern_usage(match.matched_subpattern_ids)
            self.registry.record_subpattern_cooccurrence(ensured_subpattern_ids)
            encoded_bundle = self.codec.encode_memory_bundle_hierarchical(
                current_bundle=memory_bundle,
                registry=self.registry,
                subpattern_ids=ensured_subpattern_ids,
            )
            preferred_encoding = "subpatterns"
        else:
            self.registry.ensure_pattern(candidate, ensured_subpattern_ids)
            self.registry.record_subpattern_cooccurrence(ensured_subpattern_ids)

        if encoded_bundle is not None and encoded_bundle.total_bytes <= len(full_payload):
            entry = StoredHierarchicalMemory(
                memory_id=memory_id,
                encoding_type=preferred_encoding,
                payload=encoded_bundle.payload,
                pattern_id=encoded_bundle.pattern_id,
                subpattern_ids=list(encoded_bundle.subpattern_ids),
                resolved_bundle=memory_bundle,
                match=match,
                encoded_bundle=encoded_bundle,
                full_size_bytes=len(full_payload),
                stored_size_bytes=encoded_bundle.total_bytes,
                variable_slot_count=encoded_bundle.variable_slot_count,
                new_primitives=list(encoded_bundle.new_primitives),
            )
        else:
            entry = StoredHierarchicalMemory(
                memory_id=memory_id,
                encoding_type="full",
                payload=full_payload,
                pattern_id=pattern_definition.pattern_id if pattern_definition is not None else None,
                subpattern_ids=list(ensured_subpattern_ids),
                resolved_bundle=memory_bundle,
                match=match,
                encoded_bundle=encoded_bundle,
                full_size_bytes=len(full_payload),
                stored_size_bytes=len(full_payload),
                variable_slot_count=encoded_bundle.variable_slot_count if encoded_bundle is not None else 0,
                new_primitives=list(encoded_bundle.new_primitives) if encoded_bundle is not None else [],
            )

        self.entries[memory_id] = entry
        self.order.append(memory_id)
        return entry

    def resolve_bundle(self, memory_id: int) -> Dict[str, List[List[List[int]]]]:
        entry = self.entries.get(memory_id)
        if entry is None:
            raise BrainHierarchicalPatternCodecError(f"Unknown memory ID: {memory_id}")

        if entry.resolved_bundle is not None:
            return entry.resolved_bundle

        if entry.encoding_type == "full":
            bundle = self.codec.bitpacked_codec.decode_memory_bundle_bitpacked(entry.payload)
            entry.resolved_bundle = bundle
            return bundle

        bundle = self.codec.decode_memory_bundle_hierarchical(entry.payload, self.registry)
        entry.resolved_bundle = bundle
        return bundle


def extract_subpatterns(
    memory_bundle: Dict[str, List[List[List[int]]]],
    codec: Optional[BrainHierarchicalPatternCodec] = None,
) -> HierarchicalPatternCandidate:
    active_codec = codec or BrainHierarchicalPatternCodec(codec=BrainCodec(auto_register=False))
    return active_codec.extract_subpatterns(memory_bundle)


def find_best_pattern(
    memory_bundle: Dict[str, List[List[List[int]]]],
    registry: BrainHierarchicalPatternRegistry,
    codec: Optional[BrainHierarchicalPatternCodec] = None,
) -> HierarchicalPatternMatch:
    active_codec = codec or registry.codec
    return active_codec.find_best_pattern(memory_bundle, registry)


def encode_memory_bundle_hierarchical(
    current_bundle: Dict[str, List[List[List[int]]]],
    registry: BrainHierarchicalPatternRegistry,
    codec: Optional[BrainHierarchicalPatternCodec] = None,
    pattern: Optional[HierarchicalPatternDefinition] = None,
    subpattern_ids: Optional[Sequence[int]] = None,
) -> HierarchicalEncodedBundle:
    active_codec = codec or registry.codec
    return active_codec.encode_memory_bundle_hierarchical(
        current_bundle=current_bundle,
        registry=registry,
        pattern=pattern,
        subpattern_ids=subpattern_ids,
    )


def decode_memory_bundle_hierarchical(
    payload: bytes | HierarchicalEncodedBundle,
    registry: BrainHierarchicalPatternRegistry,
    codec: Optional[BrainHierarchicalPatternCodec] = None,
) -> Dict[str, List[List[List[int]]]]:
    active_codec = codec or registry.codec
    return active_codec.decode_memory_bundle_hierarchical(payload, registry)
