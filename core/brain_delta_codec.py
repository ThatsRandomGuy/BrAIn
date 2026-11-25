from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from brain_bitpacked_codec import BrainBitPackedCodec
from brain_binary_codec import BrainBinaryCodec, EMPTY_ID
from brain_codec import BrainCodec, BrainMemory


DELTA_MAGIC = b"BD"
DELTA_VERSION = 1
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_CORE_STRUCTURE_THRESHOLD = 0.9
MIN_STRUCTURAL_RELATION_SCORE = 0.75

SINGLE_FIELDS = ("actor", "intent", "action", "object", "topic", "time", "priority")
LIST_FIELDS = ("modifiers", "negations", "constraints", "conditions")

SINGLE_FIELD_WEIGHTS = {
    "actor": 0.04,
    "intent": 0.06,
    "action": 0.18,
    "object": 0.18,
    "topic": 0.10,
    "time": 0.02,
    "priority": 0.02,
}

LIST_FIELD_WEIGHTS = {
    "modifiers": 0.05,
    "negations": 0.03,
    "constraints": 0.03,
    "conditions": 0.02,
}

RELATION_COMPONENT_WEIGHTS = {
    "exact_relation_similarity": 0.07,
    "structural_relation_similarity": 0.20,
}

CORE_STRUCTURE_WEIGHTS = {
    "intent": 0.05,
    "action": 0.35,
    "object": 0.35,
    "exact_relation_similarity": 0.05,
    "structural_relation_similarity": 0.20,
}

FIELD_TO_PRIMITIVE_TYPE = {
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


@dataclass(frozen=True)
class ReferenceMatch:
    reference_id: int
    similarity: float
    core_structure_similarity: float
    matched_scalar_fields: List[str]
    changed_scalar_fields: List[str]
    component_scores: Dict[str, float]
    list_field_scores: Dict[str, float]
    exact_relation_similarity: float
    structural_relation_similarity: float
    reuse_basis: str
    reasons: List[str]


@dataclass(frozen=True)
class DeltaBundle:
    reference_id: int
    changed_fields: Dict[str, object]
    added_relations: List[Dict[str, str]]
    removed_relations: List[Dict[str, str]]
    new_primitives: List[str]
    payload: bytes
    relation_mode: str = "patch"
    replacement_relations: List[Dict[str, str]] = field(default_factory=list)
    similarity: float = 0.0

    @property
    def total_bytes(self) -> int:
        return len(self.payload)


@dataclass
class StoredMemory:
    memory_id: int
    encoding_type: str
    payload: bytes
    reference_id: Optional[int]
    resolved_bundle: Optional[Dict[str, List[List[List[int]]]]] = None
    similarity: float = 0.0
    match: Optional[ReferenceMatch] = None
    delta_bundle: Optional[DeltaBundle] = None
    full_size_bytes: int = 0
    stored_size_bytes: int = 0
    new_primitives: List[str] = field(default_factory=list)


class BrainDeltaCodecError(Exception):
    pass


class BrainDeltaCodec:
    """
    Deterministic semantic reuse and delta encoding layer for BrAIn.

    Full memories are stored using the existing bit-packed codec. When a new
    bundle is similar enough to an existing stored memory, BrAIn stores only the
    changed semantic fields and relation differences relative to the reference.
    """

    def __init__(
        self,
        codec: BrainCodec,
        binary_codec: Optional[BrainBinaryCodec] = None,
        bitpacked_codec: Optional[BrainBitPackedCodec] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        core_structure_threshold: float = DEFAULT_CORE_STRUCTURE_THRESHOLD,
    ) -> None:
        self.codec = codec
        self.binary_codec = binary_codec or BrainBinaryCodec(codec=codec)
        self.bitpacked_codec = bitpacked_codec or BrainBitPackedCodec(binary_codec=self.binary_codec)
        self.similarity_threshold = similarity_threshold
        self.core_structure_threshold = core_structure_threshold

    def find_best_reference(
        self,
        memory_bundle: Dict[str, List[List[List[int]]]],
        existing_bundles: Sequence[StoredMemory | Dict[str, object]],
    ) -> Optional[ReferenceMatch]:
        current_memory = self.codec.decode_memory_bundle(memory_bundle)
        best_match: Optional[ReferenceMatch] = None

        for item in existing_bundles:
            reference_id, reference_bundle = self._coerce_reference(item)
            reference_memory = self.codec.decode_memory_bundle(reference_bundle)
            candidate = self._score_reference(current_memory, reference_memory, reference_id)

            if not self._is_reusable_candidate(candidate):
                continue

            if best_match is None:
                best_match = candidate
                continue

            if candidate.core_structure_similarity > best_match.core_structure_similarity:
                best_match = candidate
                continue

            if (
                candidate.core_structure_similarity == best_match.core_structure_similarity
                and candidate.similarity > best_match.similarity
            ):
                best_match = candidate
                continue

            if (
                candidate.core_structure_similarity == best_match.core_structure_similarity
                and candidate.similarity == best_match.similarity
                and candidate.reference_id < best_match.reference_id
            ):
                best_match = candidate

        return best_match

    def encode_memory_bundle_delta(
        self,
        current_bundle: Dict[str, List[List[List[int]]]],
        reference_bundle: Dict[str, List[List[List[int]]]],
        reference_id: int = 0,
        similarity: float = 0.0,
    ) -> DeltaBundle:
        self.binary_codec.sync_from_codec()

        current_memory = self.codec.decode_memory_bundle(current_bundle)
        reference_memory = self.codec.decode_memory_bundle(reference_bundle)

        changed_fields = self._diff_fields(current_memory, reference_memory)
        added_relations, removed_relations = self._diff_relations(current_memory.relations, reference_memory.relations)
        relation_mode, replacement_relations = self._choose_relation_mode(
            current_relations=current_memory.relations,
            added_relations=added_relations,
            removed_relations=removed_relations,
        )
        new_primitives = self._diff_bundle_primitives(current_bundle, reference_bundle)

        primitive_ids = self._collect_delta_primitive_ids(
            changed_fields=changed_fields,
            added_relations=added_relations if relation_mode == "patch" else [],
            removed_relations=removed_relations if relation_mode == "patch" else [],
            replacement_relations=replacement_relations,
            new_primitives=new_primitives,
        )
        primitive_id_bytes = self._primitive_id_width(primitive_ids)

        payload = self._serialize_delta(
            reference_id=reference_id,
            changed_fields=changed_fields,
            added_relations=added_relations if relation_mode == "patch" else [],
            removed_relations=removed_relations if relation_mode == "patch" else [],
            relation_mode=relation_mode,
            replacement_relations=replacement_relations,
            new_primitives=new_primitives,
            primitive_id_bytes=primitive_id_bytes,
        )

        return DeltaBundle(
            reference_id=reference_id,
            changed_fields=changed_fields,
            added_relations=added_relations if relation_mode == "patch" else [],
            removed_relations=removed_relations if relation_mode == "patch" else [],
            new_primitives=new_primitives,
            payload=payload,
            relation_mode=relation_mode,
            replacement_relations=replacement_relations,
            similarity=round(similarity, 4),
        )

    def decode_memory_bundle_delta(
        self,
        delta_bundle: DeltaBundle | bytes,
        reference_bundle: Dict[str, List[List[List[int]]]],
    ) -> Dict[str, List[List[List[int]]]]:
        parsed = delta_bundle if isinstance(delta_bundle, DeltaBundle) else self._deserialize_delta(delta_bundle)
        reference_memory = self.codec.decode_memory_bundle(reference_bundle)
        merged_memory = self._apply_delta(reference_memory, parsed)
        return self.codec.encode_memory_bundle(merged_memory)

    def full_bundle_bytes(self, bundle: Dict[str, List[List[List[int]]]]) -> bytes:
        self.binary_codec.sync_from_codec()
        return self.bitpacked_codec.encode_jsonable_bundle_bitpacked(bundle).payload

    def delta_bundle_size(self, delta_bundle: DeltaBundle | bytes) -> int:
        return delta_bundle.total_bytes if isinstance(delta_bundle, DeltaBundle) else len(delta_bundle)

    def _coerce_reference(
        self,
        item: StoredMemory | Dict[str, object],
    ) -> Tuple[int, Dict[str, List[List[List[int]]]]]:
        if isinstance(item, StoredMemory):
            if item.resolved_bundle is None:
                raise BrainDeltaCodecError(f"Stored memory {item.memory_id} does not have a resolved bundle.")
            return item.memory_id, item.resolved_bundle

        if not isinstance(item, dict):
            raise BrainDeltaCodecError("Existing bundles must be StoredMemory records or mapping objects.")
        if "id" not in item or "bundle" not in item:
            raise BrainDeltaCodecError("Reference bundle mappings must include 'id' and 'bundle'.")
        reference_bundle = item["bundle"]
        if not isinstance(reference_bundle, dict):
            raise BrainDeltaCodecError("Reference bundle must be a grid bundle mapping.")
        return int(item["id"]), reference_bundle

    def _score_reference(
        self,
        current_memory: BrainMemory,
        reference_memory: BrainMemory,
        reference_id: int,
    ) -> ReferenceMatch:
        current_dict = current_memory.to_dict()
        reference_dict = reference_memory.to_dict()

        matched_scalar_fields: List[str] = []
        changed_scalar_fields: List[str] = []
        component_scores: Dict[str, float] = {}
        list_field_scores: Dict[str, float] = {}

        for field in SINGLE_FIELDS:
            current_value = str(current_dict.get(field, "") or "")
            reference_value = str(reference_dict.get(field, "") or "")
            score = self._score_scalar_field(current_value, reference_value)
            if score is None:
                continue
            component_scores[field] = round(score, 4)
            if score >= 1.0:
                matched_scalar_fields.append(field)
            else:
                changed_scalar_fields.append(field)

        for field in LIST_FIELDS:
            current_values = list(current_dict.get(field, []) or [])
            reference_values = list(reference_dict.get(field, []) or [])
            score = self._score_list_field(current_values, reference_values)
            if score is None:
                continue
            list_field_scores[field] = round(score, 4)
            component_scores[field] = round(score, 4)

        current_relation_signatures = self._relation_signatures(current_memory.relations)
        reference_relation_signatures = self._relation_signatures(reference_memory.relations)
        exact_relation_similarity = self._relation_similarity(
            current_relation_signatures,
            reference_relation_signatures,
        )

        current_relation_shapes = self._relation_shapes(current_memory.relations)
        reference_relation_shapes = self._relation_shapes(reference_memory.relations)
        structural_relation_similarity = self._relation_similarity(
            current_relation_shapes,
            reference_relation_shapes,
        )

        if exact_relation_similarity is not None:
            component_scores["exact_relation_similarity"] = round(exact_relation_similarity, 4)
        if structural_relation_similarity is not None:
            component_scores["structural_relation_similarity"] = round(structural_relation_similarity, 4)

        similarity = round(
            self._weighted_average(
                component_scores=component_scores,
                component_weights=self._component_weights(),
            ),
            4,
        )
        core_structure_similarity = round(
            self._weighted_average(
                component_scores=component_scores,
                component_weights=CORE_STRUCTURE_WEIGHTS,
            ),
            4,
        )
        reuse_basis = self._determine_reuse_basis(
            similarity=similarity,
            core_structure_similarity=core_structure_similarity,
            component_scores=component_scores,
        )

        reasons = [
            f"weighted_similarity={similarity}",
            f"core_structure_similarity={core_structure_similarity}",
            f"action_score={component_scores.get('action', 'n/a')}",
            f"object_score={component_scores.get('object', 'n/a')}",
            f"topic_score={component_scores.get('topic', 'n/a')}",
            f"modifiers_score={component_scores.get('modifiers', 'n/a')}",
            f"exact_relation_similarity={round(exact_relation_similarity, 4) if exact_relation_similarity is not None else 'n/a'}",
            f"structural_relation_similarity={round(structural_relation_similarity, 4) if structural_relation_similarity is not None else 'n/a'}",
            f"reuse_basis={reuse_basis}",
        ]

        return ReferenceMatch(
            reference_id=reference_id,
            similarity=similarity,
            core_structure_similarity=core_structure_similarity,
            matched_scalar_fields=matched_scalar_fields,
            changed_scalar_fields=changed_scalar_fields,
            component_scores=component_scores,
            list_field_scores=list_field_scores,
            exact_relation_similarity=round(exact_relation_similarity, 4) if exact_relation_similarity is not None else 0.0,
            structural_relation_similarity=round(structural_relation_similarity, 4) if structural_relation_similarity is not None else 0.0,
            reuse_basis=reuse_basis,
            reasons=reasons,
        )

    def _diff_fields(self, current_memory: BrainMemory, reference_memory: BrainMemory) -> Dict[str, object]:
        current_dict = current_memory.to_dict()
        reference_dict = reference_memory.to_dict()
        changed: Dict[str, object] = {}

        for field in SINGLE_FIELDS:
            current_value = str(current_dict.get(field, "") or "")
            reference_value = str(reference_dict.get(field, "") or "")
            if current_value != reference_value:
                changed[field] = current_value

        for field in LIST_FIELDS:
            current_values = list(current_dict.get(field, []) or [])
            reference_values = list(reference_dict.get(field, []) or [])
            if current_values != reference_values:
                changed[field] = current_values

        return changed

    def _diff_relations(
        self,
        current_relations: Sequence[Dict[str, str]],
        reference_relations: Sequence[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        current_map = {self._relation_signature(relation): dict(relation) for relation in current_relations}
        reference_map = {self._relation_signature(relation): dict(relation) for relation in reference_relations}

        added_keys = sorted(set(current_map.keys()) - set(reference_map.keys()))
        removed_keys = sorted(set(reference_map.keys()) - set(current_map.keys()))

        return [current_map[key] for key in added_keys], [reference_map[key] for key in removed_keys]

    def _apply_delta(self, reference_memory: BrainMemory, delta_bundle: DeltaBundle) -> BrainMemory:
        merged = BrainMemory.from_dict(reference_memory.to_dict())

        for field in SINGLE_FIELDS:
            if field in delta_bundle.changed_fields:
                setattr(merged, field, str(delta_bundle.changed_fields[field]))

        for field in LIST_FIELDS:
            if field in delta_bundle.changed_fields:
                setattr(merged, field, list(delta_bundle.changed_fields[field]))

        if delta_bundle.relation_mode == "replace":
            merged.relations = [dict(relation) for relation in delta_bundle.replacement_relations]
            return merged

        relation_map = {
            self._relation_signature(relation): dict(relation)
            for relation in merged.relations
        }

        for relation in delta_bundle.removed_relations:
            relation_map.pop(self._relation_signature(relation), None)

        for relation in delta_bundle.added_relations:
            relation_map[self._relation_signature(relation)] = dict(relation)

        merged.relations = [relation_map[key] for key in sorted(relation_map.keys())]
        return merged

    def _serialize_delta(
        self,
        reference_id: int,
        changed_fields: Dict[str, object],
        added_relations: Sequence[Dict[str, str]],
        removed_relations: Sequence[Dict[str, str]],
        relation_mode: str,
        replacement_relations: Sequence[Dict[str, str]],
        new_primitives: Sequence[str],
        primitive_id_bytes: int,
    ) -> bytes:
        if reference_id < 0 or reference_id > 0xFFFF:
            raise BrainDeltaCodecError("Reference IDs must fit in uint16.")

        single_mask = 0
        list_mask = 0
        for index, field in enumerate(SINGLE_FIELDS):
            if field in changed_fields:
                single_mask |= 1 << index
        for index, field in enumerate(LIST_FIELDS):
            if field in changed_fields:
                list_mask |= 1 << index

        if len(added_relations) > 0xFF or len(removed_relations) > 0xFF or len(replacement_relations) > 0xFF:
            raise BrainDeltaCodecError("Delta sections are limited to 255 entries.")

        relation_mode_code = 1 if relation_mode == "patch" else 2 if relation_mode == "replace" else 0
        if relation_mode_code == 0 and (added_relations or removed_relations or replacement_relations):
            raise BrainDeltaCodecError("Invalid relation mode for non-empty relation delta.")

        parts = [
            DELTA_MAGIC,
            struct.pack(
                "<BHBBBBBB",
                DELTA_VERSION,
                reference_id,
                primitive_id_bytes,
                single_mask,
                list_mask,
                relation_mode_code,
                len(added_relations),
                len(removed_relations),
            ),
        ]

        for field in SINGLE_FIELDS:
            if field not in changed_fields:
                continue
            primitive_id = self._field_value_to_id(field, changed_fields[field])
            parts.append(self._pack_primitive_id(primitive_id, primitive_id_bytes))

        for field in LIST_FIELDS:
            if field not in changed_fields:
                continue
            values = list(changed_fields[field])
            if len(values) > 0xFF:
                raise BrainDeltaCodecError(f"Changed list field {field} exceeds 255 entries.")
            parts.append(struct.pack("<B", len(values)))
            for value in values:
                primitive_id = self._field_value_to_id(field, value)
                parts.append(self._pack_primitive_id(primitive_id, primitive_id_bytes))

        relation_records = list(added_relations) + list(removed_relations) if relation_mode == "patch" else list(replacement_relations)
        for relation in relation_records:
            for primitive_id in self._relation_to_primitive_ids(relation):
                parts.append(self._pack_primitive_id(primitive_id, primitive_id_bytes))

        return b"".join(parts)

    def _deserialize_delta(self, payload: bytes) -> DeltaBundle:
        if len(payload) < 10:
            raise BrainDeltaCodecError("Delta payload is too short.")
        if payload[:2] != DELTA_MAGIC:
            raise BrainDeltaCodecError("Delta payload magic header is invalid.")

        (
            version,
            reference_id,
            primitive_id_bytes,
            single_mask,
            list_mask,
            relation_mode_code,
            added_count,
            removed_count,
        ) = struct.unpack("<BHBBBBBB", payload[2:11])

        if version != DELTA_VERSION:
            raise BrainDeltaCodecError(f"Unsupported delta payload version: {version}")
        if primitive_id_bytes not in (1, 2, 4):
            raise BrainDeltaCodecError("Primitive ID width must be 1, 2, or 4 bytes.")

        relation_mode = "patch" if relation_mode_code == 1 else "replace" if relation_mode_code == 2 else "none"
        if relation_mode == "none" and (added_count or removed_count):
            raise BrainDeltaCodecError("Delta payload has relation counts without a valid relation mode.")

        offset = 11
        changed_fields: Dict[str, object] = {}

        for index, field in enumerate(SINGLE_FIELDS):
            if not (single_mask & (1 << index)):
                continue
            primitive_id, offset = self._unpack_primitive_id(payload, offset, primitive_id_bytes)
            changed_fields[field] = self._primitive_id_to_field_value(field, primitive_id)

        for index, field in enumerate(LIST_FIELDS):
            if not (list_mask & (1 << index)):
                continue
            if offset >= len(payload):
                raise BrainDeltaCodecError("Delta payload ended while reading list length.")
            count = payload[offset]
            offset += 1
            values: List[str] = []
            for _ in range(count):
                primitive_id, offset = self._unpack_primitive_id(payload, offset, primitive_id_bytes)
                values.append(str(self._primitive_id_to_field_value(field, primitive_id)))
            changed_fields[field] = values

        added_relations: List[Dict[str, str]] = []
        removed_relations: List[Dict[str, str]] = []
        replacement_relations: List[Dict[str, str]] = []
        if relation_mode == "patch":
            for _ in range(added_count):
                relation, offset = self._unpack_relation(payload, offset, primitive_id_bytes)
                added_relations.append(relation)
            for _ in range(removed_count):
                relation, offset = self._unpack_relation(payload, offset, primitive_id_bytes)
                removed_relations.append(relation)
        elif relation_mode == "replace":
            for _ in range(added_count):
                relation, offset = self._unpack_relation(payload, offset, primitive_id_bytes)
                replacement_relations.append(relation)

        if offset != len(payload):
            raise BrainDeltaCodecError("Delta payload contains trailing bytes.")

        return DeltaBundle(
            reference_id=reference_id,
            changed_fields=changed_fields,
            added_relations=added_relations,
            removed_relations=removed_relations,
            new_primitives=[],
            payload=payload,
            relation_mode=relation_mode,
            replacement_relations=replacement_relations,
        )

    def _field_value_to_id(self, field: str, value: object) -> int:
        text_value = str(value or "")
        if not text_value:
            return EMPTY_ID
        primitive_type = FIELD_TO_PRIMITIVE_TYPE[field]
        primitive = self.codec.make_primitive(primitive_type, text_value)
        return self.binary_codec.primitive_id(primitive)

    def _primitive_id_to_field_value(self, field: str, primitive_id: int) -> str:
        if primitive_id == EMPTY_ID:
            return ""
        primitive = self.binary_codec.id_to_primitive.get(primitive_id)
        if primitive in (None, "UNKNOWN"):
            raise BrainDeltaCodecError(f"Unknown primitive ID {primitive_id} while decoding field {field}.")
        _, value = str(primitive).split(":", 1)
        return value

    def _relation_to_primitive_ids(self, relation: Dict[str, str]) -> Tuple[int, int, int]:
        relation_type = str(relation.get("type", "") or "")
        source = str(relation.get("source", "") or "")
        target = str(relation.get("target", "") or "")

        type_id = self.binary_codec.primitive_id(self.codec.make_primitive("relation_type", relation_type))
        source_primitive = self.codec._guess_relation_endpoint_primitive(
            value=source,
            relation_type=relation_type,
            endpoint_role="source",
        )
        target_primitive = self.codec._guess_relation_endpoint_primitive(
            value=target,
            relation_type=relation_type,
            endpoint_role="target",
        )
        source_id = self.binary_codec.primitive_id(source_primitive)
        target_id = self.binary_codec.primitive_id(target_primitive)
        return type_id, source_id, target_id

    def _unpack_relation(
        self,
        payload: bytes,
        offset: int,
        primitive_id_bytes: int,
    ) -> Tuple[Dict[str, str], int]:
        type_id, offset = self._unpack_primitive_id(payload, offset, primitive_id_bytes)
        source_id, offset = self._unpack_primitive_id(payload, offset, primitive_id_bytes)
        target_id, offset = self._unpack_primitive_id(payload, offset, primitive_id_bytes)

        relation_type = self._decode_relation_value(type_id, expected_prefix="relation_type")
        source = self._decode_relation_value(source_id)
        target = self._decode_relation_value(target_id)

        return {
            "type": relation_type,
            "source": source,
            "target": target,
        }, offset

    def _decode_relation_value(self, primitive_id: int, expected_prefix: str | None = None) -> str:
        if primitive_id == EMPTY_ID:
            return ""
        primitive = self.binary_codec.id_to_primitive.get(primitive_id)
        if primitive in (None, "UNKNOWN"):
            raise BrainDeltaCodecError(f"Unknown relation primitive ID: {primitive_id}")
        prefix, value = str(primitive).split(":", 1)
        if expected_prefix is not None and prefix != expected_prefix:
            raise BrainDeltaCodecError(
                f"Expected primitive prefix {expected_prefix}, got {prefix} for relation primitive {primitive_id}."
            )
        return value

    def _collect_delta_primitive_ids(
        self,
        changed_fields: Dict[str, object],
        added_relations: Sequence[Dict[str, str]],
        removed_relations: Sequence[Dict[str, str]],
        replacement_relations: Sequence[Dict[str, str]],
        new_primitives: Sequence[str],
    ) -> List[int]:
        primitive_ids: List[int] = []

        for field, value in changed_fields.items():
            if field in SINGLE_FIELDS:
                primitive_ids.append(self._field_value_to_id(field, value))
            elif field in LIST_FIELDS:
                for item in list(value):
                    primitive_ids.append(self._field_value_to_id(field, item))

        for relation in list(added_relations) + list(removed_relations) + list(replacement_relations):
            primitive_ids.extend(self._relation_to_primitive_ids(relation))

        for primitive in new_primitives:
            primitive_ids.append(self.binary_codec.primitive_id(primitive))

        return [primitive_id for primitive_id in primitive_ids if primitive_id != EMPTY_ID]

    def _choose_relation_mode(
        self,
        current_relations: Sequence[Dict[str, str]],
        added_relations: Sequence[Dict[str, str]],
        removed_relations: Sequence[Dict[str, str]],
    ) -> Tuple[str, List[Dict[str, str]]]:
        if not added_relations and not removed_relations:
            return "none", []

        patch_count = len(added_relations) + len(removed_relations)
        replace_count = len(current_relations)

        if replace_count <= patch_count:
            replacement_relations = [dict(relation) for relation in current_relations]
            replacement_relations.sort(key=self._relation_signature)
            return "replace", replacement_relations

        return "patch", []

    @staticmethod
    def _score_scalar_field(current_value: str, reference_value: str) -> Optional[float]:
        if not current_value and not reference_value:
            return None
        return 1.0 if current_value == reference_value else 0.0

    def _score_list_field(self, current_values: Sequence[str], reference_values: Sequence[str]) -> Optional[float]:
        if not current_values and not reference_values:
            return None
        return self._jaccard_strings(current_values, reference_values)

    def _relation_similarity(
        self,
        left: set[Tuple],
        right: set[Tuple],
    ) -> Optional[float]:
        if not left and not right:
            return None
        return self._jaccard_tuples(left, right)

    @staticmethod
    def _component_weights() -> Dict[str, float]:
        weights = {}
        weights.update(SINGLE_FIELD_WEIGHTS)
        weights.update(LIST_FIELD_WEIGHTS)
        weights.update(RELATION_COMPONENT_WEIGHTS)
        return weights

    @staticmethod
    def _weighted_average(
        component_scores: Dict[str, float],
        component_weights: Dict[str, float],
    ) -> float:
        weighted_total = 0.0
        weight_total = 0.0
        for component, weight in component_weights.items():
            if component not in component_scores:
                continue
            weighted_total += component_scores[component] * weight
            weight_total += weight
        if weight_total <= 0:
            return 0.0
        return weighted_total / weight_total

    def _determine_reuse_basis(
        self,
        similarity: float,
        core_structure_similarity: float,
        component_scores: Dict[str, float],
    ) -> str:
        if similarity >= self.similarity_threshold:
            return "similarity_threshold"
        if self._has_core_structure_match(core_structure_similarity, component_scores):
            return "core_structure_gate"
        return "none"

    def _has_core_structure_match(
        self,
        core_structure_similarity: float,
        component_scores: Dict[str, float],
    ) -> bool:
        action_score = component_scores.get("action")
        object_score = component_scores.get("object")
        structural_score = component_scores.get("structural_relation_similarity")

        if action_score != 1.0:
            return False
        if object_score is not None and object_score != 1.0:
            return False
        if structural_score is not None and structural_score < MIN_STRUCTURAL_RELATION_SCORE:
            return False
        return core_structure_similarity >= self.core_structure_threshold

    def _is_reusable_candidate(self, candidate: ReferenceMatch) -> bool:
        return candidate.reuse_basis != "none"

    def _diff_bundle_primitives(
        self,
        current_bundle: Dict[str, List[List[List[int]]]],
        reference_bundle: Dict[str, List[List[List[int]]]],
    ) -> List[str]:
        current_ids = self._bundle_primitive_ids(current_bundle)
        reference_ids = self._bundle_primitive_ids(reference_bundle)
        new_ids = sorted(current_ids - reference_ids)

        primitives: List[str] = []
        for primitive_id in new_ids:
            primitive = self.binary_codec.id_to_primitive.get(primitive_id)
            if primitive in (None, "UNKNOWN"):
                continue
            primitives.append(str(primitive))
        return primitives

    def _bundle_primitive_ids(self, bundle: Dict[str, List[List[List[int]]]]) -> set[int]:
        fixed_bundle = self.binary_codec.encode_jsonable_bundle(bundle)
        return {
            primitive_id
            for primitive_id in list(fixed_bundle.core_ids) + list(fixed_bundle.relation_ids)
            if primitive_id != EMPTY_ID
        }

    @staticmethod
    def _pack_primitive_id(primitive_id: int, width: int) -> bytes:
        if width == 1:
            return struct.pack("<B", primitive_id)
        if width == 2:
            return struct.pack("<H", primitive_id)
        if width == 4:
            return struct.pack("<I", primitive_id)
        raise BrainDeltaCodecError(f"Unsupported primitive ID width: {width}")

    @staticmethod
    def _unpack_primitive_id(payload: bytes, offset: int, width: int) -> Tuple[int, int]:
        end = offset + width
        if end > len(payload):
            raise BrainDeltaCodecError("Delta payload ended while reading primitive ID.")
        if width == 1:
            return payload[offset], end
        if width == 2:
            return struct.unpack("<H", payload[offset:end])[0], end
        if width == 4:
            return struct.unpack("<I", payload[offset:end])[0], end
        raise BrainDeltaCodecError(f"Unsupported primitive ID width: {width}")

    @staticmethod
    def _primitive_id_width(primitive_ids: Sequence[int]) -> int:
        max_id = max([0] + [primitive_id for primitive_id in primitive_ids if primitive_id >= 0])
        if max_id <= 0xFF:
            return 1
        if max_id <= 0xFFFF:
            return 2
        return 4

    @staticmethod
    def _relation_signature(relation: Dict[str, str]) -> Tuple[str, str, str]:
        return (
            str(relation.get("type", "") or ""),
            str(relation.get("source", "") or ""),
            str(relation.get("target", "") or ""),
        )

    def _relation_signatures(self, relations: Sequence[Dict[str, str]]) -> set[Tuple[str, str, str]]:
        return {self._relation_signature(relation) for relation in relations}

    def _relation_shapes(self, relations: Sequence[Dict[str, str]]) -> set[Tuple[str, str]]:
        return {
            (
                str(relation.get("type", "") or ""),
                str(relation.get("source", "") or ""),
            )
            for relation in relations
        }

    @staticmethod
    def _jaccard_strings(left: Sequence[str], right: Sequence[str]) -> float:
        left_set = {str(item) for item in left}
        right_set = {str(item) for item in right}
        if not left_set and not right_set:
            return 1.0
        union = left_set | right_set
        if not union:
            return 1.0
        return len(left_set & right_set) / len(union)

    @staticmethod
    def _jaccard_tuples(left: set[Tuple], right: set[Tuple]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 1.0
        return len(left & right) / len(union)


class BrainDeltaStore:
    """
    Simple in-memory storage for full and delta-encoded BrAIn memories.
    """

    def __init__(self, delta_codec: BrainDeltaCodec):
        self.delta_codec = delta_codec
        self.entries: Dict[int, StoredMemory] = {}
        self.order: List[int] = []

    def add_memory(
        self,
        memory_id: int,
        memory_bundle: Dict[str, List[List[List[int]]]],
    ) -> StoredMemory:
        if memory_id in self.entries:
            raise BrainDeltaCodecError(f"Memory ID already exists in delta store: {memory_id}")

        full_payload = self.delta_codec.full_bundle_bytes(memory_bundle)
        best_match = self.delta_codec.find_best_reference(
            memory_bundle=memory_bundle,
            existing_bundles=[self.entries[item_id] for item_id in self.order],
        )

        if best_match is not None:
            reference_bundle = self.resolve_bundle(best_match.reference_id)
            delta_bundle = self.delta_codec.encode_memory_bundle_delta(
                current_bundle=memory_bundle,
                reference_bundle=reference_bundle,
                reference_id=best_match.reference_id,
                similarity=best_match.similarity,
            )
            if delta_bundle.total_bytes <= len(full_payload):
                entry = StoredMemory(
                    memory_id=memory_id,
                    encoding_type="delta",
                    payload=delta_bundle.payload,
                    reference_id=best_match.reference_id,
                    resolved_bundle=memory_bundle,
                    similarity=best_match.similarity,
                    match=best_match,
                    delta_bundle=delta_bundle,
                    full_size_bytes=len(full_payload),
                    stored_size_bytes=delta_bundle.total_bytes,
                    new_primitives=list(delta_bundle.new_primitives),
                )
            else:
                entry = StoredMemory(
                    memory_id=memory_id,
                    encoding_type="full",
                    payload=full_payload,
                    reference_id=None,
                    resolved_bundle=memory_bundle,
                    similarity=0.0,
                    match=best_match,
                    delta_bundle=None,
                    full_size_bytes=len(full_payload),
                    stored_size_bytes=len(full_payload),
                    new_primitives=[],
                )
        else:
            entry = StoredMemory(
                memory_id=memory_id,
                encoding_type="full",
                payload=full_payload,
                reference_id=None,
                resolved_bundle=memory_bundle,
                similarity=0.0,
                match=None,
                delta_bundle=None,
                full_size_bytes=len(full_payload),
                stored_size_bytes=len(full_payload),
                new_primitives=[],
            )

        self.entries[memory_id] = entry
        self.order.append(memory_id)
        return entry

    def resolve_bundle(self, memory_id: int) -> Dict[str, List[List[List[int]]]]:
        entry = self.entries.get(memory_id)
        if entry is None:
            raise BrainDeltaCodecError(f"Unknown memory ID: {memory_id}")

        if entry.resolved_bundle is not None:
            return entry.resolved_bundle

        if entry.encoding_type == "full":
            bundle = self.delta_codec.bitpacked_codec.decode_memory_bundle_bitpacked(entry.payload)
            entry.resolved_bundle = bundle
            return bundle

        if entry.reference_id is None:
            raise BrainDeltaCodecError(f"Delta memory {memory_id} is missing a reference ID.")

        reference_bundle = self.resolve_bundle(entry.reference_id)
        bundle = self.delta_codec.decode_memory_bundle_delta(entry.payload, reference_bundle)
        entry.resolved_bundle = bundle
        return bundle

    def resolve_memory(self, memory_id: int) -> BrainMemory:
        return self.delta_codec.codec.decode_memory_bundle(self.resolve_bundle(memory_id))

    def previous_entries(self) -> List[StoredMemory]:
        return [self.entries[memory_id] for memory_id in self.order]


def find_best_reference(
    memory_bundle: Dict[str, List[List[List[int]]]],
    existing_bundles: Sequence[StoredMemory | Dict[str, object]],
    codec: Optional[BrainCodec] = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    core_structure_threshold: float = DEFAULT_CORE_STRUCTURE_THRESHOLD,
) -> Optional[ReferenceMatch]:
    active_codec = codec or BrainCodec(auto_register=False)
    delta_codec = BrainDeltaCodec(
        codec=active_codec,
        similarity_threshold=similarity_threshold,
        core_structure_threshold=core_structure_threshold,
    )
    return delta_codec.find_best_reference(memory_bundle, existing_bundles)


def encode_memory_bundle_delta(
    current_bundle: Dict[str, List[List[List[int]]]],
    reference_bundle: Dict[str, List[List[List[int]]]],
    codec: Optional[BrainCodec] = None,
    reference_id: int = 0,
    similarity: float = 0.0,
) -> DeltaBundle:
    active_codec = codec or BrainCodec(auto_register=False)
    delta_codec = BrainDeltaCodec(codec=active_codec)
    return delta_codec.encode_memory_bundle_delta(
        current_bundle=current_bundle,
        reference_bundle=reference_bundle,
        reference_id=reference_id,
        similarity=similarity,
    )


def decode_memory_bundle_delta(
    delta_bundle: DeltaBundle | bytes,
    reference_bundle: Dict[str, List[List[List[int]]]],
    codec: Optional[BrainCodec] = None,
) -> Dict[str, List[List[List[int]]]]:
    active_codec = codec or BrainCodec(auto_register=False)
    delta_codec = BrainDeltaCodec(codec=active_codec)
    return delta_codec.decode_memory_bundle_delta(delta_bundle, reference_bundle)
