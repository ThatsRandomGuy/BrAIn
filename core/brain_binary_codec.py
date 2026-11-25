from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from brain_codec import (
    CORE_GRID_COLS,
    CORE_GRID_ROWS,
    EMPTY,
    RELATION_GRID_COLS,
    RELATION_GRID_ROWS,
    UNKNOWN,
    BrainCodec,
    BrainMemory,
    Grid,
)


EMPTY_ID = 0
UNKNOWN_ID = 1
UINT32_BYTES = 4


@dataclass(frozen=True)
class BinaryBundle:
    core_ids: List[int]
    relation_ids: List[int]
    core_bytes: bytes
    relation_bytes: bytes

    @property
    def total_bytes(self) -> int:
        return len(self.core_bytes) + len(self.relation_bytes)


class BrainBinaryCodecError(Exception):
    pass


class BrainBinaryCodec:
    """
    Encodes BrAIn's fixed grids into fixed-size integer arrays and packed bytes.

    Reserved IDs:
    - 0 => empty cell
    - 1 => unknown primitive

    All other primitives are assigned stable integer IDs within the lifetime of
    the codec instance. New primitives discovered in the backing BrainCodec are
    appended without renumbering existing IDs.
    """

    def __init__(self, codec: BrainCodec):
        self.codec = codec
        self.primitive_to_id: Dict[str, int] = {}
        self.id_to_primitive: Dict[int, Optional[str]] = {
            EMPTY_ID: None,
            UNKNOWN_ID: "UNKNOWN",
        }
        self._next_id = 2
        self.sync_from_codec()

    @property
    def bytes_per_cell(self) -> int:
        return UINT32_BYTES

    @property
    def core_cell_count(self) -> int:
        return CORE_GRID_ROWS * CORE_GRID_COLS

    @property
    def relation_cell_count(self) -> int:
        return RELATION_GRID_ROWS * RELATION_GRID_COLS

    def sync_from_codec(self) -> None:
        unseen = sorted(
            primitive
            for primitive in self.codec.primitive_to_color.keys()
            if primitive not in self.primitive_to_id
        )
        for primitive in unseen:
            self.primitive_to_id[primitive] = self._next_id
            self.id_to_primitive[self._next_id] = primitive
            self._next_id += 1

    def encode_memory(self, memory: BrainMemory) -> List[int]:
        grid = self.codec.encode_memory(memory)
        self.sync_from_codec()
        return self.encode_core_grid(grid)

    def decode_memory(self, core_ids: Sequence[int]) -> BrainMemory:
        grid = self.decode_core_grid(core_ids)
        return self.codec.decode_grid(grid)

    def encode_relations(self, memory: BrainMemory) -> List[int]:
        grid = self.codec.encode_relations(memory)
        self.sync_from_codec()
        return self.encode_relation_grid(grid)

    def decode_relations(self, relation_ids: Sequence[int]) -> List[Dict[str, str]]:
        grid = self.decode_relation_grid(relation_ids)
        return self.codec.decode_relations(grid)

    def encode_memory_bundle(self, memory: BrainMemory) -> BinaryBundle:
        bundle = self.codec.encode_memory_bundle(memory)
        return self.encode_jsonable_bundle(bundle)

    def decode_memory_bundle(
        self,
        core_ids: Sequence[int],
        relation_ids: Sequence[int],
    ) -> BrainMemory:
        memory = self.decode_memory(core_ids)
        memory.relations = self.decode_relations(relation_ids)
        return memory

    def encode_jsonable_bundle(self, bundle: Dict[str, List[List[List[int]]]]) -> BinaryBundle:
        self.sync_from_codec()
        core_grid = self.codec.jsonable_to_grid(bundle["core_grid"])
        relation_grid = self.codec.jsonable_to_grid(bundle["relation_grid"])

        core_ids = self.encode_core_grid(core_grid)
        relation_ids = self.encode_relation_grid(relation_grid)
        return BinaryBundle(
            core_ids=core_ids,
            relation_ids=relation_ids,
            core_bytes=self.pack_integer_array(core_ids),
            relation_bytes=self.pack_integer_array(relation_ids),
        )

    def decode_jsonable_bundle(
        self,
        core_ids: Sequence[int],
        relation_ids: Sequence[int],
    ) -> Dict[str, List[List[List[int]]]]:
        core_grid = self.decode_core_grid(core_ids)
        relation_grid = self.decode_relation_grid(relation_ids)
        return {
            "core_grid": self.codec.grid_to_jsonable(core_grid),
            "relation_grid": self.codec.grid_to_jsonable(relation_grid),
        }

    def encode_core_grid(self, grid: Grid) -> List[int]:
        return self._encode_grid(grid, CORE_GRID_ROWS, CORE_GRID_COLS)

    def decode_core_grid(self, values: Sequence[int]) -> Grid:
        return self._decode_grid(values, CORE_GRID_ROWS, CORE_GRID_COLS)

    def encode_relation_grid(self, grid: Grid) -> List[int]:
        return self._encode_grid(grid, RELATION_GRID_ROWS, RELATION_GRID_COLS)

    def decode_relation_grid(self, values: Sequence[int]) -> Grid:
        return self._decode_grid(values, RELATION_GRID_ROWS, RELATION_GRID_COLS)

    def pack_core_ids(self, values: Sequence[int]) -> bytes:
        if len(values) != self.core_cell_count:
            raise BrainBinaryCodecError(
                f"Core grid must contain exactly {self.core_cell_count} integer IDs."
            )
        return self.pack_integer_array(values)

    def unpack_core_ids(self, payload: bytes) -> List[int]:
        return self.unpack_integer_array(payload, self.core_cell_count)

    def pack_relation_ids(self, values: Sequence[int]) -> bytes:
        if len(values) != self.relation_cell_count:
            raise BrainBinaryCodecError(
                f"Relation grid must contain exactly {self.relation_cell_count} integer IDs."
            )
        return self.pack_integer_array(values)

    def unpack_relation_ids(self, payload: bytes) -> List[int]:
        return self.unpack_integer_array(payload, self.relation_cell_count)

    @staticmethod
    def pack_integer_array(values: Sequence[int]) -> bytes:
        checked = [BrainBinaryCodec._checked_uint32(value) for value in values]
        return struct.pack(f"<{len(checked)}I", *checked)

    @staticmethod
    def unpack_integer_array(payload: bytes, expected_length: int) -> List[int]:
        expected_bytes = expected_length * UINT32_BYTES
        if len(payload) != expected_bytes:
            raise BrainBinaryCodecError(
                f"Expected {expected_bytes} bytes for {expected_length} integers, got {len(payload)}."
            )

        if expected_length == 0:
            return []

        return list(struct.unpack(f"<{expected_length}I", payload))

    @staticmethod
    def measure_integer_array_bytes(values: Sequence[int]) -> int:
        return len(values) * UINT32_BYTES

    def primitive_id(self, primitive: Optional[str]) -> int:
        if primitive is None:
            return EMPTY_ID
        if primitive == "UNKNOWN":
            return UNKNOWN_ID

        existing = self.primitive_to_id.get(primitive)
        if existing is not None:
            return existing

        self.codec.ensure_primitive(primitive)
        self.sync_from_codec()

        resolved = self.primitive_to_id.get(primitive)
        if resolved is None:
            raise BrainBinaryCodecError(f"Primitive could not be assigned an integer ID: {primitive}")
        return resolved

    def id_to_color(self, primitive_id: int) -> tuple[int, int, int]:
        if primitive_id == EMPTY_ID:
            return EMPTY
        if primitive_id == UNKNOWN_ID:
            return UNKNOWN

        primitive = self.id_to_primitive.get(primitive_id)
        if primitive is None or primitive == "UNKNOWN":
            raise BrainBinaryCodecError(f"Unknown primitive ID: {primitive_id}")

        return self.codec.primitive_to_rgb(primitive)

    def _encode_grid(self, grid: Grid, expected_rows: int, expected_cols: int) -> List[int]:
        self._validate_grid(grid, expected_rows, expected_cols)

        encoded: List[int] = []
        for row in grid:
            for color in row:
                primitive = self.codec.rgb_to_primitive(color)
                encoded.append(self.primitive_id(primitive))
        return encoded

    def _decode_grid(self, values: Sequence[int], expected_rows: int, expected_cols: int) -> Grid:
        expected_length = expected_rows * expected_cols
        if len(values) != expected_length:
            raise BrainBinaryCodecError(
                f"Expected {expected_length} integer IDs, got {len(values)}."
            )

        grid: Grid = []
        for row_index in range(expected_rows):
            row: List[tuple[int, int, int]] = []
            start = row_index * expected_cols
            end = start + expected_cols
            for primitive_id in values[start:end]:
                row.append(self.id_to_color(self._checked_uint32(primitive_id)))
            grid.append(row)
        return grid

    @staticmethod
    def _checked_uint32(value: int) -> int:
        if not isinstance(value, int):
            raise BrainBinaryCodecError("Integer array values must be integers.")
        if value < 0 or value > 0xFFFFFFFF:
            raise BrainBinaryCodecError("Integer array values must fit in uint32.")
        return value

    @staticmethod
    def _validate_grid(grid: Grid, expected_rows: int, expected_cols: int) -> None:
        if len(grid) != expected_rows:
            raise BrainBinaryCodecError(f"Grid must have {expected_rows} rows.")
        for row in grid:
            if len(row) != expected_cols:
                raise BrainBinaryCodecError(f"Each grid row must have {expected_cols} columns.")
            for color in row:
                if not isinstance(color, tuple) or len(color) != 3:
                    raise BrainBinaryCodecError("Each cell must be an (R, G, B) tuple.")

