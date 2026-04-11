from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Sequence

from brain_binary_codec import BrainBinaryCodec, EMPTY_ID
from brain_codec import CORE_GRID_COLS, CORE_GRID_ROWS, RELATION_GRID_COLS, RELATION_GRID_ROWS, BrainMemory


MAGIC = b"BS"
VERSION = 1
GRID_TYPE_CORE = 0
GRID_TYPE_RELATION = 1

FORMAT_MAP = {
    1: "B",
    2: "H",
    4: "I",
}


@dataclass(frozen=True)
class SparseCell:
    grid_type: str
    cell_index: int
    primitive_id: int


@dataclass(frozen=True)
class SparseBundle:
    cells: List[SparseCell]
    payload: bytes
    index_bytes: int
    primitive_id_bytes: int
    core_populated_cells: int
    relation_populated_cells: int

    @property
    def total_bytes(self) -> int:
        return len(self.payload)

    @property
    def populated_cells(self) -> int:
        return self.core_populated_cells + self.relation_populated_cells


class BrainSparseCodecError(Exception):
    pass


class BrainSparseCodec:
    """
    Sparse binary encoding for BrAIn's fixed core and relation grids.

    Format:
    - 2 bytes magic: b"BS"
    - 1 byte version
    - 1 byte index width
    - 1 byte primitive-id width
    - 1 byte populated-cell count
    - repeated populated-cell entries:
      - 1 byte grid type (0 core, 1 relation)
      - index (uint8 or uint16)
      - primitive id (uint8, uint16, or uint32)

    Empty cells are skipped entirely.
    """

    def __init__(self, binary_codec: BrainBinaryCodec):
        self.binary_codec = binary_codec

    @property
    def core_cell_count(self) -> int:
        return self.binary_codec.core_cell_count

    @property
    def relation_cell_count(self) -> int:
        return self.binary_codec.relation_cell_count

    def encode_memory_bundle_sparse(self, memory: BrainMemory) -> SparseBundle:
        bundle = self.binary_codec.codec.encode_memory_bundle(memory)
        return self.encode_jsonable_bundle_sparse(bundle)

    def encode_jsonable_bundle_sparse(self, bundle: Dict[str, List[List[List[int]]]]) -> SparseBundle:
        self.binary_codec.sync_from_codec()
        fixed_bundle = self.binary_codec.encode_jsonable_bundle(bundle)

        core_cells = self._collect_sparse_cells("core", fixed_bundle.core_ids)
        relation_cells = self._collect_sparse_cells("relation", fixed_bundle.relation_ids)
        all_cells = core_cells + relation_cells

        index_bytes = self._minimal_index_bytes()
        primitive_id_bytes = self._minimal_primitive_id_bytes(all_cells)

        payload = self._pack_sparse_payload(
            cells=all_cells,
            index_bytes=index_bytes,
            primitive_id_bytes=primitive_id_bytes,
        )

        return SparseBundle(
            cells=all_cells,
            payload=payload,
            index_bytes=index_bytes,
            primitive_id_bytes=primitive_id_bytes,
            core_populated_cells=len(core_cells),
            relation_populated_cells=len(relation_cells),
        )

    def decode_memory_bundle_sparse(self, payload: bytes) -> Dict[str, List[List[List[int]]]]:
        parsed = self._unpack_sparse_payload(payload)

        core_ids = [EMPTY_ID] * self.core_cell_count
        relation_ids = [EMPTY_ID] * self.relation_cell_count

        for cell in parsed.cells:
            if cell.grid_type == "core":
                if cell.cell_index >= self.core_cell_count:
                    raise BrainSparseCodecError(
                        f"Core sparse index {cell.cell_index} is outside the core grid."
                    )
                if core_ids[cell.cell_index] != EMPTY_ID:
                    raise BrainSparseCodecError(
                        f"Duplicate sparse core cell index {cell.cell_index}."
                    )
                core_ids[cell.cell_index] = cell.primitive_id
            elif cell.grid_type == "relation":
                if cell.cell_index >= self.relation_cell_count:
                    raise BrainSparseCodecError(
                        f"Relation sparse index {cell.cell_index} is outside the relation grid."
                    )
                if relation_ids[cell.cell_index] != EMPTY_ID:
                    raise BrainSparseCodecError(
                        f"Duplicate sparse relation cell index {cell.cell_index}."
                    )
                relation_ids[cell.cell_index] = cell.primitive_id
            else:
                raise BrainSparseCodecError(f"Unknown sparse grid type: {cell.grid_type}")

        return self.binary_codec.decode_jsonable_bundle(core_ids, relation_ids)

    def decode_sparse_memory(self, payload: bytes) -> BrainMemory:
        bundle = self.decode_memory_bundle_sparse(payload)
        return self.binary_codec.codec.decode_memory_bundle(bundle)

    def measure_sparse_size(self, bundle: SparseBundle) -> int:
        return bundle.total_bytes

    def _collect_sparse_cells(self, grid_type: str, values: Sequence[int]) -> List[SparseCell]:
        cells: List[SparseCell] = []
        for index, primitive_id in enumerate(values):
            if primitive_id == EMPTY_ID:
                continue
            cells.append(
                SparseCell(
                    grid_type=grid_type,
                    cell_index=index,
                    primitive_id=primitive_id,
                )
            )
        return cells

    def _pack_sparse_payload(
        self,
        cells: Sequence[SparseCell],
        index_bytes: int,
        primitive_id_bytes: int,
    ) -> bytes:
        if len(cells) > 255:
            raise BrainSparseCodecError("Sparse payload supports at most 255 populated cells.")

        index_format = self._format_for_bytes(index_bytes)
        primitive_format = self._format_for_bytes(primitive_id_bytes)

        parts = [
            MAGIC,
            struct.pack(
                "<BBBB",
                VERSION,
                index_bytes,
                primitive_id_bytes,
                len(cells),
            ),
        ]

        for cell in cells:
            grid_code = self._grid_type_to_code(cell.grid_type)
            parts.append(struct.pack("<B", grid_code))
            parts.append(struct.pack(f"<{index_format}", cell.cell_index))
            parts.append(struct.pack(f"<{primitive_format}", cell.primitive_id))

        return b"".join(parts)

    def _unpack_sparse_payload(self, payload: bytes) -> SparseBundle:
        if len(payload) < 6:
            raise BrainSparseCodecError("Sparse payload is too short.")
        if payload[:2] != MAGIC:
            raise BrainSparseCodecError("Sparse payload magic header is invalid.")

        version, index_bytes, primitive_id_bytes, cell_count = struct.unpack("<BBBB", payload[2:6])
        if version != VERSION:
            raise BrainSparseCodecError(f"Unsupported sparse payload version: {version}")

        index_format = self._format_for_bytes(index_bytes)
        primitive_format = self._format_for_bytes(primitive_id_bytes)

        offset = 6
        cells: List[SparseCell] = []
        core_count = 0
        relation_count = 0

        for _ in range(cell_count):
            entry_size = 1 + index_bytes + primitive_id_bytes
            if offset + entry_size > len(payload):
                raise BrainSparseCodecError("Sparse payload ended before all cells were decoded.")

            grid_code = struct.unpack("<B", payload[offset:offset + 1])[0]
            offset += 1

            cell_index = struct.unpack(
                f"<{index_format}",
                payload[offset:offset + index_bytes],
            )[0]
            offset += index_bytes

            primitive_id = struct.unpack(
                f"<{primitive_format}",
                payload[offset:offset + primitive_id_bytes],
            )[0]
            offset += primitive_id_bytes

            grid_type = self._grid_code_to_type(grid_code)
            cells.append(
                SparseCell(
                    grid_type=grid_type,
                    cell_index=cell_index,
                    primitive_id=primitive_id,
                )
            )

            if grid_type == "core":
                core_count += 1
            else:
                relation_count += 1

        if offset != len(payload):
            raise BrainSparseCodecError("Sparse payload contains trailing bytes.")

        return SparseBundle(
            cells=cells,
            payload=payload,
            index_bytes=index_bytes,
            primitive_id_bytes=primitive_id_bytes,
            core_populated_cells=core_count,
            relation_populated_cells=relation_count,
        )

    def _minimal_index_bytes(self) -> int:
        max_index = max(self.core_cell_count - 1, self.relation_cell_count - 1)
        return 1 if max_index <= 0xFF else 2

    @staticmethod
    def _minimal_primitive_id_bytes(cells: Sequence[SparseCell]) -> int:
        if not cells:
            return 1

        max_primitive_id = max(cell.primitive_id for cell in cells)
        if max_primitive_id <= 0xFF:
            return 1
        if max_primitive_id <= 0xFFFF:
            return 2
        return 4

    @staticmethod
    def _format_for_bytes(width: int) -> str:
        fmt = FORMAT_MAP.get(width)
        if fmt is None:
            raise BrainSparseCodecError(f"Unsupported integer width: {width}")
        return fmt

    @staticmethod
    def _grid_type_to_code(grid_type: str) -> int:
        if grid_type == "core":
            return GRID_TYPE_CORE
        if grid_type == "relation":
            return GRID_TYPE_RELATION
        raise BrainSparseCodecError(f"Unsupported sparse grid type: {grid_type}")

    @staticmethod
    def _grid_code_to_type(grid_code: int) -> str:
        if grid_code == GRID_TYPE_CORE:
            return "core"
        if grid_code == GRID_TYPE_RELATION:
            return "relation"
        raise BrainSparseCodecError(f"Unsupported sparse grid code: {grid_code}")

