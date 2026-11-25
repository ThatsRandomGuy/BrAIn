from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from brain_binary_codec import BrainBinaryCodec, EMPTY_ID
from brain_codec import BrainMemory


MAGIC = b"BC"
VERSION = 1
CORE_INDEX_BITS = 4
RELATION_INDEX_BITS = 4


@dataclass(frozen=True)
class BitPackedBundle:
    payload: bytes
    primitive_id_bits: int
    core_populated_cells: int
    relation_populated_cells: int
    core_entries: List[Tuple[int, int]]
    relation_entries: List[Tuple[int, int]]

    @property
    def total_bytes(self) -> int:
        return len(self.payload)

    @property
    def populated_cells(self) -> int:
        return self.core_populated_cells + self.relation_populated_cells


class BrainBitPackedCodecError(Exception):
    pass


class BrainBitPackedCodec:
    """
    Compact bit-packed encoding for BrAIn's sparse grid contents.

    Layout:
    - 2 bytes magic
    - 1 byte version
    - 1 byte primitive-id bit width
    - 1 byte counts header
      - upper nibble: populated core cells (0-15)
      - lower nibble: populated relation cells (0-15)
    - bitstream payload:
      - core entries first, sorted by cell index
      - relation entries second, sorted by cell index
      - each entry stores:
        - 4 bits cell index
        - N bits primitive id

    The grouped format removes per-entry grid-type bytes, and the 4-bit indices
    take advantage of BrAIn's small fixed grids.
    """

    def __init__(self, binary_codec: BrainBinaryCodec):
        self.binary_codec = binary_codec

    def encode_memory_bundle_bitpacked(self, memory: BrainMemory) -> BitPackedBundle:
        bundle = self.binary_codec.codec.encode_memory_bundle(memory)
        return self.encode_jsonable_bundle_bitpacked(bundle)

    def encode_jsonable_bundle_bitpacked(self, bundle: Dict[str, List[List[List[int]]]]) -> BitPackedBundle:
        self.binary_codec.sync_from_codec()
        fixed_bundle = self.binary_codec.encode_jsonable_bundle(bundle)

        core_entries = self._collect_entries(fixed_bundle.core_ids)
        relation_entries = self._collect_entries(fixed_bundle.relation_ids)

        if len(core_entries) > 15:
            raise BrainBitPackedCodecError("Core populated cell count exceeds 4-bit header capacity.")
        if len(relation_entries) > 15:
            raise BrainBitPackedCodecError("Relation populated cell count exceeds 4-bit header capacity.")

        primitive_id_bits = self._primitive_id_bit_width(core_entries, relation_entries)

        writer = _BitWriter()
        for cell_index, primitive_id in core_entries:
            writer.write_bits(cell_index, CORE_INDEX_BITS)
            writer.write_bits(primitive_id, primitive_id_bits)

        for cell_index, primitive_id in relation_entries:
            writer.write_bits(cell_index, RELATION_INDEX_BITS)
            writer.write_bits(primitive_id, primitive_id_bits)

        counts_header = ((len(core_entries) & 0x0F) << 4) | (len(relation_entries) & 0x0F)
        payload = b"".join(
            [
                MAGIC,
                bytes([VERSION, primitive_id_bits, counts_header]),
                writer.finish(),
            ]
        )

        return BitPackedBundle(
            payload=payload,
            primitive_id_bits=primitive_id_bits,
            core_populated_cells=len(core_entries),
            relation_populated_cells=len(relation_entries),
            core_entries=core_entries,
            relation_entries=relation_entries,
        )

    def decode_memory_bundle_bitpacked(self, payload: bytes) -> Dict[str, List[List[List[int]]]]:
        parsed = self._parse_payload(payload)

        core_ids = [EMPTY_ID] * self.binary_codec.core_cell_count
        relation_ids = [EMPTY_ID] * self.binary_codec.relation_cell_count

        for cell_index, primitive_id in parsed.core_entries:
            if cell_index >= self.binary_codec.core_cell_count:
                raise BrainBitPackedCodecError(f"Core cell index {cell_index} is out of range.")
            if core_ids[cell_index] != EMPTY_ID:
                raise BrainBitPackedCodecError(f"Duplicate core cell index {cell_index} in bit-packed payload.")
            core_ids[cell_index] = primitive_id

        for cell_index, primitive_id in parsed.relation_entries:
            if cell_index >= self.binary_codec.relation_cell_count:
                raise BrainBitPackedCodecError(f"Relation cell index {cell_index} is out of range.")
            if relation_ids[cell_index] != EMPTY_ID:
                raise BrainBitPackedCodecError(f"Duplicate relation cell index {cell_index} in bit-packed payload.")
            relation_ids[cell_index] = primitive_id

        return self.binary_codec.decode_jsonable_bundle(core_ids, relation_ids)

    def decode_bitpacked_memory(self, payload: bytes) -> BrainMemory:
        bundle = self.decode_memory_bundle_bitpacked(payload)
        return self.binary_codec.codec.decode_memory_bundle(bundle)

    def _parse_payload(self, payload: bytes) -> BitPackedBundle:
        if len(payload) < 5:
            raise BrainBitPackedCodecError("Bit-packed payload is too short.")
        if payload[:2] != MAGIC:
            raise BrainBitPackedCodecError("Bit-packed payload magic header is invalid.")

        version = payload[2]
        primitive_id_bits = payload[3]
        counts_header = payload[4]

        if version != VERSION:
            raise BrainBitPackedCodecError(f"Unsupported bit-packed payload version: {version}")
        if primitive_id_bits <= 0:
            raise BrainBitPackedCodecError("Primitive-id bit width must be positive.")

        core_count = (counts_header >> 4) & 0x0F
        relation_count = counts_header & 0x0F

        reader = _BitReader(payload[5:])

        core_entries: List[Tuple[int, int]] = []
        relation_entries: List[Tuple[int, int]] = []

        for _ in range(core_count):
            core_entries.append(
                (
                    reader.read_bits(CORE_INDEX_BITS),
                    reader.read_bits(primitive_id_bits),
                )
            )

        for _ in range(relation_count):
            relation_entries.append(
                (
                    reader.read_bits(RELATION_INDEX_BITS),
                    reader.read_bits(primitive_id_bits),
                )
            )

        if not reader.only_padding_remaining():
            raise BrainBitPackedCodecError("Bit-packed payload contains trailing non-zero bits.")

        return BitPackedBundle(
            payload=payload,
            primitive_id_bits=primitive_id_bits,
            core_populated_cells=core_count,
            relation_populated_cells=relation_count,
            core_entries=core_entries,
            relation_entries=relation_entries,
        )

    @staticmethod
    def _collect_entries(values: Sequence[int]) -> List[Tuple[int, int]]:
        entries: List[Tuple[int, int]] = []
        for index, primitive_id in enumerate(values):
            if primitive_id == EMPTY_ID:
                continue
            entries.append((index, primitive_id))
        return entries

    @staticmethod
    def _primitive_id_bit_width(
        core_entries: Sequence[Tuple[int, int]],
        relation_entries: Sequence[Tuple[int, int]],
    ) -> int:
        max_primitive_id = 0
        for _, primitive_id in list(core_entries) + list(relation_entries):
            max_primitive_id = max(max_primitive_id, primitive_id)
        return max(1, max_primitive_id.bit_length())


class _BitWriter:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current_byte = 0
        self._bits_filled = 0

    def write_bits(self, value: int, bit_count: int) -> None:
        if bit_count < 0:
            raise BrainBitPackedCodecError("Bit count must be non-negative.")
        if bit_count == 0:
            return
        if value < 0 or value >= (1 << bit_count):
            raise BrainBitPackedCodecError(f"Value {value} does not fit in {bit_count} bits.")

        remaining = bit_count
        local_value = value

        while remaining > 0:
            available = 8 - self._bits_filled
            take = min(available, remaining)
            chunk_mask = (1 << take) - 1
            chunk = local_value & chunk_mask

            self._current_byte |= chunk << self._bits_filled
            self._bits_filled += take
            local_value >>= take
            remaining -= take

            if self._bits_filled == 8:
                self._buffer.append(self._current_byte)
                self._current_byte = 0
                self._bits_filled = 0

    def finish(self) -> bytes:
        if self._bits_filled > 0:
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._bits_filled = 0
        return bytes(self._buffer)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._byte_index = 0
        self._bit_index = 0

    def read_bits(self, bit_count: int) -> int:
        if bit_count < 0:
            raise BrainBitPackedCodecError("Bit count must be non-negative.")
        result = 0
        bits_written = 0

        while bits_written < bit_count:
            if self._byte_index >= len(self._payload):
                raise BrainBitPackedCodecError("Bit-packed payload ended unexpectedly.")

            current_byte = self._payload[self._byte_index]
            available = 8 - self._bit_index
            take = min(available, bit_count - bits_written)
            chunk_mask = (1 << take) - 1
            chunk = (current_byte >> self._bit_index) & chunk_mask

            result |= chunk << bits_written
            bits_written += take
            self._bit_index += take

            if self._bit_index == 8:
                self._byte_index += 1
                self._bit_index = 0

        return result

    def only_padding_remaining(self) -> bool:
        if self._byte_index >= len(self._payload):
            return True

        if self._bit_index == 0:
            return all(byte == 0 for byte in self._payload[self._byte_index:])

        current_byte = self._payload[self._byte_index]
        remaining_mask = ((1 << (8 - self._bit_index)) - 1) << self._bit_index
        if current_byte & remaining_mask:
            return False

        return all(byte == 0 for byte in self._payload[self._byte_index + 1 :])

