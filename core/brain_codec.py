from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import json

try:
    from PIL import Image
except ImportError:  # Optional dependency for image export/import
    Image = None


Color = Tuple[int, int, int]
Grid = List[List[Color]]

EMPTY: Color = (0, 0, 0)
UNKNOWN: Color = (255, 0, 255)
UNKNOWN_PRIMITIVE = "__unknown__"


# -----------------------------
# Core schema
# -----------------------------

@dataclass
class BrainMemory:
    actor: str = ""
    intent: str = ""
    action: str = ""
    object: str = ""
    topic: str = ""
    modifiers: List[str] = field(default_factory=list)
    negations: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    relations: List[Dict[str, str]] = field(default_factory=list)
    confidence: Dict[str, object] = field(default_factory=dict)
    time: Optional[str] = None
    priority: str = "normal"

    def to_dict(self) -> Dict:
        return {
            "actor": self.actor,
            "intent": self.intent,
            "action": self.action,
            "object": self.object,
            "topic": self.topic,
            "modifiers": self.modifiers,
            "negations": self.negations,
            "constraints": self.constraints,
            "conditions": self.conditions,
            "relations": self.relations,
            "confidence": self.confidence,
            "time": self.time,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BrainMemory":
        return cls(
            actor=data.get("actor", ""),
            intent=data.get("intent", ""),
            action=data.get("action", ""),
            object=data.get("object", ""),
            topic=data.get("topic", ""),
            modifiers=list(data.get("modifiers", [])),
            negations=list(data.get("negations", [])),
            constraints=list(data.get("constraints", [])),
            conditions=list(data.get("conditions", [])),
            relations=list(data.get("relations", [])),
            confidence=dict(data.get("confidence", {})),
            time=data.get("time"),
            priority=data.get("priority", "normal"),
        )


# -----------------------------
# Primitive palette
# -----------------------------

DEFAULT_PRIMITIVE_TO_COLOR: Dict[str, Color] = {
    # Actors
    "actor:user": (255, 0, 0),
    "actor:assistant": (200, 0, 0),

    # Intent
    "intent:request": (0, 0, 255),
    "intent:question": (0, 80, 255),
    "intent:inform": (0, 150, 255),

    # Actions
    "action:create": (255, 192, 203),
    "action:modify": (255, 140, 0),
    "action:store": (128, 0, 128),
    "action:delete": (139, 0, 0),
    "action:compare": (0, 206, 209),

    # Objects
    "object:game": (0, 255, 0),
    "object:app": (0, 200, 0),
    "object:website": (34, 139, 34),
    "object:file": (50, 205, 50),

    # Topics
    "topic:apples": (255, 255, 0),
    "topic:brain": (255, 215, 0),
    "topic:compression": (218, 165, 32),

    # Modifiers
    "modifier:horror": (17, 17, 17),
    "modifier:funny": (255, 105, 180),
    "modifier:educational": (135, 206, 250),

    # Negations / constraints / conditions / metadata
    "negation:educational": (255, 165, 0),
    "constraint:low_hardware": (128, 128, 128),
    "condition:unless_requested": (75, 0, 130),
    "priority:normal": (245, 245, 245),
    "priority:high": (255, 69, 0),
    "priority:low": (169, 169, 169),
    "time:now": (173, 216, 230),

    # Relation types
    "relation_type:about": (0, 255, 255),
    "relation_type:has_modifier": (186, 85, 211),
    "relation_type:negates": (255, 99, 71),
    "relation_type:constraint": (112, 128, 144),
    "relation_type:condition": (72, 61, 139),
    "relation_type:time": (70, 130, 180),
    "relation_type:compare_left": (154, 205, 50),
    "relation_type:compare_right": (107, 142, 35),
}


# -----------------------------
# Grid layout
# -----------------------------

CORE_GRID_ROWS = 5
CORE_GRID_COLS = 3

RELATION_GRID_ROWS = 4
RELATION_GRID_COLS = 3

CELL_SIZE = 32

CORE_GRID_MAP = {
    "actor": (0, 0),
    "intent": (0, 1),
    "priority": (0, 2),
    "action": (1, 0),
    "object": (1, 1),
    "topic": (1, 2),
    "modifiers[0]": (2, 0),
    "modifiers[1]": (2, 1),
    "modifiers[2]": (2, 2),
    "negations[0]": (3, 0),
    "constraints[0]": (3, 1),
    "constraints[1]": (3, 2),
    "conditions[0]": (4, 0),
    "time": (4, 1),
    "reserved": (4, 2),
}


class BrainCodecError(Exception):
    pass


class BrainCodec:
    def __init__(
        self,
        primitive_to_color: Optional[Dict[str, Color]] = None,
        registry_path: str | Path = "brain_registry.json",
        auto_register: bool = True,
    ):
        self.registry_path = Path(registry_path)
        self.auto_register = auto_register

        self.primitive_to_color: Dict[str, Color] = dict(DEFAULT_PRIMITIVE_TO_COLOR)
        if primitive_to_color:
            self.primitive_to_color.update(primitive_to_color)

        self._load_registry()

        self.color_to_primitive: Dict[Color, str] = {}
        self._rebuild_reverse_index()

    # -----------------------------
    # Registry management
    # -----------------------------

    def _load_registry(self) -> None:
        if not self.registry_path.exists():
            return

        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for primitive, color_list in data.items():
            if not isinstance(color_list, list) or len(color_list) != 3:
                continue
            color = tuple(int(c) for c in color_list)
            self.primitive_to_color[primitive] = color  # type: ignore[assignment]

    def save_registry(self) -> None:
        serializable = {
            primitive: [color[0], color[1], color[2]]
            for primitive, color in sorted(self.primitive_to_color.items())
            if primitive not in DEFAULT_PRIMITIVE_TO_COLOR or DEFAULT_PRIMITIVE_TO_COLOR[primitive] != color
        }

        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def _rebuild_reverse_index(self) -> None:
        self.color_to_primitive = {}
        for primitive, color in self.primitive_to_color.items():
            if color in (EMPTY, UNKNOWN):
                raise BrainCodecError(f"Primitive {primitive} conflicts with reserved color {color}")
            self.color_to_primitive[color] = primitive

    def register_primitive(self, primitive: str) -> Color:
        if primitive in self.primitive_to_color:
            return self.primitive_to_color[primitive]

        color = self._generate_unique_color(primitive)
        self.primitive_to_color[primitive] = color
        self.color_to_primitive[color] = primitive
        self.save_registry()
        return color

    def _generate_unique_color(self, primitive: str) -> Color:
        attempt = 0
        while True:
            seed = primitive if attempt == 0 else f"{primitive}#{attempt}"
            digest = hashlib.sha256(seed.encode("utf-8")).digest()

            r = max(1, digest[0])
            g = max(1, digest[1])
            b = max(1, digest[2])

            color = (r, g, b)

            if color not in (EMPTY, UNKNOWN) and color not in self.color_to_primitive:
                return color

            attempt += 1

    # -----------------------------
    # Primitive helpers
    # -----------------------------

    @staticmethod
    def make_primitive(field_type: str, value: str) -> str:
        return f"{field_type}:{value}"

    def ensure_primitive(self, primitive: str) -> Color:
        if primitive in self.primitive_to_color:
            return self.primitive_to_color[primitive]

        if not self.auto_register:
            return UNKNOWN

        return self.register_primitive(primitive)

    def primitive_to_rgb(self, primitive: str) -> Color:
        return self.ensure_primitive(primitive)

    def rgb_to_primitive(self, color: Color) -> Optional[str]:
        if color == EMPTY:
            return None
        if color == UNKNOWN:
            return "UNKNOWN"
        return self.color_to_primitive.get(color, "UNKNOWN")

    # -----------------------------
    # Resolution helpers
    # -----------------------------

    @staticmethod
    def resolve_memory(memory: BrainMemory) -> BrainMemory:
        resolved = BrainMemory.from_dict(memory.to_dict())
        negated = set(resolved.negations)
        resolved.modifiers = [m for m in resolved.modifiers if m not in negated]
        return resolved

    # -----------------------------
    # Core grid encoding
    # -----------------------------

    def encode_memory(self, memory: BrainMemory) -> Grid:
        grid: Grid = [[EMPTY for _ in range(CORE_GRID_COLS)] for _ in range(CORE_GRID_ROWS)]

        def set_cell(row: int, col: int, primitive: Optional[str]) -> None:
            if not primitive:
                return
            grid[row][col] = self.primitive_to_rgb(primitive)

        single_fields = ["actor", "intent", "priority", "action", "object", "topic", "time"]
        for field_name in single_fields:
            value = getattr(memory, field_name)
            if value:
                row, col = CORE_GRID_MAP[field_name]
                primitive = self.make_primitive(field_name, value)
                set_cell(row, col, primitive)

        for idx in range(3):
            if idx < len(memory.modifiers):
                row, col = CORE_GRID_MAP[f"modifiers[{idx}]"]
                primitive = self.make_primitive("modifier", memory.modifiers[idx])
                set_cell(row, col, primitive)

        if memory.negations:
            row, col = CORE_GRID_MAP["negations[0]"]
            primitive = self.make_primitive("negation", memory.negations[0])
            set_cell(row, col, primitive)

        for idx in range(2):
            if idx < len(memory.constraints):
                row, col = CORE_GRID_MAP[f"constraints[{idx}]"]
                primitive = self.make_primitive("constraint", memory.constraints[idx])
                set_cell(row, col, primitive)

        if memory.conditions:
            row, col = CORE_GRID_MAP["conditions[0]"]
            primitive = self.make_primitive("condition", memory.conditions[0])
            set_cell(row, col, primitive)

        return grid

    def decode_grid(self, grid: Grid) -> BrainMemory:
        self._validate_grid(grid, CORE_GRID_ROWS, CORE_GRID_COLS)
        memory = BrainMemory()

        def read_cell(field_name: str) -> Optional[str]:
            row, col = CORE_GRID_MAP[field_name]
            primitive = self.rgb_to_primitive(grid[row][col])
            if primitive in (None, "UNKNOWN"):
                return primitive
            _, value = primitive.split(":", 1)
            return value

        for field_name in ["actor", "intent", "priority", "action", "object", "topic", "time"]:
            value = read_cell(field_name)
            if value and value != "UNKNOWN":
                setattr(memory, field_name, value)

        for idx in range(3):
            row, col = CORE_GRID_MAP[f"modifiers[{idx}]"]
            primitive = self.rgb_to_primitive(grid[row][col])
            if primitive and primitive != "UNKNOWN":
                _, value = primitive.split(":", 1)
                memory.modifiers.append(value)

        row, col = CORE_GRID_MAP["negations[0]"]
        primitive = self.rgb_to_primitive(grid[row][col])
        if primitive and primitive != "UNKNOWN":
            _, value = primitive.split(":", 1)
            memory.negations.append(value)

        for idx in range(2):
            row, col = CORE_GRID_MAP[f"constraints[{idx}]"]
            primitive = self.rgb_to_primitive(grid[row][col])
            if primitive and primitive != "UNKNOWN":
                _, value = primitive.split(":", 1)
                memory.constraints.append(value)

        row, col = CORE_GRID_MAP["conditions[0]"]
        primitive = self.rgb_to_primitive(grid[row][col])
        if primitive and primitive != "UNKNOWN":
            _, value = primitive.split(":", 1)
            memory.conditions.append(value)

        return memory

    # -----------------------------
    # Relation grid encoding
    # -----------------------------

    def encode_relations(self, memory: BrainMemory) -> Grid:
        grid: Grid = [[EMPTY for _ in range(RELATION_GRID_COLS)] for _ in range(RELATION_GRID_ROWS)]

        for row_index, relation in enumerate(memory.relations[:RELATION_GRID_ROWS]):
            rel_type = relation.get("type", "")
            source = relation.get("source", "")
            target = relation.get("target", "")

            if rel_type:
                grid[row_index][0] = self.primitive_to_rgb(self.make_primitive("relation_type", rel_type))
            if source:
                grid[row_index][1] = self.primitive_to_rgb(
                    self._guess_relation_endpoint_primitive(
                        value=source,
                        relation_type=rel_type,
                        endpoint_role="source",
                    )
                )
            if target:
                grid[row_index][2] = self.primitive_to_rgb(
                    self._guess_relation_endpoint_primitive(
                        value=target,
                        relation_type=rel_type,
                        endpoint_role="target",
                    )
                )

        return grid

    def decode_relations(self, relation_grid: Grid) -> List[Dict[str, str]]:
        self._validate_grid(relation_grid, RELATION_GRID_ROWS, RELATION_GRID_COLS)
        relations: List[Dict[str, str]] = []

        for row in relation_grid:
            type_primitive = self.rgb_to_primitive(row[0])
            source_primitive = self.rgb_to_primitive(row[1])
            target_primitive = self.rgb_to_primitive(row[2])

            if not type_primitive or type_primitive == "UNKNOWN":
                continue
            if not type_primitive.startswith("relation_type:"):
                continue

            rel_type = type_primitive.split(":", 1)[1]
            source = self._primitive_value_or_unknown(source_primitive)
            target = self._primitive_value_or_unknown(target_primitive)

            relations.append({
                "type": rel_type,
                "source": source,
                "target": target,
            })

        return relations

    def _guess_relation_endpoint_primitive(
        self,
        value: str,
        relation_type: str = "",
        endpoint_role: str = "target",
    ) -> str:
        candidates = [
            f"object:{value}",
            f"topic:{value}",
            f"modifier:{value}",
            f"negation:{value}",
            f"constraint:{value}",
            f"condition:{value}",
            f"action:{value}",
            f"time:{value}",
            f"actor:{value}",
            f"intent:{value}",
            f"priority:{value}",
        ]

        for primitive in candidates:
            if primitive in self.primitive_to_color:
                return primitive

        if value == "action":
            return "action:compare" if "action:compare" in self.primitive_to_color else self.make_primitive("topic", value)

        if relation_type in {"compare_left", "compare_right"} and endpoint_role == "target":
            # Preserve compare operands as relation-first entities instead of
            # leaking them into topic/object space.
            return self.make_primitive("compare_entity", value)

        dynamic_candidates = [
            self.make_primitive("topic", value),
            self.make_primitive("object", value),
            self.make_primitive("modifier", value),
            self.make_primitive("constraint", value),
            self.make_primitive("condition", value),
        ]

        return dynamic_candidates[0]

    def _primitive_value_or_unknown(self, primitive: Optional[str]) -> str:
        if primitive in (None, "UNKNOWN"):
            return "unknown"
        return primitive.split(":", 1)[1]

    # -----------------------------
    # Combined bundle helpers
    # -----------------------------

    def encode_memory_bundle(self, memory: BrainMemory) -> Dict[str, List[List[List[int]]]]:
        core_grid = self.encode_memory(memory)
        relation_grid = self.encode_relations(memory)
        return {
            "core_grid": self.grid_to_jsonable(core_grid),
            "relation_grid": self.grid_to_jsonable(relation_grid),
        }

    def decode_memory_bundle(self, bundle: Dict[str, List[List[List[int]]]]) -> BrainMemory:
        core_grid = self.jsonable_to_grid(bundle["core_grid"])
        relation_grid = self.jsonable_to_grid(bundle["relation_grid"])

        memory = self.decode_grid(core_grid)
        memory.relations = self.decode_relations(relation_grid)
        return memory

    # -----------------------------
    # Serialization helpers
    # -----------------------------

    @staticmethod
    def grid_to_jsonable(grid: Grid) -> List[List[List[int]]]:
        return [[[r, g, b] for (r, g, b) in row] for row in grid]

    @staticmethod
    def jsonable_to_grid(data: List[List[List[int]]]) -> Grid:
        return [[tuple(cell) for cell in row] for row in data]  # type: ignore[arg-type]

    def save_grid_json(self, grid: Grid, path: str | Path) -> None:
        rows = len(grid)
        cols = len(grid[0]) if grid else 0
        self._validate_grid(grid, rows, cols)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.grid_to_jsonable(grid), f, indent=2)

    def load_grid_json(self, path: str | Path) -> Grid:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.jsonable_to_grid(data)

    # -----------------------------
    # Image helpers
    # -----------------------------

    def export_grid_image(self, grid: Grid, path: str | Path, cell_size: int = CELL_SIZE) -> None:
        if Image is None:
            raise BrainCodecError("Pillow is not installed. Run: pip install pillow")

        rows = len(grid)
        cols = len(grid[0]) if grid else 0
        self._validate_grid(grid, rows, cols)

        img = Image.new("RGB", (cols * cell_size, rows * cell_size))
        for row in range(rows):
            for col in range(cols):
                color = grid[row][col]
                for y in range(row * cell_size, (row + 1) * cell_size):
                    for x in range(col * cell_size, (col + 1) * cell_size):
                        img.putpixel((x, y), color)
        img.save(path)

    def import_grid_image(self, path: str | Path, rows: int, cols: int, cell_size: int = CELL_SIZE) -> Grid:
        if Image is None:
            raise BrainCodecError("Pillow is not installed. Run: pip install pillow")

        img = Image.open(path).convert("RGB")
        expected_size = (cols * cell_size, rows * cell_size)
        if img.size != expected_size:
            raise BrainCodecError(f"Unexpected image size {img.size}. Expected {expected_size}.")

        grid: Grid = [[EMPTY for _ in range(cols)] for _ in range(rows)]
        for row in range(rows):
            for col in range(cols):
                sample_x = col * cell_size + cell_size // 2
                sample_y = row * cell_size + cell_size // 2
                grid[row][col] = img.getpixel((sample_x, sample_y))

        self._validate_grid(grid, rows, cols)
        return grid

    # -----------------------------
    # Human reconstruction
    # -----------------------------

    @staticmethod
    def reconstruct_text(memory: BrainMemory) -> str:
        memory = BrainCodec.resolve_memory(memory)

        if memory.action == "compare":
            left = None
            right = None
            for relation in memory.relations:
                if relation.get("type") == "compare_left":
                    left = relation.get("target")
                elif relation.get("type") == "compare_right":
                    right = relation.get("target")

            if left and right:
                sentence = f"Compare {left.replace('_', ' ')} with {right.replace('_', ' ')}"
            elif memory.topic:
                sentence = f"Compare {memory.topic}"
            elif memory.object:
                sentence = f"Compare {memory.object}"
            else:
                sentence = "Compare"

        elif memory.action == "create":
            parts = ["create"]
            if memory.modifiers:
                parts.extend(memory.modifiers)
            if memory.object:
                parts.append(memory.object)
            if memory.topic:
                parts.append(f"about {memory.topic}")
            sentence = " ".join(parts).strip() or "Create"

        elif memory.action == "modify":
            parts = ["modify"]
            if memory.object:
                parts.append(memory.object)
            if memory.topic:
                parts.append(f"about {memory.topic}")
            sentence = " ".join(parts).strip() or "Modify"

        else:
            parts = []
            if memory.action:
                parts.append(memory.action)
            if memory.object:
                parts.append(memory.object)
            if memory.topic:
                parts.append(f"about {memory.topic}")
            sentence = " ".join(parts).strip() or "No recoverable semantic content"

        if memory.negations:
            sentence += ", not " + ", not ".join(memory.negations)

        if memory.constraints:
            sentence += ". Constraints: " + ", ".join(memory.constraints)

        if memory.conditions:
            sentence += ". Conditions: " + ", ".join(memory.conditions)

        if memory.time:
            sentence += f". Time: {memory.time}"

        if memory.priority and memory.priority != "normal":
            sentence += f". Priority: {memory.priority}"

        sentence = sentence.strip()
        return sentence[0].upper() + sentence[1:] + ("" if sentence.endswith(".") else ".")

    # -----------------------------
    # Validation
    # -----------------------------

    @staticmethod
    def _validate_grid(grid: Grid, expected_rows: int, expected_cols: int) -> None:
        if len(grid) != expected_rows:
            raise BrainCodecError(f"Grid must have {expected_rows} rows.")
        for row in grid:
            if len(row) != expected_cols:
                raise BrainCodecError(f"Each row must have {expected_cols} columns.")
            for color in row:
                if not isinstance(color, tuple) or len(color) != 3:
                    raise BrainCodecError("Each cell must be an (R, G, B) tuple.")
                for channel in color:
                    if not isinstance(channel, int) or not (0 <= channel <= 255):
                        raise BrainCodecError("RGB channels must be integers between 0 and 255.")


if __name__ == "__main__":
    codec = BrainCodec()

    memory = BrainMemory(
        actor="user",
        intent="request",
        action="create",
        object="simulator",
        topic="strawberries",
        modifiers=["cozy", "farming"],
        relations=[
            {"type": "about", "source": "simulator", "target": "strawberries"},
            {"type": "has_modifier", "source": "simulator", "target": "cozy"},
            {"type": "has_modifier", "source": "simulator", "target": "farming"},
        ],
        confidence={
            "actor": {"value": "user", "source": "default", "score": 1.0},
            "intent": {"value": "request", "source": "pattern", "score": 0.95},
            "action": {"value": "create", "source": "pattern", "score": 0.95},
            "object": {"value": "simulator", "source": "promotion", "score": 0.68},
            "topic": {"value": "strawberries", "source": "fallback", "score": 0.72},
            "modifiers": [
                {"value": "cozy", "source": "promotion", "score": 0.62},
                {"value": "farming", "source": "promotion", "score": 0.62},
            ],
        },
    )

    bundle = codec.encode_memory_bundle(memory)
    print("Encoded bundle:")
    print(json.dumps(bundle, indent=2))

    decoded = codec.decode_memory_bundle(bundle)
    decoded.confidence = memory.confidence
    print("\nDecoded semantic memory:")
    print(json.dumps(decoded.to_dict(), indent=2))

    print("\nReconstructed text:")
    print(codec.reconstruct_text(decoded))

    print("\nRegistry path:")
    print(codec.registry_path)
