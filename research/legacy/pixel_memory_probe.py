from __future__ import annotations

# Historical harnesses run with the maintained core on the import path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/"core"))

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from brain_hybrid_prototype import BrainHybridPrototype
from brain_pixel_memory import PixelMemoryCanvas, PixelProjectionConcept, PixelRelationship


@dataclass(frozen=True)
class PixelProbeCase:
    name: str
    query: str
    concepts: List[PixelProjectionConcept]
    relationships: List[PixelRelationship]
    expected_concept_id: str


@dataclass(frozen=True)
class PixelDecisionProbeCase:
    name: str
    turns: List[str]
    query: str
    expected_fragment: str
    expected_source: str


def _graph_only_retrieval(query: str, concepts: Sequence[PixelProjectionConcept]) -> Dict[str, object]:
    query_terms = set(query.lower().replace("?", " ").replace(".", " ").split())
    ranked = []
    for concept in concepts:
        terms = set(concept.terms) or set(concept.label.lower().split())
        if not terms:
            continue
        overlap = len(query_terms & terms) / float(len(query_terms | terms))
        ranked.append(
            {
                "concept_id": concept.concept_id,
                "label": concept.label,
                "score": round(overlap, 4),
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["concept_id"])))
    return {
        "top_concept_id": ranked[0]["concept_id"] if ranked else "",
        "ranked": ranked[:5],
    }


def _cases() -> List[PixelProbeCase]:
    return [
        PixelProbeCase(
            name="simple_factual_memory",
            query="What stores laptop charge?",
            concepts=[
                PixelProjectionConcept("fact:battery", "battery stores laptop charge", ["battery", "stores", "laptop", "charge"], importance=0.8),
                PixelProjectionConcept("fact:sky", "sky appears blue", ["sky", "blue"], importance=0.45),
            ],
            relationships=[],
            expected_concept_id="fact:battery",
        ),
        PixelProbeCase(
            name="overwritten_state_recall",
            query="Where are the frozen berries now?",
            concepts=[
                PixelProjectionConcept("state:berries:old", "frozen berries in pantry", ["frozen", "berries", "pantry"], importance=0.45, contradiction_links=["state:berries:new"]),
                PixelProjectionConcept("state:berries:new", "frozen berries in freezer", ["frozen", "berries", "freezer"], importance=0.85),
            ],
            relationships=[
                PixelRelationship("state:berries:old", "state:berries:new", "contradicts", 1.0),
                PixelRelationship("state:berries:old", "state:berries:new", "temporal", 0.8),
            ],
            expected_concept_id="state:berries:new",
        ),
        PixelProbeCase(
            name="merged_constraint_recall",
            query="What packing constraints did I give?",
            concepts=[
                PixelProjectionConcept("constraint:backpack", "bring one backpack", ["packing", "constraint", "one", "backpack"], importance=0.7),
                PixelProjectionConcept("constraint:weight", "keep weight under 8 kilograms", ["packing", "constraint", "weight", "under", "8"], importance=0.7),
                PixelProjectionConcept("constraint:laptop", "laptop easy access pocket", ["packing", "constraint", "laptop", "easy", "access"], importance=0.7),
            ],
            relationships=[
                PixelRelationship("constraint:backpack", "constraint:weight", "semantic", 0.75),
                PixelRelationship("constraint:weight", "constraint:laptop", "semantic", 0.75),
            ],
            expected_concept_id="constraint:backpack",
        ),
        PixelProbeCase(
            name="temporal_sequence_recall",
            query="What happened after reproducing the bug?",
            concepts=[
                PixelProjectionConcept("step:reproduce", "reproduce the bug", ["reproduce", "bug"], importance=0.62, temporal_links=["step:test"]),
                PixelProjectionConcept("step:test", "update the tests", ["after", "update", "tests"], importance=0.72),
                PixelProjectionConcept("step:review", "wait for review", ["review"], importance=0.58),
            ],
            relationships=[
                PixelRelationship("step:reproduce", "step:test", "temporal", 1.0),
                PixelRelationship("step:test", "step:review", "temporal", 0.8),
            ],
            expected_concept_id="step:test",
        ),
        PixelProbeCase(
            name="concept_hierarchy_recall",
            query="Which fruit snack did I mention?",
            concepts=[
                PixelProjectionConcept("category:fruit", "fruit", ["fruit"], abstraction_level=3, importance=0.65, hierarchy_links=["item:apple"]),
                PixelProjectionConcept("item:apple", "apple fruit snack", ["apple", "fruit", "snack"], abstraction_level=2, importance=0.78),
                PixelProjectionConcept("item:charger", "phone charger", ["phone", "charger"], abstraction_level=1, importance=0.5),
            ],
            relationships=[PixelRelationship("category:fruit", "item:apple", "hierarchy", 1.0)],
            expected_concept_id="item:apple",
        ),
        PixelProbeCase(
            name="contradiction_handling",
            query="What color should I assume the status light is?",
            concepts=[
                PixelProjectionConcept("status:red", "status light is red", ["status", "light", "red"], importance=0.55, contradiction_links=["status:green"]),
                PixelProjectionConcept("status:green", "status light is green", ["status", "light", "green"], importance=0.86),
            ],
            relationships=[PixelRelationship("status:red", "status:green", "contradicts", 1.0)],
            expected_concept_id="status:green",
        ),
    ]


def _decision_cases() -> List[PixelDecisionProbeCase]:
    return [
        PixelDecisionProbeCase(
            name="overwrite_memory_conflict",
            turns=[
                "I keep frozen berries in the pantry.",
                "Actually, I moved the frozen berries to the freezer.",
            ],
            query="pantry",
            expected_fragment="freezer",
            expected_source="pixel",
        ),
        PixelDecisionProbeCase(
            name="contradiction_resolution",
            turns=[
                "The status light is red.",
                "Correction: the status light is green.",
            ],
            query="red status light",
            expected_fragment="green",
            expected_source="pixel",
        ),
        PixelDecisionProbeCase(
            name="temporal_next_step_prediction",
            turns=[
                "First reproduce the bug.",
                "Then update the tests.",
                "After tests pass, request review.",
            ],
            query="What should happen after reproducing the bug?",
            expected_fragment="update the tests",
            expected_source="pixel",
        ),
        PixelDecisionProbeCase(
            name="partial_reconstruction_edge_case",
            turns=[
                "For the demo kit, the red cable belongs with the sensor board.",
                "The calibration note says sensor board first, then cable check.",
            ],
            query="red wiring",
            expected_fragment="red cable",
            expected_source="pixel_or_hybrid",
        ),
        PixelDecisionProbeCase(
            name="constraint_merging_multi_region",
            turns=[
                "Packing constraint: bring one backpack.",
                "Packing constraint: keep weight under eight kilograms.",
                "Packing constraint: laptop must stay in the easy access pocket.",
            ],
            query="What packing constraints did I give?",
            expected_fragment="backpack",
            expected_source="pixel_or_hybrid",
        ),
    ]


def _run_reconstruction(case: PixelDecisionProbeCase, *, pixel_memory_enabled: bool) -> Dict[str, object]:
    temp_dir = Path(tempfile.mkdtemp(prefix="brain_pixel_decision_probe_"))
    try:
        prototype = BrainHybridPrototype(
            state_dir=temp_dir,
            llm_provider="deterministic",
            bootstrap_reasoning=False,
            pixel_memory_enabled=pixel_memory_enabled,
        )
        for turn in case.turns:
            prototype.process_user_turn(turn, inspect=False)
        return prototype.reconstruct_memory_context(
            user_input_text=case.query,
            retrieved=[],
            recalled_turns=[],
            merged_memory_state={},
        ).to_dict()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _first_partial_text(reconstructed: Dict[str, object]) -> str:
    partials = reconstructed.get("partial_matches", [])
    if not isinstance(partials, list) or not partials:
        return ""
    first = partials[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("text", ""))


def _pixel_only_result(reconstructed: Dict[str, object]) -> Dict[str, object]:
    evidence = reconstructed.get("evidence_breakdown", {})
    pixel_readback = evidence.get("pixel_readback", {}) if isinstance(evidence, dict) else {}
    if not isinstance(pixel_readback, dict):
        return {}
    dominant = pixel_readback.get("dominant_concepts", [])
    temporal_next = pixel_readback.get("temporal_next_concepts", [])
    first_dominant = dominant[0] if isinstance(dominant, list) and dominant else {}
    first_temporal = temporal_next[0] if isinstance(temporal_next, list) and temporal_next else {}
    return {
        "spatial_confidence": pixel_readback.get("spatial_confidence"),
        "dominance_gap": pixel_readback.get("dominance_gap"),
        "reconstructed_summary": pixel_readback.get("reconstructed_summary"),
        "top_dominant_concept": first_dominant,
        "top_temporal_next_concept": first_temporal,
        "dominant_regions": pixel_readback.get("dominant_regions", []),
        "region_strengths": pixel_readback.get("region_strengths", []),
    }


def run_pixel_decision_probes() -> Dict[str, object]:
    results = []
    for case in _decision_cases():
        graph_only = _run_reconstruction(case, pixel_memory_enabled=False)
        final = _run_reconstruction(case, pixel_memory_enabled=True)
        evidence = final.get("evidence_breakdown", {})
        pixel_reasoning = evidence.get("pixel_reasoning", {}) if isinstance(evidence, dict) else {}
        decision_source = str(evidence.get("decision_source", ""))
        final_text = _first_partial_text(final)
        if case.expected_source == "pixel_or_hybrid":
            source_pass = decision_source in {"pixel", "hybrid"}
        else:
            source_pass = decision_source == case.expected_source
        fragment_pass = case.expected_fragment.lower() in final_text.lower()
        results.append(
            {
                "case": case.name,
                "query": case.query,
                "expected_fragment": case.expected_fragment,
                "expected_source": case.expected_source,
                "graph_only_result": {
                    "top_partial_text": _first_partial_text(graph_only),
                    "evidence_breakdown": graph_only.get("evidence_breakdown", {}),
                },
                "pixel_only_result": _pixel_only_result(final),
                "final_result": {
                    "decision_source": decision_source,
                    "graph_confidence": evidence.get("graph_confidence"),
                    "pixel_confidence": evidence.get("pixel_confidence"),
                    "dominance_gap": evidence.get("dominance_gap"),
                    "pixel_decision_margin": evidence.get("pixel_decision_margin"),
                    "reasoning_mode": pixel_reasoning.get("reasoning_mode") if isinstance(pixel_reasoning, dict) else "",
                    "selected_output_type": pixel_reasoning.get("selected_output_type") if isinstance(pixel_reasoning, dict) else "",
                    "active_region_count": pixel_reasoning.get("active_region_count") if isinstance(pixel_reasoning, dict) else 0,
                    "strong_region_count": pixel_reasoning.get("strong_region_count") if isinstance(pixel_reasoning, dict) else 0,
                    "top_partial_text": final_text,
                    "source_type": (final.get("partial_matches", [{}]) or [{}])[0].get("source_type", ""),
                },
                "passed": source_pass and fragment_pass,
            }
        )
    return {
        "probe_count": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "results": results,
    }


def run_pixel_memory_probes() -> Dict[str, object]:
    results = []
    for case in _cases():
        canvas = PixelMemoryCanvas()
        canvas.project_concepts(case.concepts, case.relationships)
        canvas.activate_query(case.query)
        readback = canvas.readback()
        graph_only = _graph_only_retrieval(case.query, case.concepts)
        dominant_ids = [str(item.get("concept_id", "")) for item in readback.dominant_concepts]
        results.append(
            {
                "case": case.name,
                "query": case.query,
                "expected_concept_id": case.expected_concept_id,
                "graph_only_top_concept_id": graph_only["top_concept_id"],
                "pixel_top_concept_id": dominant_ids[0] if dominant_ids else "",
                "graph_plus_pixel_match": case.expected_concept_id in dominant_ids[:3],
                "graph_only": graph_only,
                "pixel_readback": readback.to_dict(),
            }
        )
    return {
        "probe_count": len(results),
        "passed": sum(1 for item in results if item["graph_plus_pixel_match"]),
        "results": results,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "canvas_projection_probes": run_pixel_memory_probes(),
                "decision_authority_probes": run_pixel_decision_probes(),
            },
            indent=2,
            sort_keys=True,
        )
    )
