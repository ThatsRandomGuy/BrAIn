import unittest
from pathlib import Path
import shutil
import tempfile

from brain_hybrid_prototype import BrainHybridPrototype
from brain_pixel_memory import PixelMemoryCanvas, PixelProjectionConcept, PixelRelationship, detect_reasoning_mode


class PixelMemoryLayerTests(unittest.TestCase):
    def test_detect_reasoning_mode_uses_pixel_threshold_contract(self) -> None:
        self.assertEqual(detect_reasoning_mode({"dominance_gap": 0.91, "dominant_regions": [{}, {}]}), "single_answer")
        self.assertEqual(detect_reasoning_mode({"dominance_gap": 0.05, "dominant_regions": [{}, {}]}), "multi_constraint")
        self.assertEqual(detect_reasoning_mode({"dominance_gap": 0.5, "dominant_regions": [{}], "temporal_next_concepts": [{}]}), "temporal")
        self.assertEqual(detect_reasoning_mode({"dominance_gap": 0.5, "dominant_regions": [{}], "temporal_next_concepts": []}), "hybrid")

    def test_projection_is_stable_and_hierarchy_is_nearby(self) -> None:
        concepts = [
            PixelProjectionConcept(
                concept_id="parent:fruit",
                label="fruit",
                terms=["fruit"],
                abstraction_level=3,
                importance=0.75,
                hierarchy_links=["child:apple"],
            ),
            PixelProjectionConcept(
                concept_id="child:apple",
                label="apple fruit snack",
                terms=["apple", "fruit", "snack"],
                abstraction_level=2,
                importance=0.6,
            ),
            PixelProjectionConcept(
                concept_id="contradiction:printer",
                label="printer is offline",
                terms=["printer", "offline"],
                contradiction_links=["child:apple"],
                importance=0.55,
            ),
        ]
        relationships = [
            PixelRelationship("parent:fruit", "child:apple", "hierarchy", 1.0),
            PixelRelationship("contradiction:printer", "child:apple", "contradicts", 1.0),
        ]
        first = PixelMemoryCanvas()
        second = PixelMemoryCanvas()
        first.project_concepts(concepts, relationships)
        second.project_concepts(concepts, relationships)

        self.assertEqual(
            {key: value.to_dict() for key, value in first.placements.items()},
            {key: value.to_dict() for key, value in second.placements.items()},
        )
        hierarchy_distance = abs(first.placements["parent:fruit"].x - first.placements["child:apple"].x) + abs(
            first.placements["parent:fruit"].y - first.placements["child:apple"].y
        )
        contradiction_distance = abs(first.placements["contradiction:printer"].x - first.placements["child:apple"].x) + abs(
            first.placements["contradiction:printer"].y - first.placements["child:apple"].y
        )
        self.assertLess(hierarchy_distance, contradiction_distance)

    def test_activation_readback_recovers_query_concepts(self) -> None:
        canvas = PixelMemoryCanvas()
        concepts = [
            PixelProjectionConcept(
                concept_id="fact:battery",
                label="battery stores charge for laptop",
                terms=["battery", "stores", "charge", "laptop"],
                importance=0.8,
                prediction_accuracy=0.7,
                uncertainty=0.2,
            ),
            PixelProjectionConcept(
                concept_id="fact:sky",
                label="sky appears blue during daytime",
                terms=["sky", "blue", "daytime"],
                importance=0.45,
            ),
        ]
        canvas.project_concepts(concepts)
        canvas.activate_query("What stores charge for my laptop?")
        readback = canvas.readback()
        dominant_ids = {item["concept_id"] for item in readback.dominant_concepts}
        self.assertIn("fact:battery", dominant_ids)
        self.assertGreater(readback.confidence, 0.0)
        self.assertTrue(readback.active_regions)
        self.assertTrue(readback.heatmap_lines)

    def test_memory_reconstruction_can_consult_pixel_readback_when_graph_is_weak(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="brain_pixel_memory_reconstruction_test_"))
        try:
            prototype = BrainHybridPrototype(
                state_dir=temp_dir,
                llm_provider="deterministic",
                bootstrap_reasoning=False,
            )
            prototype.process_user_turn("The lab door code is 2468.", inspect=True)
            result = prototype.process_user_turn("What did I ask earlier about the lab door code?", inspect=True)
            reconstructed = result.trace.merged_memory_state.get("reconstructed_context", {})
            self.assertTrue(reconstructed)
            evidence = reconstructed.get("evidence_breakdown", {})
            self.assertTrue(evidence.get("pixel_memory_enabled"))
            self.assertIn("pixel_memory_consulted", evidence)
            pixel_payload = evidence.get("pixel_readback", {})
            if evidence.get("pixel_memory_consulted"):
                self.assertTrue(pixel_payload.get("dominant_concepts"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pixel_retrieval_can_be_disabled(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="brain_pixel_memory_disabled_test_"))
        try:
            prototype = BrainHybridPrototype(
                state_dir=temp_dir,
                llm_provider="deterministic",
                bootstrap_reasoning=False,
                pixel_memory_enabled=False,
            )
            prototype.process_user_turn("The spare key is in the blue bowl.", inspect=True)
            result = prototype.process_user_turn("What did I ask earlier about the spare key?", inspect=True)
            reconstructed = result.trace.merged_memory_state.get("reconstructed_context", {})
            evidence = reconstructed.get("evidence_breakdown", {}) if isinstance(reconstructed, dict) else {}
            self.assertFalse(evidence.get("pixel_memory_enabled"))
            self.assertFalse(evidence.get("pixel_memory_consulted"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pixel_can_override_overwrite_conflict(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="brain_pixel_overwrite_override_test_"))
        try:
            prototype = BrainHybridPrototype(
                state_dir=temp_dir,
                llm_provider="deterministic",
                bootstrap_reasoning=False,
                pixel_memory_enabled=True,
            )
            prototype.process_user_turn("I keep frozen berries in the pantry.", inspect=True)
            prototype.process_user_turn("Actually, I moved the frozen berries to the freezer.", inspect=True)
            reconstructed = prototype.reconstruct_memory_context(
                user_input_text="pantry",
                retrieved=[],
                recalled_turns=[],
                merged_memory_state={},
            ).to_dict()
            evidence = reconstructed.get("evidence_breakdown", {})
            partials = reconstructed.get("partial_matches", [])
            self.assertEqual(evidence.get("decision_source"), "pixel")
            self.assertGreater(float(evidence.get("pixel_confidence", 0.0)), float(evidence.get("graph_confidence", 0.0)))
            self.assertEqual(partials[0].get("source_type"), "pixel_readback")
            self.assertIn("freezer", str(partials[0].get("text", "")).lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pixel_can_override_contradiction_resolution(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="brain_pixel_contradiction_override_test_"))
        try:
            prototype = BrainHybridPrototype(
                state_dir=temp_dir,
                llm_provider="deterministic",
                bootstrap_reasoning=False,
                pixel_memory_enabled=True,
            )
            prototype.process_user_turn("The status light is red.", inspect=True)
            prototype.process_user_turn("Correction: the status light is green.", inspect=True)
            reconstructed = prototype.reconstruct_memory_context(
                user_input_text="red status light",
                retrieved=[],
                recalled_turns=[],
                merged_memory_state={},
            ).to_dict()
            evidence = reconstructed.get("evidence_breakdown", {})
            pixel_reasoning = evidence.get("pixel_reasoning", {})
            partials = reconstructed.get("partial_matches", [])
            self.assertEqual(evidence.get("decision_source"), "pixel")
            self.assertIn(pixel_reasoning.get("reasoning_mode"), {"single_answer", "multi_constraint", "temporal", "hybrid"})
            self.assertEqual(partials[0].get("source_type"), "pixel_readback")
            self.assertIn("green", str(partials[0].get("text", "")).lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pixel_improves_temporal_next_step_prediction(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="brain_pixel_temporal_next_test_"))
        try:
            prototype = BrainHybridPrototype(
                state_dir=temp_dir,
                llm_provider="deterministic",
                bootstrap_reasoning=False,
                pixel_memory_enabled=True,
            )
            prototype.process_user_turn("First reproduce the bug.", inspect=True)
            prototype.process_user_turn("Then update the tests.", inspect=True)
            prototype.process_user_turn("After tests pass, request review.", inspect=True)
            reconstructed = prototype.reconstruct_memory_context(
                user_input_text="What should happen after reproducing the bug?",
                retrieved=[],
                recalled_turns=[],
                merged_memory_state={},
            ).to_dict()
            evidence = reconstructed.get("evidence_breakdown", {})
            pixel_readback = evidence.get("pixel_readback", {})
            pixel_reasoning = evidence.get("pixel_reasoning", {})
            temporal_next = pixel_readback.get("temporal_next_concepts", []) if isinstance(pixel_readback, dict) else []
            partials = reconstructed.get("partial_matches", [])
            self.assertEqual(evidence.get("decision_source"), "pixel")
            self.assertEqual(pixel_reasoning.get("reasoning_mode"), "temporal")
            self.assertEqual(pixel_reasoning.get("selected_output_type"), "next_step_concepts")
            self.assertTrue(temporal_next)
            self.assertIn("update the tests", str(temporal_next[0].get("label", "")).lower())
            self.assertIn("update the tests", str(partials[0].get("text", "")).lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pixel_improves_partial_reconstruction_edge_case(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="brain_pixel_partial_edge_test_"))
        try:
            prototype = BrainHybridPrototype(
                state_dir=temp_dir,
                llm_provider="deterministic",
                bootstrap_reasoning=False,
                pixel_memory_enabled=True,
            )
            prototype.process_user_turn("For the demo kit, the red cable belongs with the sensor board.", inspect=True)
            prototype.process_user_turn("The calibration note says sensor board first, then cable check.", inspect=True)
            reconstructed = prototype.reconstruct_memory_context(
                user_input_text="red wiring",
                retrieved=[],
                recalled_turns=[],
                merged_memory_state={},
            ).to_dict()
            evidence = reconstructed.get("evidence_breakdown", {})
            partials = reconstructed.get("partial_matches", [])
            self.assertIn(evidence.get("decision_source"), {"pixel", "hybrid"})
            self.assertTrue(evidence.get("pixel_memory_consulted"))
            self.assertEqual(partials[0].get("source_type"), "pixel_readback")
            self.assertIn("red cable", str(partials[0].get("text", "")).lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pixel_preserves_multi_constraint_regions(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="brain_pixel_multi_constraint_test_"))
        try:
            prototype = BrainHybridPrototype(
                state_dir=temp_dir,
                llm_provider="deterministic",
                bootstrap_reasoning=False,
                pixel_memory_enabled=True,
            )
            prototype.process_user_turn("Packing constraint: bring one backpack.", inspect=True)
            prototype.process_user_turn("Packing constraint: keep weight under eight kilograms.", inspect=True)
            prototype.process_user_turn("Packing constraint: laptop must stay in the easy access pocket.", inspect=True)
            reconstructed = prototype.reconstruct_memory_context(
                user_input_text="What packing constraints did I give?",
                retrieved=[],
                recalled_turns=[],
                merged_memory_state={},
            ).to_dict()
            evidence = reconstructed.get("evidence_breakdown", {})
            pixel_readback = evidence.get("pixel_readback", {})
            pixel_reasoning = evidence.get("pixel_reasoning", {})
            partials = reconstructed.get("partial_matches", [])
            self.assertTrue(evidence.get("multi_constraint_result"))
            self.assertEqual(pixel_reasoning.get("reasoning_mode"), "multi_constraint")
            self.assertEqual(pixel_reasoning.get("selected_output_type"), "constraint_list")
            self.assertGreaterEqual(len(pixel_reasoning.get("constraints", [])), 2)
            self.assertTrue(pixel_readback.get("multi_constraint_concepts"))
            self.assertEqual(partials[0].get("source_type"), "pixel_readback")
            text = str(partials[0].get("text", "")).lower()
            self.assertIn("backpack", text)
            self.assertIn("laptop", text)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
