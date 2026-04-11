from __future__ import annotations

import shutil
import tempfile
import unittest
from math import dist
from pathlib import Path

from brain_codec import BrainCodec, BrainMemory
from brain_hybrid_prototype import BrainHybridPrototype
from llm_interface import render_llm_payload_v2_answer
from brain_reasoning_graph import (
    QueryIntent,
    ReasoningGraph,
    ReasoningGraphCandidate,
    ReasoningGraphCluster,
    ReasoningGraphClusterRelationship,
    ReasoningGraphScoringConfig,
    ReasoningGraphEvaluator,
    ReasoningGraphNode,
    ReasoningSpatialConcept,
    ReasoningTemporalTransition,
    RelationshipType,
    detect_query_intent,
)


class ReasoningGraphLayerTests(unittest.TestCase):
    def test_reasoning_graph_promotes_supported_explanatory_entry(self) -> None:
        evaluator = ReasoningGraphEvaluator()
        graph = evaluator.build_graph(
            query_text="Why is the sky blue?",
            candidates=[
                ReasoningGraphCandidate(
                    entry_id=1,
                    label="entry:1",
                    base_score=5.2,
                    topic_key="sky_color:daytime",
                    answer_text="The sky is blue during the day.",
                    answer_signature="the sky is blue during the day",
                    terms=["sky", "blue", "day"],
                    facts=["The sky is blue during the day."],
                ),
                ReasoningGraphCandidate(
                    entry_id=2,
                    label="entry:2",
                    base_score=5.0,
                    topic_key="sky_color:daytime",
                    answer_text="The sky is blue because shorter blue wavelengths scatter more strongly in Earth's atmosphere.",
                    answer_signature="the sky is blue because shorter blue wavelengths scatter more strongly in earth s atmosphere",
                    terms=["sky", "blue", "wavelengths", "scatter", "atmosphere"],
                    facts=["The sky is blue because shorter blue wavelengths scatter more strongly in Earth's atmosphere."],
                ),
                ReasoningGraphCandidate(
                    entry_id=3,
                    label="entry:3",
                    base_score=4.8,
                    topic_key="sky_scattering:reason",
                    answer_text="Blue light scatters more strongly in Earth's atmosphere because shorter wavelengths spread more widely.",
                    answer_signature="blue light scatters more strongly in earth s atmosphere because shorter wavelengths spread more widely",
                    terms=["blue", "light", "scatter", "atmosphere", "wavelengths"],
                    facts=["Blue light scatters more strongly in Earth's atmosphere because shorter wavelengths spread more widely."],
                ),
            ]
        )
        node_lookup = graph.node_map()

        self.assertEqual(graph.query_intent, QueryIntent.EXPLANATION)
        self.assertIn(
            RelationshipType.CAUSES,
            {edge.relationship_type for edge in graph.edges},
        )
        self.assertGreater(node_lookup[2].final_score, node_lookup[1].final_score)
        self.assertGreater(node_lookup[2].support_bonus, 0.0)
        self.assertIsNotNone(graph.spatial_grid)
        self.assertTrue(graph.spatial_grid.top_regions if graph.spatial_grid is not None else [])
        spatial_grid = graph.spatial_grid
        assert spatial_grid is not None
        mapping_lookup = {item.entry_id: item for item in spatial_grid.node_mappings}
        self.assertTrue(spatial_grid.region_dominance_log)
        self.assertTrue(spatial_grid.temporal_update_log)
        self.assertTrue(spatial_grid.concept_creation_log)
        support_distance = dist(
            (mapping_lookup[2].center_x, mapping_lookup[2].center_y),
            (mapping_lookup[3].center_x, mapping_lookup[3].center_y),
        )
        causal_distance = dist(
            (mapping_lookup[2].center_x, mapping_lookup[2].center_y),
            (mapping_lookup[1].center_x, mapping_lookup[1].center_y),
        )
        self.assertLessEqual(support_distance, causal_distance + 1.5)

    def test_spatial_grid_tracks_mapping_and_inhibition(self) -> None:
        evaluator = ReasoningGraphEvaluator()
        graph = evaluator.build_graph(
            query_text="What color is the sky during the day?",
            candidates=[
                ReasoningGraphCandidate(
                    entry_id=1,
                    label="entry:1",
                    base_score=5.2,
                    topic_key="sky_color:daytime",
                    answer_text="The sky is blue during the day.",
                    answer_signature="the sky is blue during the day",
                    terms=["sky", "blue", "day"],
                    facts=["The sky is blue during the day."],
                ),
                ReasoningGraphCandidate(
                    entry_id=2,
                    label="entry:2",
                    base_score=5.1,
                    topic_key="sky_color:daytime",
                    answer_text="The sky is green during the day.",
                    answer_signature="the sky is green during the day",
                    terms=["sky", "green", "day"],
                    facts=["The sky is green during the day."],
                ),
                ReasoningGraphCandidate(
                    entry_id=3,
                    label="entry:3",
                    base_score=5.0,
                    topic_key="sky_color:daytime",
                    answer_text="Daytime sky color is blue in normal atmospheric conditions.",
                    answer_signature="daytime sky color is blue in normal atmospheric conditions",
                    terms=["daytime", "sky", "color", "blue", "atmospheric"],
                    facts=["Daytime sky color is blue in normal atmospheric conditions."],
                ),
            ],
        )

        self.assertIsNotNone(graph.spatial_grid)
        spatial_grid = graph.spatial_grid
        assert spatial_grid is not None
        self.assertEqual(spatial_grid.width, 32)
        self.assertEqual(spatial_grid.height, 32)
        self.assertEqual(len(spatial_grid.node_mappings), 3)
        self.assertTrue(spatial_grid.inhibition_log)
        self.assertTrue(spatial_grid.region_competition_log)
        self.assertTrue(spatial_grid.region_dominance_log)
        self.assertTrue(spatial_grid.temporal_update_log)
        self.assertTrue(spatial_grid.dominance_evolution_log)
        self.assertTrue(spatial_grid.concept_creation_log)
        self.assertGreaterEqual(len(spatial_grid.top_regions), 2)
        node_mapping_lookup = {item.entry_id: item for item in spatial_grid.node_mappings}
        self.assertGreater(node_mapping_lookup[1].final_region_activation, node_mapping_lookup[2].final_region_activation)
        self.assertGreater(node_mapping_lookup[3].final_region_activation, node_mapping_lookup[2].final_region_activation)
        contradiction_distance = dist(
            (node_mapping_lookup[1].center_x, node_mapping_lookup[1].center_y),
            (node_mapping_lookup[2].center_x, node_mapping_lookup[2].center_y),
        )
        self.assertGreaterEqual(contradiction_distance, 7.0)
        self.assertTrue(any(1 in region.source_nodes for region in spatial_grid.top_regions))
        self.assertIsNotNone(spatial_grid.top_regions[0].dominant_node_id)

    def test_spatial_concept_memory_biases_future_queries(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="brain_spatial_concept_memory_test_"))
        try:
            memory_path = temp_dir / "spatial_concept_memory.json"
            evaluator = ReasoningGraphEvaluator(concept_memory_path=memory_path)
            candidates = [
                ReasoningGraphCandidate(
                    entry_id=1,
                    label="entry:1",
                    base_score=5.2,
                    topic_key="sky_color:daytime",
                    answer_text="The sky is blue during the day.",
                    answer_signature="the sky is blue during the day",
                    terms=["sky", "blue", "day"],
                    facts=["The sky is blue during the day."],
                ),
                ReasoningGraphCandidate(
                    entry_id=2,
                    label="entry:2",
                    base_score=5.0,
                    topic_key="sky_color:daytime",
                    answer_text="The sky is blue because shorter blue wavelengths scatter more strongly in Earth's atmosphere.",
                    answer_signature="the sky is blue because shorter blue wavelengths scatter more strongly in earth s atmosphere",
                    terms=["sky", "blue", "wavelengths", "scatter", "atmosphere"],
                    facts=["The sky is blue because shorter blue wavelengths scatter more strongly in Earth's atmosphere."],
                ),
                ReasoningGraphCandidate(
                    entry_id=3,
                    label="entry:3",
                    base_score=4.8,
                    topic_key="sky_scattering:reason",
                    answer_text="Blue light scatters more strongly in Earth's atmosphere because shorter wavelengths spread more widely.",
                    answer_signature="blue light scatters more strongly in earth s atmosphere because shorter wavelengths spread more widely",
                    terms=["blue", "light", "scatter", "atmosphere", "wavelengths"],
                    facts=["Blue light scatters more strongly in Earth's atmosphere because shorter wavelengths spread more widely."],
                ),
            ]

            first_graph = evaluator.build_graph(query_text="Why is the sky blue?", candidates=candidates)
            self.assertIsNotNone(first_graph.spatial_grid)
            assert first_graph.spatial_grid is not None
            self.assertTrue(first_graph.spatial_grid.concept_creation_log)
            self.assertFalse(first_graph.spatial_grid.concept_reuse_log)
            self.assertTrue(memory_path.exists())

            reloaded_evaluator = ReasoningGraphEvaluator(concept_memory_path=memory_path)
            second_graph = reloaded_evaluator.build_graph(query_text="Why is the sky blue?", candidates=candidates)
            self.assertIsNotNone(second_graph.spatial_grid)
            assert second_graph.spatial_grid is not None
            self.assertTrue(second_graph.spatial_grid.concept_reuse_log)
            self.assertTrue(second_graph.spatial_grid.concept_activation_log)
            self.assertTrue(second_graph.spatial_grid.concept_pre_activation_log)
            self.assertTrue(second_graph.spatial_grid.concept_importance_log)
            self.assertTrue(second_graph.spatial_grid.concept_evolution_log)
            self.assertTrue(second_graph.spatial_grid.activated_concepts)
            self.assertTrue(second_graph.spatial_grid.concept_memory_snapshot)
            self.assertTrue(
                any(concept.reuse_count >= 1 for concept in second_graph.spatial_grid.concept_memory_snapshot)
            )
            self.assertTrue(
                any(concept.importance > 0.0 for concept in second_graph.spatial_grid.concept_memory_snapshot)
            )
            self.assertTrue(
                any(node.concept_predictive_bias > 0.0 for node in second_graph.nodes)
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_spatial_concepts_compete_when_reused(self) -> None:
        evaluator = ReasoningGraphEvaluator()
        candidates = [
            ReasoningGraphCandidate(
                entry_id=1,
                label="entry:1",
                base_score=5.2,
                topic_key="sky_color:daytime",
                answer_text="The sky is blue during the day.",
                answer_signature="the sky is blue during the day",
                terms=["sky", "blue", "day"],
                facts=["The sky is blue during the day."],
            ),
            ReasoningGraphCandidate(
                entry_id=2,
                label="entry:2",
                base_score=5.1,
                topic_key="sky_color:daytime",
                answer_text="The sky is green during the day.",
                answer_signature="the sky is green during the day",
                terms=["sky", "green", "day"],
                facts=["The sky is green during the day."],
            ),
            ReasoningGraphCandidate(
                entry_id=3,
                label="entry:3",
                base_score=5.0,
                topic_key="sky_color:daytime",
                answer_text="Daytime sky color is blue in normal atmospheric conditions.",
                answer_signature="daytime sky color is blue in normal atmospheric conditions",
                terms=["daytime", "sky", "color", "blue", "atmospheric"],
                facts=["Daytime sky color is blue in normal atmospheric conditions."],
            ),
        ]

        first_graph = evaluator.build_graph(query_text="What color is the sky during the day?", candidates=candidates)
        self.assertIsNotNone(first_graph.spatial_grid)
        assert first_graph.spatial_grid is not None
        self.assertGreaterEqual(len(first_graph.spatial_grid.concept_memory_snapshot), 2)

        second_graph = evaluator.build_graph(query_text="What color is the sky during the day?", candidates=candidates)
        self.assertIsNotNone(second_graph.spatial_grid)
        assert second_graph.spatial_grid is not None
        self.assertGreaterEqual(len(second_graph.spatial_grid.activated_concepts), 2)
        self.assertTrue(second_graph.spatial_grid.concept_competition_log)
        self.assertTrue(second_graph.spatial_grid.concept_evolution_log)
        self.assertTrue(second_graph.spatial_grid.concept_hierarchy_log)
        self.assertTrue(second_graph.spatial_grid.concept_cluster_log)
        self.assertTrue(second_graph.spatial_grid.concept_clusters)
        self.assertTrue(
            any(concept.is_meta_concept for concept in second_graph.spatial_grid.concept_memory_snapshot)
        )

        third_graph = evaluator.build_graph(query_text="What color is the sky during the day?", candidates=candidates)
        self.assertIsNotNone(third_graph.spatial_grid)
        assert third_graph.spatial_grid is not None
        self.assertTrue(third_graph.spatial_grid.anticipation_log)
        self.assertTrue(third_graph.spatial_grid.meta_concept_influence_log)
        self.assertTrue(third_graph.spatial_grid.prediction_accuracy_log)
        self.assertTrue(third_graph.spatial_grid.concept_importance_evolution_log)
        self.assertTrue(third_graph.spatial_grid.anticipated_concepts)
        self.assertTrue(
            any(concept.prediction_total_count >= 1 for concept in third_graph.spatial_grid.concept_memory_snapshot)
        )
        self.assertTrue(
            any(concept.prediction_accuracy > 0.0 for concept in third_graph.spatial_grid.concept_memory_snapshot)
        )

    def test_query_intent_detection(self) -> None:
        self.assertEqual(detect_query_intent("What color is the sky during the day?"), QueryIntent.FACT)
        self.assertEqual(detect_query_intent("Why is the sky blue?"), QueryIntent.EXPLANATION)
        self.assertEqual(detect_query_intent("How does light scattering work?"), QueryIntent.PROCESS)

    def test_concept_clusters_compete_when_activated(self) -> None:
        evaluator = ReasoningGraphEvaluator(
            scoring=ReasoningGraphScoringConfig(
                concept_cluster_similarity_threshold=1.1,
                concept_cross_domain_overlap_threshold=1.1,
                concept_generalization_overlap_threshold=1.1,
            )
        )
        evaluator.spatial_concept_memory = [
            ReasoningSpatialConcept(
                concept_id=1,
                centroid_x=15.0,
                centroid_y=15.0,
                activation_value=4.0,
                associated_nodes=[1],
                associated_clusters=[],
                relationship_types=["SUPPORTS"],
                associated_terms=["sky", "blue", "answer"],
                average_dominance=0.95,
                stability_score=0.92,
            ),
            ReasoningSpatialConcept(
                concept_id=2,
                centroid_x=20.0,
                centroid_y=15.0,
                activation_value=1.8,
                associated_nodes=[3],
                associated_clusters=[],
                relationship_types=["EXTENDS"],
                associated_terms=["grass", "green", "answer"],
                average_dominance=0.45,
                stability_score=0.5,
            ),
        ]
        graph = evaluator.build_graph(
            query_text="What color are the sky and grass?",
            candidates=[
                ReasoningGraphCandidate(
                    entry_id=1,
                    label="entry:1",
                    base_score=5.2,
                    topic_key="sky_color",
                    answer_text="The sky is blue.",
                    answer_signature="the sky is blue",
                    terms=["sky", "blue"],
                    facts=["The sky is blue."],
                ),
                ReasoningGraphCandidate(
                    entry_id=3,
                    label="entry:3",
                    base_score=4.6,
                    topic_key="grass_color",
                    answer_text="Grass is green.",
                    answer_signature="grass is green",
                    terms=["grass", "green"],
                    facts=["Grass is green."],
                ),
            ],
        )
        self.assertIsNotNone(graph.spatial_grid)
        assert graph.spatial_grid is not None
        self.assertTrue(graph.spatial_grid.concept_cluster_competition_log)
        self.assertTrue(
            any(cluster.competition_penalty > 0.0 for cluster in graph.spatial_grid.concept_clusters)
        )

    def test_concept_stability_exploration_and_generalization_controls(self) -> None:
        evaluator = ReasoningGraphEvaluator(
            scoring=ReasoningGraphScoringConfig(
                concept_exploration_rate=1.0,
                concept_memory_merge_distance=1.0,
            )
        )
        evaluator.spatial_concept_memory = [
            ReasoningSpatialConcept(
                concept_id=1,
                centroid_x=14.0,
                centroid_y=14.0,
                activation_value=3.4,
                associated_nodes=[1],
                associated_clusters=[],
                relationship_types=["SUPPORTS"],
                associated_terms=["sky", "blue", "color", "answer"],
                average_dominance=1.0,
                stability_score=1.0,
                importance_memory_adjustment=0.35,
            ),
            ReasoningSpatialConcept(
                concept_id=2,
                centroid_x=19.0,
                centroid_y=14.0,
                activation_value=1.4,
                associated_nodes=[3],
                associated_clusters=[],
                relationship_types=["EXTENDS"],
                associated_terms=["grass", "green", "color", "answer"],
                average_dominance=0.3,
                stability_score=0.35,
            ),
        ]
        evaluator.temporal_transition_memory = {
            (2, 3): ReasoningTemporalTransition(
                source_concept_id=2,
                target_concept_id=3,
                frequency=3,
                confidence=0.6,
                success_count=2,
                failure_count=1,
                reliability=0.58,
                simulation_total_count=2,
                simulation_correct_count=1,
                simulation_accuracy=0.5,
            ),
            (3, 1): ReasoningTemporalTransition(
                source_concept_id=3,
                target_concept_id=1,
                frequency=2,
                confidence=0.4,
                success_count=1,
                failure_count=1,
                reliability=0.5,
                simulation_total_count=2,
                simulation_correct_count=1,
                simulation_accuracy=0.5,
            ),
        }

        graph = evaluator.build_graph(
            query_text="What color are the sky and grass?",
            candidates=[
                ReasoningGraphCandidate(
                    entry_id=1,
                    label="entry:1",
                    base_score=5.2,
                    topic_key="sky_color",
                    answer_text="The sky is blue.",
                    answer_signature="the sky is blue",
                    terms=["sky", "blue"],
                    facts=["The sky is blue."],
                ),
                ReasoningGraphCandidate(
                    entry_id=3,
                    label="entry:3",
                    base_score=4.6,
                    topic_key="grass_color",
                    answer_text="Grass is green.",
                    answer_signature="grass is green",
                    terms=["grass", "green"],
                    facts=["Grass is green."],
                ),
            ],
        )

        self.assertIsNotNone(graph.spatial_grid)
        assert graph.spatial_grid is not None
        self.assertTrue(graph.spatial_grid.concept_exploration_log)
        self.assertTrue(graph.spatial_grid.concept_uncertainty_log)
        self.assertTrue(graph.spatial_grid.concept_cross_domain_link_log)
        self.assertTrue(graph.spatial_grid.concept_generalization_log)
        self.assertTrue(graph.spatial_grid.concept_stability_log)
        self.assertTrue(graph.spatial_grid.concept_abstraction_log)
        self.assertTrue(graph.spatial_grid.temporal_sequence_log)
        self.assertTrue(graph.spatial_grid.goal_selection_log)
        self.assertTrue(graph.spatial_grid.goal_plan_log)
        self.assertTrue(graph.spatial_grid.goal_feedback_log)
        self.assertIsNotNone(graph.spatial_grid.active_goal)
        self.assertTrue(graph.spatial_grid.goal_plans)
        self.assertTrue(
            any(
                plan.simulated_success_probability >= 0.0
                and plan.simulated_uncertainty >= 0.0
                and plan.predicted_outcome in {"success", "miss"}
                for plan in graph.spatial_grid.goal_plans
            )
        )
        self.assertTrue(any("simulated_success" in line for line in graph.spatial_grid.goal_plan_log))
        self.assertTrue(any(len(plan.path) >= 3 for plan in graph.spatial_grid.goal_plans))
        self.assertTrue(any("prediction_error" in line for line in graph.spatial_grid.goal_feedback_log))
        self.assertTrue(any("simulation error attribution" in line for line in graph.spatial_grid.goal_feedback_log))
        self.assertGreaterEqual(evaluator.simulation_model_state.total_predictions, 1)
        self.assertTrue(
            any(concept.is_generalized_concept for concept in graph.spatial_grid.concept_memory_snapshot)
        )
        self.assertTrue(
            any(concept.abstraction_level == 3 for concept in graph.spatial_grid.concept_memory_snapshot)
        )
        self.assertTrue(graph.spatial_grid.temporal_transitions)
        self.assertTrue(
            all(
                concept.importance <= evaluator.scoring.concept_importance_cap
                for concept in graph.spatial_grid.concept_memory_snapshot
            )
        )

    def test_fact_query_downweights_extension_and_tracks_clusters(self) -> None:
        evaluator = ReasoningGraphEvaluator()
        graph = evaluator.build_graph(
            query_text="What color is the sky during the day?",
            candidates=[
                ReasoningGraphCandidate(
                    entry_id=1,
                    label="entry:1",
                    base_score=5.2,
                    topic_key="sky_color:daytime",
                    answer_text="The sky is blue during the day.",
                    answer_signature="the sky is blue during the day",
                    terms=["sky", "blue", "day"],
                    facts=["The sky is blue during the day."],
                ),
                ReasoningGraphCandidate(
                    entry_id=2,
                    label="entry:2",
                    base_score=5.1,
                    topic_key="sky_color:daytime",
                    answer_text="Daytime sky color is blue in normal atmospheric conditions and appears lighter near the horizon on clear days.",
                    answer_signature="daytime sky color is blue in normal atmospheric conditions and appears lighter near the horizon on clear days",
                    terms=["daytime", "sky", "color", "blue", "atmospheric", "conditions", "horizon"],
                    facts=["Daytime sky color is blue in normal atmospheric conditions and appears lighter near the horizon on clear days."],
                ),
                ReasoningGraphCandidate(
                    entry_id=3,
                    label="entry:3",
                    base_score=5.0,
                    topic_key="sky_color:daytime",
                    answer_text="The daytime sky is blue.",
                    answer_signature="the daytime sky is blue",
                    terms=["daytime", "sky", "blue"],
                    facts=["The daytime sky is blue."],
                ),
            ],
        )
        node_lookup = graph.node_map()

        self.assertEqual(graph.query_intent, QueryIntent.FACT)
        self.assertEqual(len(graph.clusters), 2)
        self.assertGreater(node_lookup[2].raw_extension_score, 0.5)
        self.assertLess(node_lookup[2].extension_bonus, node_lookup[2].raw_extension_score * 0.05)
        self.assertGreater(node_lookup[1].cluster_score, 0.0)
        self.assertEqual(node_lookup[1].cluster_id, node_lookup[3].cluster_id)
        self.assertNotEqual(node_lookup[1].cluster_id, node_lookup[2].cluster_id)
        self.assertGreater(node_lookup[1].dense_cluster_bonus, 0.0)

    def test_explanation_selection_prefers_best_cluster_then_best_node(self) -> None:
        evaluator = ReasoningGraphEvaluator()
        graph = ReasoningGraph(
            query_intent=QueryIntent.EXPLANATION,
            nodes=[
                ReasoningGraphNode(entry_id=1, label="entry:1", base_score=9.8, cluster_id=2, cluster_score=9.8),
                ReasoningGraphNode(entry_id=2, label="entry:2", base_score=7.0, cluster_id=1, cluster_score=14.2),
                ReasoningGraphNode(entry_id=3, label="entry:3", base_score=6.8, cluster_id=1, cluster_score=14.2),
            ],
            edges=[],
            clusters=[
                ReasoningGraphCluster(cluster_id=1, nodes=[2, 3], cluster_score=14.2, support_edges=2, contradiction_edges=0, density=1.0, cluster_final_score=14.8),
                ReasoningGraphCluster(cluster_id=2, nodes=[1], cluster_score=9.8, support_edges=0, contradiction_edges=0, density=0.0, cluster_final_score=9.8),
            ],
            cluster_relationships=[
                ReasoningGraphClusterRelationship(
                    source_cluster_id=1,
                    target_cluster_id=2,
                    relationship_type=RelationshipType.SUPPORTS,
                    strength=0.8,
                    node_edge_count=1,
                )
            ],
        )

        selection = evaluator.select_nodes(graph)

        self.assertEqual(selection.selection_mode, "cluster_graph_first")
        self.assertEqual(selection.selected_cluster_id, 1)
        self.assertEqual(selection.selected_node_entry_id, 2)
        self.assertEqual(selection.ranked_entry_ids[:3], [2, 3, 1])
        self.assertEqual(selection.selected_reasoning_path, [2])
        self.assertGreaterEqual(selection.selected_path_score, 0.0)
        self.assertGreaterEqual(selection.selected_path_activation, 0.0)
        self.assertTrue(selection.candidate_paths)

    def test_cache_ranking_uses_graph_final_score(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="brain_reasoning_graph_test_"))
        try:
            prototype = BrainHybridPrototype(
                state_dir=temp_dir,
                bootstrap_reasoning=False,
            )
            self._add_cache_entry(
                prototype,
                query_text="Why is the sky blue?",
                answer_text="The sky is blue during the day.",
                topic_key="sky_color:daytime",
                confidence_estimate=0.78,
                reuse_count=2,
                successful_uses=1,
            )
            self._add_cache_entry(
                prototype,
                query_text="Why is the sky blue?",
                answer_text="The sky is blue because shorter blue wavelengths scatter more strongly in Earth's atmosphere.",
                topic_key="sky_color:daytime",
                confidence_estimate=0.74,
                reuse_count=1,
                successful_uses=1,
            )
            self._add_cache_entry(
                prototype,
                query_text="Why is the sky blue?",
                answer_text="Blue light scatters more strongly in Earth's atmosphere because shorter wavelengths spread more widely.",
                topic_key="sky_scattering:reason",
                confidence_estimate=0.7,
                reuse_count=0,
                successful_uses=0,
            )

            selected, ranking = prototype._cache_rank_candidates(
                user_input_text="Why is the sky blue?",
                top_k=5,
                update_reuse=False,
            )

            self.assertTrue(selected)
            self.assertEqual(selected[0].entry_id, ranking.get("selected_entry_id"))
            self.assertTrue(ranking.get("reasoning_graph_nodes"))
            self.assertTrue(ranking.get("reasoning_graph_edges"))
            self.assertIn("base_score", ranking["ranking_scores"][0])
            self.assertIn("final_score", ranking["ranking_scores"][0])
            self.assertIn("support_bonus", ranking["ranking_scores"][0])
            self.assertIn("contradiction_penalty", ranking["ranking_scores"][0])
            self.assertIn("extension_bonus", ranking["ranking_scores"][0])
            self.assertIn("cluster_id", ranking["ranking_scores"][0])
            self.assertIn("cluster_score", ranking["ranking_scores"][0])
            self.assertIn("cluster_influence_bonus", ranking["ranking_scores"][0])
            self.assertIn("activation_value", ranking["ranking_scores"][0])
            self.assertIn("base_activation_value", ranking["ranking_scores"][0])
            self.assertIn("propagated_activation_bonus", ranking["ranking_scores"][0])
            self.assertIn("inhibition_penalty", ranking["ranking_scores"][0])
            self.assertEqual(ranking.get("query_intent"), QueryIntent.EXPLANATION.value)
            self.assertEqual(ranking.get("selection_mode"), "cluster_graph_first")
            self.assertIsNotNone(ranking.get("selected_cluster_id"))
            self.assertEqual(ranking.get("selected_node_entry_id"), selected[0].entry_id)
            self.assertTrue(ranking.get("reasoning_graph_clusters"))
            self.assertIn("cluster_final_score", ranking["reasoning_graph_clusters"][0])
            self.assertIn("activation_value", ranking["reasoning_graph_clusters"][0])
            self.assertIn("base_activation_value", ranking["reasoning_graph_clusters"][0])
            self.assertIn("reasoning_graph_cluster_relationships", ranking)
            self.assertIn("reasoning_spatial_grid", ranking)
            self.assertTrue(ranking.get("selected_reasoning_path"))
            self.assertGreater(float(ranking.get("selected_path_score", 0.0) or 0.0), 0.0)
            self.assertGreater(float(ranking.get("selected_path_final_score", 0.0) or 0.0), 0.0)
            self.assertIn("activation", str(ranking.get("selected_path_breakdown", "")).lower())
            self.assertTrue(ranking.get("candidate_reasoning_paths"))
            self.assertIn("relative_strength", ranking["candidate_reasoning_paths"][0])
            self.assertTrue(ranking["reasoning_spatial_grid"].get("node_mappings"))
            self.assertIsNotNone(ranking.get("selected_spatial_region_id"))
            self.assertTrue(ranking.get("selected_spatial_nodes"))
            self.assertIn(selected[0].entry_id, ranking.get("selected_spatial_nodes", []))
            self.assertIn("placement_map_lines", ranking["reasoning_spatial_grid"])
            self.assertTrue(ranking["reasoning_spatial_grid"].get("region_dominance_log"))
            self.assertTrue(ranking["reasoning_spatial_grid"].get("temporal_update_log"))
            self.assertTrue(ranking["reasoning_spatial_grid"].get("concept_creation_log"))
            self.assertIn(
                selected[0].source_text,
                {
                    "The sky is blue during the day.",
                    "The sky is blue because shorter blue wavelengths scatter more strongly in Earth's atmosphere.",
                },
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_fact_query_prefers_direct_supported_answer(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="brain_reasoning_graph_fact_test_"))
        try:
            prototype = BrainHybridPrototype(
                state_dir=temp_dir,
                bootstrap_reasoning=False,
            )
            self._add_cache_entry(
                prototype,
                query_text="What color is the sky during the day?",
                answer_text="The sky is blue during the day.",
                topic_key="sky_color:daytime",
                confidence_estimate=0.80,
                reuse_count=2,
                successful_uses=2,
            )
            self._add_cache_entry(
                prototype,
                query_text="What color is the sky during the day?",
                answer_text="The sky is green during the day.",
                topic_key="sky_color:daytime",
                confidence_estimate=0.82,
                reuse_count=2,
                successful_uses=2,
            )
            self._add_cache_entry(
                prototype,
                query_text="What color is the sky during the day?",
                answer_text="Daytime sky color is blue in normal atmospheric conditions.",
                topic_key="sky_color:daytime",
                confidence_estimate=0.76,
                reuse_count=1,
                successful_uses=1,
            )

            selected, ranking = prototype._cache_rank_candidates(
                user_input_text="What color is the sky during the day?",
                top_k=5,
                update_reuse=False,
            )

            self.assertTrue(selected)
            self.assertEqual(ranking.get("query_intent"), QueryIntent.FACT.value)
            self.assertEqual(selected[0].source_text, "The sky is blue during the day.")
            top_score = ranking["ranking_scores"][0]
            self.assertGreater(float(top_score.get("directness_bonus", 0.0) or 0.0), 0.0)
            self.assertGreater(float(top_score.get("contradiction_penalty", 0.0) or 0.0), 0.0)
            self.assertIn("cluster_id", top_score)
            self.assertIn("cluster_score", top_score)
            self.assertIn("activation_value", top_score)
            self.assertEqual(ranking.get("selection_mode"), "node_first")
            self.assertTrue(ranking.get("reasoning_graph_cluster_relationships"))
            self.assertIn("reasoning_spatial_grid", ranking)
            self.assertTrue(ranking["reasoning_spatial_grid"].get("top_regions"))
            self.assertTrue(ranking["reasoning_spatial_grid"].get("region_competition_log"))
            self.assertTrue(ranking["reasoning_spatial_grid"].get("region_dominance_log"))
            self.assertTrue(ranking["reasoning_spatial_grid"].get("temporal_update_log"))
            self.assertTrue(ranking["reasoning_spatial_grid"].get("concept_creation_log"))
            self.assertEqual(ranking.get("selected_reasoning_path"), [selected[0].entry_id])
            self.assertIsNotNone(ranking.get("selected_spatial_region_id"))
            self.assertIn(selected[0].entry_id, ranking.get("selected_spatial_nodes", []))
            self.assertTrue(ranking.get("candidate_reasoning_paths"))
            self.assertLess(float(ranking["ranking_scores"][1].get("extension_bonus", 0.0) or 0.0), 0.05)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_memory_reconstruction_handles_benchmark_recall_patterns(self) -> None:
        scenarios = [
            (
                [
                    "Help me charge my laptop safely.",
                    "What did I ask earlier?",
                ],
                ["charge", "laptop"],
                "temporal_recall",
            ),
            (
                [
                    "Help me finish my task safely.",
                    "Charging my laptop before I leave.",
                    "What was my last planning request?",
                ],
                ["charging", "laptop", "before", "leave"],
                "turn",
            ),
            (
                [
                    "I keep frozen berries in the pantry.",
                    "Actually, I moved the frozen berries to the freezer.",
                    "Where should you assume the frozen berries are now?",
                ],
                ["frozen", "berries", "freezer"],
                "state_memory",
            ),
            (
                [
                    "I do not eat peanuts, but almonds are fine.",
                    "Correction: almonds also bother me, but cashews are okay.",
                    "What nuts should you currently assume are okay for me?",
                ],
                ["cashews"],
                "state_memory",
            ),
            (
                [
                    "The draft document is titled Launch Notes v1.",
                    "Rename it to Launch Notes Final Draft.",
                    "What is the current document title?",
                ],
                ["launch", "notes", "final", "draft"],
                "state_memory",
            ),
            (
                [
                    "For tomorrow's trip, I can only bring one backpack.",
                    "I also need to keep the total weight under 8 kilograms.",
                    "And the laptop must stay in an easy-access pocket.",
                    "What packing constraints have I given you?",
                ],
                ["one backpack", "under 8", "laptop"],
                "state_memory",
            ),
        ]
        for turns, expected_terms, expected_source in scenarios:
            temp_dir = Path(tempfile.mkdtemp(prefix="brain_memory_reconstruction_test_"))
            try:
                prototype = BrainHybridPrototype(
                    state_dir=temp_dir,
                    llm_provider="deterministic",
                    bootstrap_reasoning=False,
                )
                result = None
                for turn in turns:
                    result = prototype.process_user_turn(turn, inspect=True)
                assert result is not None
                self.assertEqual(result.route_decision.route_type, "memory_question")
                answer = result.llm_explanation.user_facing_text.lower()
                for term in expected_terms:
                    self.assertIn(term, answer)
                reconstructed = result.trace.merged_memory_state.get("reconstructed_context", {})
                self.assertTrue(reconstructed)
                self.assertGreaterEqual(float(reconstructed.get("confidence", 0.0)), 0.4)
                self.assertIn(reconstructed.get("answer_mode"), {"medium", "high"})
                self.assertTrue(reconstructed.get("synthesized_summary"))
                source_types = {
                    item.get("source_type")
                    for item in reconstructed.get("partial_matches", [])
                    if isinstance(item, dict)
                }
                self.assertIn(expected_source, source_types)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_response_layer_translates_planning_to_action_language(self) -> None:
        examples = [
            (
                "Plan my bugfixing work. I must reproduce the bug before changing code, I must update tests before merging, and deployment can only happen after review passes.",
                {
                    "source_of_truth": "brain",
                    "task_type": "planning",
                    "progress": {
                        "clarification_required": True,
                        "next_step": "What concrete task or object do you want help with?",
                    },
                    "decision_context": {"decision_summary": "planning"},
                },
                ["reproducing the bug", "tests", "review", "deploy"],
            ),
            (
                "Plan my study week. I work Monday to Friday from 9 AM to 5 PM, need 6 hours of study total, cannot study more than 2 hours in a day, and want Sunday completely off.",
                {
                    "source_of_truth": "brain",
                    "task_type": "planning",
                    "progress": {
                        "status": "stalled",
                        "next_step": "Provide a more concrete next action or object so BrAIn can continue goal tracking.",
                    },
                    "decision_context": {
                        "decision_summary": "Strategy B: Conservative Retry",
                        "next_action": "Pause work on week so the failure does not cascade.",
                    },
                },
                ["six one-hour", "2-hour", "sunday"],
            ),
            (
                "My printer is not printing. The printer has power, paper is loaded, and the document is stuck in the queue. Give me the best next troubleshooting path.",
                {
                    "source_of_truth": "brain",
                    "task_type": "planning",
                    "progress": {},
                    "decision_context": {},
                },
                ["queue", "spooler", "test page"],
            ),
        ]
        forbidden = [
            "what concrete task",
            "recommended plan",
            "conservative retry",
            "could not confidently route",
            "focus on complete_task",
            "brain selected",
        ]
        for query, payload, expected_terms in examples:
            answer = render_llm_payload_v2_answer(
                user_input_text=query,
                route_type="planning_request",
                llm_payload_v2=payload,
                mode="direct_answer",
            ).lower()
            for term in expected_terms:
                self.assertIn(term, answer)
            for term in forbidden:
                self.assertNotIn(term, answer)

    def test_response_layer_polishes_memory_recall_language(self) -> None:
        answer = render_llm_payload_v2_answer(
            user_input_text="What nuts should you currently assume are okay for me?",
            route_type="memory_question",
            llm_payload_v2={
                "source_of_truth": "brain",
                "task_type": "memory_recall",
                "answer_state": {"dietary_ok:cashews": "cashews okay"},
            },
            mode="direct_answer",
        )
        lowered = answer.lower()
        self.assertIn("cashews", lowered)
        self.assertIn("okay", lowered)
        self.assertNotIn("related dietary memory", lowered)
        self.assertNotIn("current remembered state", lowered)

    def _add_cache_entry(
        self,
        prototype: BrainHybridPrototype,
        *,
        query_text: str,
        answer_text: str,
        topic_key: str,
        confidence_estimate: float,
        reuse_count: int,
        successful_uses: int,
    ) -> None:
        parsed_memory = BrainMemory(
            actor="assistant",
            intent="inform",
            action="answer",
            topic=topic_key,
        )
        parsed_brain_input = BrainCodec.resolve_memory(parsed_memory)
        topic_keys = prototype.expand_topic_keys(query_text, topic_key)
        normalized_facts = prototype._distill_cached_answer_facts(answer_text)
        entry = prototype.memory_store.add(
            entry_type="llm_knowledge_cache",
            source_text=answer_text,
            parsed_memory=parsed_memory,
            brain_input=parsed_brain_input,
            metadata={
                "source": "llm",
                "trust_level": "low",
                "verification_status": "unverified",
                "topic_key": topic_key,
                "topic_keys": topic_keys,
                "query_text": query_text,
                "answer_text": answer_text,
                "normalized_facts": normalized_facts,
                "answer_state": prototype._cached_answer_state(topic_key, normalized_facts),
                "confidence_estimate": confidence_estimate,
                "reuse_count": reuse_count,
                "successful_uses": successful_uses,
                "conflict_win_count": 0,
                "selected_streak_count": 0,
                "stabilized": False,
                "last_reused_at": "",
                "route_type": "factual_question",
            },
        )
        self.assertIsNotNone(entry)


if __name__ == "__main__":
    unittest.main()
