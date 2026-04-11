from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

from brain_codec import BrainCodec, BrainMemory
from brain_hybrid_prototype import BrainHybridPrototype
from brain_reasoning_graph import (
    ReasoningGraphCandidate,
    ReasoningGraphEvaluator,
    ReasoningGraphScoringConfig,
    ReasoningSpatialConcept,
    ReasoningTemporalTransition,
)


ProbeEntry = Dict[str, object]


def _add_cache_entry(
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
    prototype.memory_store.add(
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


def _format_relationships(relationships: Sequence[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for relationship in relationships:
        lines.append(
            "- "
            + f"{relationship.get('relationship_type', 'UNKNOWN')} -> entry {relationship.get('related_entry_id')} "
            + f"(strength={float(relationship.get('strength', 0.0) or 0.0):.4f}, "
            + f"evidence={relationship.get('evidence', [])})"
        )
    return lines


def _score_breakdown(item: Dict[str, object]) -> str:
    return (
        f"base={float(item.get('base_score', 0.0) or 0.0):.4f} "
        f"+ support={float(item.get('support_bonus', 0.0) or 0.0):.4f} "
        f"+ extension={float(item.get('extension_bonus', 0.0) or 0.0):.4f} "
        f"+ directness={float(item.get('directness_bonus', 0.0) or 0.0):.4f} "
        f"+ coherence={float(item.get('coherence_bonus', 0.0) or 0.0):.4f} "
        f"+ dense_cluster={float(item.get('dense_cluster_bonus', 0.0) or 0.0):.4f} "
        f"+ cluster_influence={float(item.get('cluster_influence_bonus', 0.0) or 0.0):.4f} "
        f"+ intent={float(item.get('intent_bonus', 0.0) or 0.0):.4f} "
        f"+ concept_predictive={float(item.get('concept_predictive_bias', 0.0) or 0.0):.4f} "
        f"- contradiction={float(item.get('contradiction_penalty', 0.0) or 0.0):.4f}"
    )


def _run_probe(
    *,
    name: str,
    query_text: str,
    entries: Sequence[ProbeEntry],
    concept_warmup_runs: int = 1,
) -> str:
    temp_dir = Path(tempfile.mkdtemp(prefix="brain_reasoning_probe_"))
    try:
        prototype = BrainHybridPrototype(
            state_dir=temp_dir,
            bootstrap_reasoning=False,
        )
        for entry in entries:
            _add_cache_entry(prototype, query_text=query_text, **entry)
        for _ in range(max(0, int(concept_warmup_runs))):
            prototype._cache_rank_candidates(
                user_input_text=query_text,
                top_k=5,
                update_reuse=False,
            )
        selected, ranking = prototype._cache_rank_candidates(
            user_input_text=query_text,
            top_k=5,
            update_reuse=False,
        )
        lines = [f"## {name}", f"Query: `{query_text}`", ""]
        lines.append(f"Concept warmup runs: `{max(0, int(concept_warmup_runs))}`")
        lines.append(f"Query intent: `{ranking.get('query_intent', 'UNKNOWN')}`")
        lines.append(f"Selection mode: `{ranking.get('selection_mode', 'node_first')}`")
        lines.append(f"Selected cluster: `{ranking.get('selected_cluster_id', 'None')}`")
        lines.append(f"Selected cluster score: `{float(ranking.get('selected_cluster_score', 0.0) or 0.0):.4f}`")
        lines.append(f"Selected entry: `{selected[0].entry_id if selected else 'None'}`")
        lines.append(f"Cluster selection reason: `{ranking.get('cluster_selection_reason', '')}`")
        lines.append(f"Selected reasoning path: `{ranking.get('selected_reasoning_path', [])}`")
        lines.append(f"Selected path score: `{float(ranking.get('selected_path_score', 0.0) or 0.0):.4f}`")
        lines.append(f"Selected path activation: `{float(ranking.get('selected_path_activation', 0.0) or 0.0):.4f}`")
        lines.append(f"Selected path final score: `{float(ranking.get('selected_path_final_score', 0.0) or 0.0):.4f}`")
        lines.append(f"Selected path breakdown: `{ranking.get('selected_path_breakdown', '')}`")
        lines.append(f"Selected spatial region: `{ranking.get('selected_spatial_region_id', 'None')}`")
        lines.append(f"Selected spatial nodes: `{ranking.get('selected_spatial_nodes', [])}`")
        lines.append(f"Selected spatial clusters: `{ranking.get('selected_spatial_clusters', [])}`")
        lines.append(f"Selected spatial activation: `{float(ranking.get('selected_spatial_activation', 0.0) or 0.0):.4f}`")
        lines.append(f"Spatial selection reason: `{ranking.get('spatial_selection_reason', '')}`")
        lines.append(f"Spatial consistency warning: `{ranking.get('spatial_consistency_warning', '')}`")
        lines.append("")
        lines.append("### Ranking")
        for item in ranking.get("ranking_scores", []):
            lines.append(
                "- "
                + f"entry `{item['entry_id']}`: "
                + f"base=`{float(item.get('base_score', 0.0) or 0.0):.4f}`, "
                + f"support=`{float(item.get('support_bonus', 0.0) or 0.0):.4f}`, "
                + f"contradiction=`{float(item.get('contradiction_penalty', 0.0) or 0.0):.4f}`, "
                + f"extension=`{float(item.get('extension_bonus', 0.0) or 0.0):.4f}`, "
                + f"directness=`{float(item.get('directness_score', 0.0) or 0.0):.4f}`, "
                + f"coherence=`{float(item.get('coherence_bonus', 0.0) or 0.0):.4f}`, "
                + f"dense_cluster=`{float(item.get('dense_cluster_bonus', 0.0) or 0.0):.4f}`, "
                + f"cluster_influence=`{float(item.get('cluster_influence_bonus', 0.0) or 0.0):.4f}`, "
                + f"concept_predictive=`{float(item.get('concept_predictive_bias', 0.0) or 0.0):.4f}`, "
                + f"activation=`{float(item.get('activation_value', 0.0) or 0.0):.4f}`, "
                + f"final=`{float(item.get('final_score', item.get('total_score', 0.0)) or 0.0):.4f}`"
            )
            lines.append(
                "  "
                + f"query_intent=`{item.get('query_intent', 'UNKNOWN')}`, "
                + f"raw_support=`{float(item.get('raw_support_score', 0.0) or 0.0):.4f}`, "
                + f"raw_causal=`{float(item.get('raw_causal_score', 0.0) or 0.0):.4f}`, "
                + f"raw_extension=`{float(item.get('raw_extension_score', 0.0) or 0.0):.4f}`, "
                + f"raw_contradiction=`{float(item.get('raw_contradiction_score', 0.0) or 0.0):.4f}`"
            )
            lines.append(
                "  "
                + f"directness_score=`{float(item.get('directness_score', 0.0) or 0.0):.4f}`, "
                + f"directness_bonus=`{float(item.get('directness_bonus', 0.0) or 0.0):.4f}`, "
                + f"cluster_strength=`{float(item.get('cluster_strength', 0.0) or 0.0):.4f}`, "
                + f"coherence_bonus=`{float(item.get('coherence_bonus', 0.0) or 0.0):.4f}`, "
                + f"dense_cluster_bonus=`{float(item.get('dense_cluster_bonus', 0.0) or 0.0):.4f}`, "
                + f"cluster_influence_bonus=`{float(item.get('cluster_influence_bonus', 0.0) or 0.0):.4f}`, "
                + f"intent_bonus=`{float(item.get('intent_bonus', 0.0) or 0.0):.4f}`, "
                + f"concept_predictive_bias=`{float(item.get('concept_predictive_bias', 0.0) or 0.0):.4f}`"
            )
            lines.append(
                "  "
                + f"cluster_id=`{int(item.get('cluster_id', -1) or -1)}`, "
                + f"cluster_score=`{float(item.get('cluster_score', 0.0) or 0.0):.4f}`, "
                + f"cluster_rank=`{int(item.get('cluster_rank', 0) or 0)}`, "
                + f"cluster_size=`{int(item.get('cluster_size', 0) or 0)}`, "
                + f"cluster_density=`{float(item.get('cluster_density', 0.0) or 0.0):.4f}`, "
                + f"base_activation=`{float(item.get('base_activation_value', 0.0) or 0.0):.4f}`, "
                + f"propagated_activation=`{float(item.get('propagated_activation_bonus', 0.0) or 0.0):.4f}`, "
                + f"inhibition=`{float(item.get('inhibition_penalty', 0.0) or 0.0):.4f}`, "
                + f"activation_value=`{float(item.get('activation_value', 0.0) or 0.0):.4f}`"
            )
            lines.append(f"  breakdown: `{_score_breakdown(item)}`")
            for relationship_line in _format_relationships(item.get("graph_relationships", [])):
                lines.append(f"  {relationship_line}")
        lines.append("")
        lines.append("### Clusters")
        for cluster in ranking.get("reasoning_graph_clusters", []):
            lines.append(
                "- "
                + f"cluster `{cluster.get('cluster_id', -1)}`: "
                + f"score=`{float(cluster.get('cluster_score', 0.0) or 0.0):.4f}`, "
                + f"final_score=`{float(cluster.get('cluster_final_score', 0.0) or 0.0):.4f}`, "
                + f"node_sum=`{float(cluster.get('node_score_sum', 0.0) or 0.0):.4f}`, "
                + f"structural_bonus=`{float(cluster.get('structural_bonus', 0.0) or 0.0):.4f}`, "
                + f"support_bonus=`{float(cluster.get('cluster_support_bonus', 0.0) or 0.0):.4f}`, "
                + f"contradiction_penalty=`{float(cluster.get('cluster_contradiction_penalty', 0.0) or 0.0):.4f}`, "
                + f"base_activation=`{float(cluster.get('base_activation_value', 0.0) or 0.0):.4f}`, "
                + f"propagated_activation=`{float(cluster.get('propagated_activation_bonus', 0.0) or 0.0):.4f}`, "
                + f"inhibition=`{float(cluster.get('inhibition_penalty', 0.0) or 0.0):.4f}`, "
                + f"activation=`{float(cluster.get('activation_value', 0.0) or 0.0):.4f}`, "
                + f"density=`{float(cluster.get('density', 0.0) or 0.0):.4f}`, "
                + f"support_edges=`{int(cluster.get('support_edges', cluster.get('support_edge_count', 0)) or 0)}`, "
                + f"contradiction_edges=`{int(cluster.get('contradiction_edges', 0) or 0)}`, "
                + f"entries=`{cluster.get('nodes', cluster.get('entry_ids', []))}`"
            )
        lines.append("")
        lines.append("### Path Competition")
        for path in ranking.get("candidate_reasoning_paths", []):
            lines.append(
                "- "
                + f"path `{path.get('node_ids', [])}`: "
                + f"score=`{float(path.get('path_score', 0.0) or 0.0):.4f}`, "
                + f"relative_strength=`{float(path.get('relative_strength', 0.0) or 0.0):.4f}`, "
                + f"adjusted_score=`{float(path.get('adjusted_score', 0.0) or 0.0):.4f}`, "
                + f"activation=`{float(path.get('path_activation', 0.0) or 0.0):.4f}`, "
                + f"final=`{float(path.get('path_final_score', 0.0) or 0.0):.4f}`"
            )
            lines.append(f"  breakdown: `{path.get('breakdown', '')}`")
        lines.append("")
        spatial_grid = ranking.get("reasoning_spatial_grid", {}) or {}
        lines.append("### Spatial Grid")
        lines.append(
            "- "
            + f"grid=`{spatial_grid.get('width', 0)}x{spatial_grid.get('height', 0)}`, "
            + f"decay=`{float(spatial_grid.get('decay_factor', 0.0) or 0.0):.4f}`, "
            + f"steps=`{int(spatial_grid.get('propagation_steps', 0) or 0)}`"
        )
        lines.append("#### Node Mapping")
        for mapping in spatial_grid.get("node_mappings", []):
            center_x = int(mapping.get("center_x", 0) or 0)
            center_y = int(mapping.get("center_y", 0) or 0)
            lines.append(
                "- "
                + f"entry `{mapping.get('entry_id')}` -> grid `({center_x}, {center_y})`, "
                + f"initial=`{float(mapping.get('initial_region_activation', 0.0) or 0.0):.4f}`, "
                + f"propagated=`{float(mapping.get('propagated_region_activation', 0.0) or 0.0):.4f}`, "
                + f"final=`{float(mapping.get('final_region_activation', 0.0) or 0.0):.4f}`"
            )
        lines.append("#### Cluster Mapping")
        for mapping in spatial_grid.get("cluster_mappings", []):
            lines.append(
                "- "
                + f"cluster `{mapping.get('cluster_id')}` center=`({int(mapping.get('center_x', 0) or 0)}, {int(mapping.get('center_y', 0) or 0)})`, "
                + f"radius=`{int(mapping.get('radius', 0) or 0)}`, "
                + f"activation=`{float(mapping.get('activation_value', 0.0) or 0.0):.4f}`, "
                + f"final_region=`{float(mapping.get('final_region_activation', 0.0) or 0.0):.4f}`, "
                + f"entries=`{mapping.get('entry_ids', [])}`"
            )
        lines.append("#### Propagation")
        for log_line in spatial_grid.get("propagation_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Inhibition")
        for log_line in spatial_grid.get("inhibition_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Temporal Updates")
        for log_line in spatial_grid.get("temporal_update_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Dominance Evolution")
        for log_line in spatial_grid.get("dominance_evolution_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Reuse")
        for log_line in spatial_grid.get("concept_reuse_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Pre-Activation")
        for log_line in spatial_grid.get("concept_pre_activation_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Anticipation")
        for log_line in spatial_grid.get("anticipation_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Activation")
        for log_line in spatial_grid.get("concept_activation_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Importance")
        for log_line in spatial_grid.get("concept_importance_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Importance Evolution")
        for log_line in spatial_grid.get("concept_importance_evolution_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Prediction Accuracy")
        for log_line in spatial_grid.get("prediction_accuracy_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Exploration")
        for log_line in spatial_grid.get("concept_exploration_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Uncertainty")
        for log_line in spatial_grid.get("concept_uncertainty_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Cross-Domain Links")
        for log_line in spatial_grid.get("concept_cross_domain_link_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Generalization")
        for log_line in spatial_grid.get("concept_generalization_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Stability Controls")
        for log_line in spatial_grid.get("concept_stability_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Abstraction Layers")
        for log_line in spatial_grid.get("concept_abstraction_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Temporal Sequences")
        for log_line in spatial_grid.get("temporal_sequence_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Temporal Predictions")
        for log_line in spatial_grid.get("temporal_prediction_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Goals And Plans")
        for log_line in spatial_grid.get("goal_selection_log", []):
            lines.append(f"- `{log_line}`")
        for log_line in spatial_grid.get("goal_plan_log", []):
            lines.append(f"- `{log_line}`")
        for log_line in spatial_grid.get("goal_feedback_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Evolution")
        for log_line in spatial_grid.get("concept_evolution_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Competition")
        for log_line in spatial_grid.get("concept_competition_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Meta-Concept Influence")
        for log_line in spatial_grid.get("meta_concept_influence_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Cluster Competition")
        for log_line in spatial_grid.get("concept_cluster_competition_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Hierarchy")
        for log_line in spatial_grid.get("concept_hierarchy_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Clusters")
        for log_line in spatial_grid.get("concept_cluster_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Creation")
        for log_line in spatial_grid.get("concept_creation_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Concept Merging")
        for log_line in spatial_grid.get("concept_merge_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Region Competition")
        for log_line in spatial_grid.get("region_competition_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Region Dominance")
        for log_line in spatial_grid.get("region_dominance_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Region Merging")
        for log_line in spatial_grid.get("region_merge_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Region Expansion")
        for log_line in spatial_grid.get("region_expansion_log", []):
            lines.append(f"- `{log_line}`")
        lines.append("#### Placement Map")
        for line in spatial_grid.get("placement_map_lines", []):
            lines.append(f"`{line}`")
        lines.append("#### Cluster Boundaries")
        for line in spatial_grid.get("cluster_boundary_lines", []):
            lines.append(f"- `{line}`")
        lines.append("#### Pre-Competition Regions")
        for region in spatial_grid.get("pre_competition_regions", []):
            lines.append(
                "- "
                + f"region `{region.get('region_id')}`: "
                + f"total=`{float(region.get('total_activation', 0.0) or 0.0):.4f}`, "
                + f"peak=`{float(region.get('peak_activation', 0.0) or 0.0):.4f}`, "
                + f"centroid=`({float(region.get('centroid_x', 0.0) or 0.0):.2f}, {float(region.get('centroid_y', 0.0) or 0.0):.2f})`, "
                + f"nodes=`{region.get('source_nodes', [])}`, "
                + f"clusters=`{region.get('source_clusters', [])}`, "
                + f"dominance=`{float(region.get('dominance_value', 0.0) or 0.0):.4f}`, "
                + f"merged=`{region.get('merged_region_ids', [])}`, "
                + f"dominant_node=`{region.get('dominant_node_id')}`"
            )
        lines.append("#### Top Regions")
        for region in spatial_grid.get("top_regions", []):
            lines.append(
                "- "
                + f"region `{region.get('region_id')}`: "
                + f"total=`{float(region.get('total_activation', 0.0) or 0.0):.4f}`, "
                + f"peak=`{float(region.get('peak_activation', 0.0) or 0.0):.4f}`, "
                + f"centroid=`({float(region.get('centroid_x', 0.0) or 0.0):.2f}, {float(region.get('centroid_y', 0.0) or 0.0):.2f})`, "
                + f"nodes=`{region.get('source_nodes', [])}`, "
                + f"clusters=`{region.get('source_clusters', [])}`, "
                + f"competition_penalty=`{float(region.get('competition_penalty', 0.0) or 0.0):.4f}`, "
                + f"rank=`{int(region.get('post_competition_rank', 0) or 0)}`, "
                + f"dominance=`{float(region.get('dominance_value', 0.0) or 0.0):.4f}`, "
                + f"merged=`{region.get('merged_region_ids', [])}`, "
                + f"dominant_node=`{region.get('dominant_node_id')}`"
            )
        lines.append("#### Heatmap Before")
        for line in spatial_grid.get("heatmap_before_lines", []):
            lines.append(f"`{line}`")
        lines.append("#### Heatmap After")
        for line in spatial_grid.get("heatmap_after_lines", []):
            lines.append(f"`{line}`")
        lines.append("#### Activated Concepts")
        for concept in spatial_grid.get("activated_concepts", []):
            lines.append(
                "- "
                + f"concept `{concept.get('concept_id')}`: "
                + f"similarity=`{float(concept.get('similarity', 0.0) or 0.0):.4f}`, "
                + f"influence=`{float(concept.get('influence', 0.0) or 0.0):.4f}`, "
                + f"importance=`{float(concept.get('importance', 0.0) or 0.0):.4f}`, "
                + f"pre_activation=`{float(concept.get('pre_activation_influence', 0.0) or 0.0):.4f}`, "
                + f"cluster_adjustment=`{float(concept.get('cluster_adjustment', 0.0) or 0.0):.4f}`, "
                + f"reuse=`{int(concept.get('pre_reuse_count', 0) or 0)}->{int(concept.get('post_reuse_count', 0) or 0)}`, "
                + f"nodes=`{concept.get('associated_nodes', [])}`, "
                + f"clusters=`{concept.get('associated_clusters', [])}`, "
                + f"concept_cluster=`{concept.get('concept_cluster_id', -1)}`, "
                + f"predicted=`{bool(concept.get('predicted', False))}`, "
                + f"meta=`{bool(concept.get('is_meta_concept', False))}`"
            )
        lines.append("#### Anticipated Concepts")
        for prediction in spatial_grid.get("anticipated_concepts", []):
            lines.append(
                "- "
                + f"concept `{prediction.get('concept_id')}`: "
                + f"score=`{float(prediction.get('predicted_score', 0.0) or 0.0):.4f}`, "
                + f"similarity=`{float(prediction.get('similarity', 0.0) or 0.0):.4f}`, "
                + f"importance=`{float(prediction.get('importance', 0.0) or 0.0):.4f}`, "
                + f"confidence=`{float(prediction.get('prediction_confidence', 0.0) or 0.0):.4f}`, "
                + f"meta=`{bool(prediction.get('is_meta_concept', False))}`, "
                + f"children=`{prediction.get('child_concept_ids', [])}`, "
                + f"concept_cluster=`{prediction.get('concept_cluster_id', -1)}`"
            )
        lines.append("#### Concept Cluster Snapshot")
        for cluster in spatial_grid.get("concept_clusters", []):
            lines.append(
                "- "
                + f"concept_cluster `{cluster.get('concept_cluster_id')}`: "
                + f"concepts=`{cluster.get('concept_ids', [])}`, "
                + f"score=`{float(cluster.get('cluster_score', 0.0) or 0.0):.4f}`, "
                + f"final_score=`{float(cluster.get('final_score', 0.0) or 0.0):.4f}`, "
                + f"competition_penalty=`{float(cluster.get('competition_penalty', 0.0) or 0.0):.4f}`, "
                + f"importance=`{float(cluster.get('importance', 0.0) or 0.0):.4f}`, "
                + f"centroid=`({float(cluster.get('centroid_x', 0.0) or 0.0):.2f}, {float(cluster.get('centroid_y', 0.0) or 0.0):.2f})`, "
                + f"nodes=`{cluster.get('shared_nodes', [])}`, "
                + f"relationships=`{cluster.get('shared_relationships', [])}`"
            )
        lines.append("#### Concept Memory Snapshot")
        for concept in spatial_grid.get("concept_memory_snapshot", []):
            lines.append(
                "- "
                + f"concept `{concept.get('concept_id')}`: "
                + f"centroid=`({float(concept.get('centroid_x', 0.0) or 0.0):.2f}, {float(concept.get('centroid_y', 0.0) or 0.0):.2f})`, "
                + f"activation=`{float(concept.get('activation_value', 0.0) or 0.0):.4f}`, "
                + f"importance=`{float(concept.get('importance', 0.0) or 0.0):.4f}`, "
                + f"importance_adjustment=`{float(concept.get('importance_memory_adjustment', 0.0) or 0.0):.4f}`, "
                + f"dominance=`{float(concept.get('average_dominance', 0.0) or 0.0):.4f}`, "
                + f"stability=`{float(concept.get('stability_score', 0.0) or 0.0):.4f}`, "
                + f"prediction_accuracy=`{float(concept.get('prediction_accuracy', 0.0) or 0.0):.4f}`, "
                + f"uncertainty=`{float(concept.get('uncertainty', 0.0) or 0.0):.4f}`, "
                + f"prediction_confidence=`{float(concept.get('prediction_confidence', 0.0) or 0.0):.4f}`, "
                + f"nodes=`{concept.get('associated_nodes', [])}`, "
                + f"clusters=`{concept.get('associated_clusters', [])}`, "
                + f"relationships=`{concept.get('relationship_types', [])}`, "
                + f"terms=`{concept.get('associated_terms', [])}`, "
                + f"reuse_count=`{int(concept.get('reuse_count', 0) or 0)}`, "
                + f"meta=`{bool(concept.get('is_meta_concept', False))}`, "
                + f"generalized=`{bool(concept.get('is_generalized_concept', False))}`, "
                + f"children=`{concept.get('child_concept_ids', [])}`, "
                + f"parents=`{concept.get('parent_concept_ids', [])}`, "
                + f"cross_domain_links=`{concept.get('cross_domain_links', {})}`, "
                + f"concept_cluster=`{concept.get('concept_cluster_id', -1)}`"
            )
        lines.append("")
        lines.append("### Cluster Relationships")
        for cluster_edge in ranking.get("reasoning_graph_cluster_relationships", []):
            lines.append(
                "- "
                + f"cluster `{cluster_edge.get('source_cluster_id')}` -> cluster `{cluster_edge.get('target_cluster_id')}` "
                + f"`{cluster_edge.get('relationship_type', 'UNKNOWN')}` "
                + f"(strength=`{float(cluster_edge.get('strength', 0.0) or 0.0):.4f}`, "
                + f"node_edges=`{int(cluster_edge.get('node_edge_count', 0) or 0)}`, "
                + f"evidence=`{cluster_edge.get('evidence', [])}`)"
            )
        lines.append("")
        lines.append("### Graph Edges")
        for edge in ranking.get("reasoning_graph_edges", []):
            lines.append(
                "- "
                + f"`{edge.get('source_entry_id')}` -> `{edge.get('target_entry_id')}` "
                + f"`{edge.get('relationship_type', 'UNKNOWN')}` "
                + f"(strength=`{float(edge.get('strength', 0.0) or 0.0):.4f}`, "
                + f"evidence=`{edge.get('evidence', [])}`)"
            )
        lines.append("")
        return "\n".join(lines)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _run_concept_cluster_competition_probe() -> str:
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
    selection = evaluator.select_nodes(graph)
    spatial_grid = graph.spatial_grid
    lines = [
        "## Concept Cluster Competition Fixture",
        "Query: `What color are the sky and grass?`",
        "",
        "This fixture seeds two stored concepts in nearby but separate concept clusters so cluster-level suppression is easy to inspect.",
        "",
        f"Selected entry: `{selection.selected_node_entry_id}`",
        f"Selected spatial region: `{selection.selected_spatial_region_id}`",
        f"Selected spatial nodes: `{selection.selected_spatial_nodes}`",
        "",
        "### Concept Cluster Competition",
    ]
    if spatial_grid is None:
        lines.append("- `No spatial grid was produced.`")
        return "\n".join(lines)
    for log_line in spatial_grid.concept_cluster_competition_log:
        lines.append(f"- `{log_line}`")
    if not spatial_grid.concept_cluster_competition_log:
        lines.append("- `No concept-cluster competition was triggered.`")
    lines.append("")
    lines.append("### Activated Concepts")
    for concept in spatial_grid.activated_concepts:
        lines.append(
            "- "
            + f"concept `{concept.concept_id}`: "
            + f"influence=`{concept.influence:.4f}`, "
            + f"cluster_adjustment=`{concept.cluster_adjustment:.4f}`, "
            + f"concept_cluster=`{concept.concept_cluster_id}`, "
            + f"predicted=`{concept.predicted}`"
        )
    lines.append("")
    lines.append("### Concept Cluster Snapshot")
    for cluster in spatial_grid.concept_clusters:
        lines.append(
            "- "
            + f"concept_cluster `{cluster.concept_cluster_id}`: "
            + f"concepts=`{cluster.concept_ids}`, "
            + f"score=`{cluster.cluster_score:.4f}`, "
            + f"final_score=`{cluster.final_score:.4f}`, "
            + f"competition_penalty=`{cluster.competition_penalty:.4f}`, "
            + f"importance=`{cluster.importance:.4f}`, "
            + f"centroid=`({cluster.centroid_x:.2f}, {cluster.centroid_y:.2f})`"
        )
    lines.append("")
    return "\n".join(lines)


def _run_stability_generalization_probe() -> str:
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
    spatial_grid = graph.spatial_grid
    lines = [
        "## Stability / Generalization Fixture",
        "Query: `What color are the sky and grass?`",
        "",
        "This fixture forces deterministic exploration and seeds one over-important concept so uncertainty, cross-domain linking, generalization, and importance caps can be inspected.",
        "",
        "### Concept Exploration",
    ]
    if spatial_grid is None:
        lines.append("- `No spatial grid was produced.`")
        return "\n".join(lines)
    for log_line in spatial_grid.concept_exploration_log:
        lines.append(f"- `{log_line}`")
    lines.append("### Concept Uncertainty")
    for log_line in spatial_grid.concept_uncertainty_log:
        lines.append(f"- `{log_line}`")
    lines.append("### Cross-Domain Links")
    for log_line in spatial_grid.concept_cross_domain_link_log:
        lines.append(f"- `{log_line}`")
    lines.append("### Concept Generalization")
    for log_line in spatial_grid.concept_generalization_log:
        lines.append(f"- `{log_line}`")
    lines.append("### Stability Controls")
    for log_line in spatial_grid.concept_stability_log:
        lines.append(f"- `{log_line}`")
    lines.append("### Abstraction Layers")
    for log_line in spatial_grid.concept_abstraction_log:
        lines.append(f"- `{log_line}`")
    lines.append("### Temporal Sequences")
    for log_line in spatial_grid.temporal_sequence_log:
        lines.append(f"- `{log_line}`")
    lines.append("### Temporal Predictions")
    for log_line in spatial_grid.temporal_prediction_log:
        lines.append(f"- `{log_line}`")
    lines.append("### Goals And Plans")
    for log_line in spatial_grid.goal_selection_log:
        lines.append(f"- `{log_line}`")
    for log_line in spatial_grid.goal_plan_log:
        lines.append(f"- `{log_line}`")
    for log_line in spatial_grid.goal_feedback_log:
        lines.append(f"- `{log_line}`")
    lines.append("### Temporal Transition Snapshot")
    for transition in spatial_grid.temporal_transitions:
        lines.append(
            "- "
            + f"`{transition.source_concept_id}->{transition.target_concept_id}`: "
            + f"frequency=`{transition.frequency}`, "
            + f"confidence=`{transition.confidence:.4f}`, "
            + f"reliability=`{transition.reliability:.4f}`, "
            + f"simulation_accuracy=`{transition.simulation_accuracy:.4f}`, "
            + f"last_prediction_error=`{transition.last_prediction_error:.4f}`, "
            + f"average_prediction_error=`{transition.average_prediction_error:.4f}`, "
            + f"successes=`{transition.success_count}`, "
            + f"failures=`{transition.failure_count}`, "
            + f"simulation_correct=`{transition.simulation_correct_count}/{transition.simulation_total_count}`"
        )
    lines.append("### Concept Memory Snapshot")
    for concept in spatial_grid.concept_memory_snapshot:
        lines.append(
            "- "
            + f"concept `{concept.concept_id}`: "
            + f"importance=`{concept.importance:.4f}`, "
            + f"uncertainty=`{concept.uncertainty:.4f}`, "
            + f"meta=`{concept.is_meta_concept}`, "
            + f"generalized=`{concept.is_generalized_concept}`, "
            + f"abstraction_level=`{concept.abstraction_level}`, "
            + f"links=`{concept.cross_domain_links}`"
        )
    lines.append("")
    return "\n".join(lines)


def build_report() -> str:
    sections = [
        "# Reasoning Graph Probe Report",
        "",
        "This report captures sample graph-evaluation scenarios for the current BrAIn reasoning-graph layer. Scenarios use deterministic warmup runs so concept reuse, meta-concept influence, prediction feedback, and concept evolution can be inspected on the reported run.",
        "",
        _run_probe(
            name="Support / Causal Scenario",
            query_text="Why is the sky blue?",
            entries=[
                {
                    "answer_text": "The sky is blue during the day.",
                    "topic_key": "sky_color:daytime",
                    "confidence_estimate": 0.78,
                    "reuse_count": 2,
                    "successful_uses": 1,
                },
                {
                    "answer_text": "The sky is blue because shorter blue wavelengths scatter more strongly in Earth's atmosphere.",
                    "topic_key": "sky_color:daytime",
                    "confidence_estimate": 0.74,
                    "reuse_count": 1,
                    "successful_uses": 1,
                },
                {
                    "answer_text": "Blue light scatters more strongly in Earth's atmosphere because shorter wavelengths spread more widely.",
                    "topic_key": "sky_scattering:reason",
                    "confidence_estimate": 0.70,
                    "reuse_count": 0,
                    "successful_uses": 0,
                },
            ],
        ),
        _run_probe(
            name="Contradiction / Anticipation Replay Scenario",
            query_text="What color is the sky during the day?",
            concept_warmup_runs=2,
            entries=[
                {
                    "answer_text": "The sky is blue during the day.",
                    "topic_key": "sky_color:daytime",
                    "confidence_estimate": 0.80,
                    "reuse_count": 2,
                    "successful_uses": 2,
                },
                {
                    "answer_text": "The sky is green during the day.",
                    "topic_key": "sky_color:daytime",
                    "confidence_estimate": 0.82,
                    "reuse_count": 2,
                    "successful_uses": 2,
                },
                {
                    "answer_text": "Daytime sky color is blue in normal atmospheric conditions.",
                    "topic_key": "sky_color:daytime",
                    "confidence_estimate": 0.76,
                    "reuse_count": 1,
                    "successful_uses": 1,
                },
            ],
        ),
        _run_concept_cluster_competition_probe(),
        _run_stability_generalization_probe(),
        "## Initial Improvement Targets",
        "",
        "- Re-check whether the contradiction multiplier is strong enough on larger contradiction clusters beyond these two sample scenarios.",
        "- Re-check whether directness and coherence should be raised or lowered after we probe real cached project data.",
        "- Consider whether the entry-level selection graph and the persistent concept graph should eventually share more scoring signals while staying architecturally separate.",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a sample reasoning graph probe report.")
    parser.add_argument(
        "--output",
        default=str(Path("brain_output") / "reasoning_graph_probe_report.md"),
        help="Path to the markdown report file.",
    )
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote reasoning graph probe report to {output_path}")


if __name__ == "__main__":
    main()
