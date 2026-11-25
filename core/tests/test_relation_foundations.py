import contextlib, io, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from brain_relation_memory import RelationMemory, SourceEvent, context_from_turns
from brain_hybrid_prototype import BrainHybridPrototype, StoragePolicy
from brain_query_router import QueryRouteDecision

class FoundationGuards(unittest.TestCase):
    def memory(self,*texts):
        return RelationMemory([SourceEvent(i+1,t) for i,t in enumerate(texts)],"guard")
    def test_location_pronoun_is_unresolved(self):
        m=self.memory("It is in the attic.")
        self.assertEqual(m.facts,[])
        self.assertNotIn("it",m.aliases)
    def test_indefinite_objects_not_merged(self):
        m=self.memory("A probe is in the attic.","Another probe is in the basement.")
        self.assertEqual(m.query("Where is the probe?")["supported_answer"],"")
        self.assertIn("probe",m.ambiguous_labels)
    def test_move_from_updates_destination(self):
        m=self.memory("We moved the probe from the attic to the basement.")
        self.assertEqual(m.current(m.aliases["probe"],"located_in").value,"basement")
    def test_required_placement_not_actual_location(self):
        m=self.memory("The probe must remain in the padded sleeve.")
        self.assertIsNone(m.current(m.aliases["probe"],"located_in"))
        self.assertEqual(m.query("Where is the probe?")["supported_answer"],"")
        self.assertTrue(any(f.predicate=="required_location" for f in m.facts))
    def test_generated_answers_are_not_source_authority(self):
        records=[SimpleNamespace(turn_id=1,route_type="memory_statement",
            raw_user_prompt="The probe is in the attic.",final_answer="The probe is in the basement.",
            metadata={"state":"The probe is in the basement."}),
            SimpleNamespace(turn_id=2,route_type="memory_question",raw_user_prompt="Is the probe in the basement?")]
        result=context_from_turns(records,"Where is the probe?")
        self.assertIn("attic",result["supported_answer"])
        self.assertNotIn("basement",str(result))
    def test_invalid_source_history_abstains(self):
        records=[SimpleNamespace(turn_id=1,route_type="memory_statement",raw_user_prompt=t)
            for t in ("The probe is in the attic.","The probe is in the basement.")]
        result=context_from_turns(records,"Where is the probe?")
        self.assertEqual(result["answer_status"],"unresolved")
        self.assertEqual(result["current_facts"],[])
    def test_core_restart_rebuilds_aliases_from_user_events(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            kwargs=dict(state_dir=Path(tmp),storage_policy=StoragePolicy(mode="store_all_prompts"),
                llm_provider="deterministic",bootstrap_reasoning=False,pixel_memory_enabled=False)
            brain=BrainHybridPrototype(**kwargs)
            for text in ("The copper gauge is in the attic.","I renamed the copper gauge to Dawn Marker.",
                         "Dawn Marker is now in the basement."):
                brain.process_user_turn(text,inspect=True)
            restarted=BrainHybridPrototype(**kwargs)
            route=QueryRouteDecision(route_type="memory_question",confidence=1.0,reasons=["test"],
                run_brain_reasoning=False,run_direct_llm=False,run_hybrid_mode=False,score_breakdown={})
            result=restarted.run_prompt("Where is the copper gauge now?",inspect=True,route_override=route)
            ctx=result.trace.merged_memory_state["relation_memory"]
            self.assertIn("basement",ctx["supported_answer"])
            self.assertEqual({s["id"] for s in ctx["sources"]},{1,2,3})
            self.assertIn("basement",result.llm_explanation.user_facing_text)

    def test_ownership_and_custody_are_distinct(self):
        m=self.memory("The copper gauge belongs to Iona.","Iona handed the copper gauge to Ravi.")
        self.assertEqual(m.query("Who owns the copper gauge?")["supported_answer"],"Iona")
        self.assertEqual(m.query("Who holds the copper gauge?")["supported_answer"],"Ravi")
        owner_only=self.memory("The copper gauge belongs to Iona.")
        self.assertEqual(owner_only.query("Who holds the copper gauge?")["supported_answer"],"")
