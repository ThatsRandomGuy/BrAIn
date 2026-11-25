import contextlib,io,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
from brain_relation_memory import RelationMemory,SourceEvent
from brain_hybrid_prototype import BrainHybridPrototype,StoragePolicy
from brain_query_router import QueryRouteDecision
from llm_interface import LLMExplanationResult

def memory(*texts):return RelationMemory([SourceEvent(i+1,t) for i,t in enumerate(texts)])

class UpdateSafetyTests(unittest.TestCase):
    def test_common_placements_replace_location(self):
        for report in ("I put the jade compass into the cloth pouch.","We placed the jade compass on the upper ledge.",
                       "Lena tucked the jade compass inside the cloth pouch.","The jade compass was inserted into the cloth pouch."):
            with self.subTest(report=report):
                m=memory("The jade compass is in the old case.",report)
                result=m.query("Where is the jade compass now?")
                self.assertTrue(result["supported_answer"])
                self.assertNotIn("old case",result["supported_answer"])
                fact=m.current(m.aliases["jade compass"],"located_in")
                self.assertEqual(fact.source_id,2)
    def test_unparsed_move_retires_old_location_but_not_other_objects(self):
        m=memory("The jade compass is in the old case.","The blue ruler is in the tall drawer.",
                 "I carted the jade compass somewhere else.")
        result=m.query("Where is the jade compass?")
        self.assertEqual(result["answer_status"],"unresolved")
        self.assertFalse(result["supported_answer"])
        self.assertIsNone(m.current(m.aliases["jade compass"],"located_in"))
        self.assertIn(3,{s["id"] for s in result["sources"]})
        self.assertIn("tall drawer",m.query("Where is the blue ruler?")["supported_answer"])
    def test_later_explicit_location_restores_certainty(self):
        m=memory("The jade compass is in the old case.","I carted the jade compass somewhere else.",
                 "The jade compass is now in the new cabinet.")
        self.assertIn("new cabinet",m.query("Where is the jade compass?")["supported_answer"])
        self.assertFalse(m.query("Where is the jade compass?")["issues"])
        self.assertFalse(m.snapshot(2).query("Where is the jade compass?")["supported_answer"])
    def test_future_negation_and_commands_do_not_change_actual_state(self):
        for report in ("I will put the jade compass in the cloth pouch.","I did not put the jade compass in the cloth pouch.",
                       "I never stored the jade compass in the cloth pouch.","Tomorrow I will move the jade compass.",
                       "Please put the jade compass in the cloth pouch.","If I moved the jade compass to the pouch, it would fit."):
            with self.subTest(report=report):
                m=memory("The jade compass is in the old case.",report)
                self.assertEqual(m.current(m.aliases["jade compass"],"located_in").value,"old case")
    def test_unresolved_pronoun_move_cannot_keep_old_locations_certain(self):
        m=memory("The jade compass is in the old case.","The blue ruler is in the tall drawer.","I put it somewhere else.")
        self.assertFalse(m.query("Where is the jade compass?")["supported_answer"])
        self.assertFalse(m.query("Where is the blue ruler?")["supported_answer"])

class NativeRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.output=contextlib.redirect_stdout(io.StringIO());self.output.__enter__();self.addCleanup(self.output.__exit__,None,None,None)
        self.brain=BrainHybridPrototype(state_dir=Path(self.tmp.name),storage_policy=StoragePolicy(mode="store_all_prompts"),
            llm_provider="deterministic",bootstrap_reasoning=False,pixel_memory_enabled=False)
    def seed(self,*texts):
        for text in texts:self.brain.process_user_turn(text,inspect=True)
    def test_ordinary_question_uses_history_and_new_location(self):
        self.seed("The jade compass is in the old case.","I put the jade compass into the cloth pouch.")
        result=self.brain.process_user_turn("Where is the jade compass now?",inspect=True)
        self.assertEqual(result.route_decision.route_type,"memory_question")
        self.assertIn("cloth pouch",result.llm_explanation.user_facing_text)
        self.assertEqual(result.trace.merged_memory_state["relation_memory"]["supported_answer"],"jade compass is in cloth pouch.")
    def test_unrelated_and_definitional_questions_remain_factual(self):
        self.seed("The jade compass is in the old case.")
        for question in ("What is the capital of Peru?","How does a compass work?","What is a jade compass?"):
            with self.subTest(question=question):
                self.assertEqual(self.brain.run_prompt(question).route_decision.route_type,"factual_question")
    def test_empty_history_does_not_manufacture_context(self):
        self.assertEqual(self.brain.run_prompt("Where is the jade compass?").route_decision.route_type,"factual_question")
    def test_ambiguous_followup_abstains(self):
        self.seed("The jade compass is in the old case.","The blue ruler is in the tall drawer.")
        result=self.brain.run_prompt("Where is it now?")
        self.assertEqual(result.route_decision.route_type,"memory_question")
        self.assertEqual(result.llm_explanation.user_facing_text,"UNKNOWN")
    def test_tentative_subject_still_routes_to_memory(self):
        self.seed("The jade compass might be in the old case.")
        result=self.brain.run_prompt("Where is the jade compass?")
        self.assertEqual(result.route_decision.route_type,"memory_question")
        self.assertEqual(result.llm_explanation.user_facing_text,"UNKNOWN")
    def test_explicit_override_is_respected(self):
        self.seed("The jade compass is in the old case.")
        override=QueryRouteDecision(route_type="factual_question",confidence=1,reasons=["explicit"],
            run_brain_reasoning=False,run_direct_llm=True,run_hybrid_mode=False,score_breakdown={})
        self.assertEqual(self.brain.run_prompt("Where is the jade compass?",route_override=override).route_decision.route_type,"factual_question")
    def test_supported_answer_bypasses_configured_provider(self):
        self.seed("The jade compass is in the old case.","I put the jade compass into the cloth pouch.")
        provider=Mock(provider_name="must_not_guess");provider.answer.side_effect=AssertionError("Provider must not override verified state")
        self.brain.llm.provider=provider
        result=self.brain.run_prompt("Where is the jade compass now?")
        provider.answer.assert_not_called()
        self.assertIn("cloth pouch",result.llm_explanation.user_facing_text)
        self.assertTrue(result.llm_explanation.metadata["relation_authority_enforced"])
    def test_unresolved_state_bypasses_provider_and_returns_unknown(self):
        self.seed("The jade compass is in the old case.","I carted the jade compass somewhere else.")
        provider=Mock(provider_name="must_not_guess");provider.answer.side_effect=AssertionError("Provider must not fill unresolved state")
        self.brain.llm.provider=provider
        result=self.brain.run_prompt("Where is the jade compass?")
        provider.answer.assert_not_called()
        self.assertEqual(result.llm_explanation.user_facing_text,"UNKNOWN")
        self.assertEqual(result.trace.llm_payload_v2["answer_facts"],["UNKNOWN"])
    def test_interpretation_still_reaches_provider(self):
        self.seed("The exhibit must use battery power.","The display works from a battery.")
        provider=Mock(provider_name="interpreter")
        provider.answer.return_value=LLMExplanationResult(provider="interpreter",structured_text="",reasoning_explanation="",user_facing_text="display",messages=[],metadata={})
        self.brain.llm.provider=provider
        result=self.brain.run_prompt("Which device satisfies the exhibit rules?")
        self.assertEqual(result.route_decision.route_type,"memory_question")
        provider.answer.assert_called_once()
        self.assertEqual(provider.answer.call_args[0][0]["llm_payload_v2"]["source_of_truth"],"llm")
