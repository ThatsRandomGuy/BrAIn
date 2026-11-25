import copy
import json
import unittest
from brain_state_extraction import explicit_locations, quantity_bounds, parse_number, placement_constraints
from brain_hybrid_prototype import BrainHybridPrototype
from brain_context_packet_v1 import _compact_memory_state

class SemanticMemoryTests(unittest.TestCase):
    def test_locations_preserve_full_target_and_correction(self):
        self.assertEqual(list(explicit_locations("The silver probe is currently in the locked supply cabinet.")),
                         [("silver probe","in","locked supply cabinet",False)])
        self.assertEqual(list(explicit_locations("Actually, the silver probe is now on the upper shelf instead of the cabinet.")),
                         [("silver probe","on","upper shelf",True)])
        self.assertEqual(list(explicit_locations("The field meter has been moved to the insulated transport case.")),
                         [("field meter","in","insulated transport case",True)])
        self.assertEqual(list(explicit_locations("We moved the field meter to the insulated transport case.")),
                         [("field meter","in","insulated transport case",True)])

    def test_questions_negation_and_unresolved_pronouns_abstain(self):
        for text in ("Where is the silver probe?", "The silver probe is not in the case.",
                     "It is in the case.", "The silver probe is in the case?"):
            with self.subTest(text=text):
                self.assertEqual(list(explicit_locations(text)),[])

    def test_quantity_bounds_support_words_digits_and_inclusive_limits(self):
        self.assertEqual(list(quantity_bounds("Keep the payload under twenty-five kilograms.")),
                         [("weight","under 25 kilograms")])
        self.assertEqual(list(quantity_bounds("The setup takes at most twelve minutes.")),
                         [("minute","at most 12 minutes")])
        self.assertEqual(list(quantity_bounds("Keep the parcel below 2.5 kg.")),
                         [("weight","below 2.5 kilograms")])
        self.assertEqual(list(quantity_bounds("Is the parcel under nine kilograms?")),[])
        self.assertIsNone(parse_number("seven bananas"))
        self.assertIsNone(parse_number("twenty thirty"))
        self.assertEqual(parse_number("one hundred and six"),"106")

    def test_attached_quantity_units_preserve_legacy_syntax(self):
        self.assertEqual(list(quantity_bounds("Keep it under 7kilograms.")),
                         [("weight","under 7 kilograms")])
        self.assertEqual(list(quantity_bounds("Keep it under 2.5kg.")),
                         [("weight","under 2.5 kilograms")])

    def test_placement_constraints_preserve_full_phrases(self):
        self.assertEqual(list(placement_constraints("The tablet must remain in the padded outer sleeve.")),
                         [("tablet","tablet in padded outer sleeve")])

    def test_observation_keys_match_across_location_updates(self):
        extractor=BrainHybridPrototype.__new__(BrainHybridPrototype)
        first=extractor._extract_state_observations("The silver probe is in the supply cabinet.")
        last=extractor._extract_state_observations("Correction: the silver probe is now on the upper shelf.")
        first=next(o for o in first if o["state_type"]=="object_location")
        last=next(o for o in last if o["state_type"]=="object_location")
        self.assertEqual(first["state_key"],last["state_key"])
        self.assertEqual(last["value"],"upper shelf")
        self.assertTrue(last["is_update"])

    def test_packet_retains_facts_without_nested_pixel_diagnostics(self):
        state={"items":[{"state_key":"object_location:probe","state_type":"object_location",
                        "current_value":"upper shelf","original_value":"cabinet","display_text":"probe is on upper shelf"}],
            "wants_original_state":True,
            "reconstructed_context":{"synthesized_summary":"probe is on upper shelf","confidence":0.8,"answer_mode":"high",
                "partial_matches":[{"text":"probe is on upper shelf","source_id":2,"source_type":"state_memory",
                                    "payload":{"pixel_readback":{"heatmap_lines":["x"*10000]}}},
                                   {"text":"Probe is on upper shelf.","source_id":2,"source_type":"pixel_readback"}],
                "evidence_breakdown":{"pixel_readback":{"debug_log":["trace"]}}}}
        saved=copy.deepcopy(state)
        compact=_compact_memory_state(state)
        self.assertEqual(state,saved)
        self.assertEqual(compact["items"][0]["original_value"],"cabinet")
        self.assertEqual(compact["items"][0]["current_value"],"upper shelf")
        fragments=compact["reconstructed_context"]["partial_matches"]
        self.assertEqual(len(fragments),1)
        self.assertNotIn("payload",fragments[0])
        self.assertNotIn("evidence_breakdown",compact["reconstructed_context"])

if __name__=="__main__":
    unittest.main()
