import copy
import json
from pathlib import Path
import tempfile
import unittest
from brain_relation_memory import RelationMemory, SourceEvent, fit_atomic_bundle

def memory(*texts,scope="test"):
    return RelationMemory([SourceEvent(i+1,text) for i,text in enumerate(texts)],scope)

class RelationMemoryTests(unittest.TestCase):
    def test_alias_chain_preserves_entity_and_latest_location(self):
        m=memory("The silver probe is in the blue cabinet.",
                 "I renamed the silver probe to Night Lantern.",
                 "Night Lantern is now in the upper rack.")
        result=m.query("Where is the silver probe now?")
        self.assertIn("upper rack",result["supported_answer"])
        self.assertEqual(m.aliases["silver probe"],m.aliases["night lantern"])
        f=m.current(m.aliases["silver probe"],"located_in")
        self.assertEqual(f.support_ids,(2,3))
        self.assertEqual({s["id"] for s in result["sources"]},{1,2,3})

    def test_rename_without_to_requires_known_old_name(self):
        m=memory("The ivory meter is in case D4.",
                 "I renamed the ivory meter Day Signal.",
                 "Day Signal is in case F6.")
        self.assertEqual(m.aliases["ivory meter"],m.aliases["day signal"])
        self.assertIn("F6",m.query("Where is the ivory meter?")["supported_answer"])
        unknown=memory("I renamed the tall silver meter Morning Beacon.")
        self.assertNotIn("beacon",unknown.aliases)
        self.assertTrue(any(i["kind"]=="ambiguous_rename" for i in unknown.issues))

    def test_alias_collision_does_not_merge_objects(self):
        m=memory("The red probe is in case A1.","The blue probe is in case B2.",
                 "I renamed the red probe to blue probe.")
        self.assertNotEqual(m.aliases["red probe"],m.aliases["blue probe"])
        self.assertTrue(any(i["kind"]=="alias_collision" for i in m.issues))
        self.assertEqual(m.query("Where is the blue probe?")["supported_answer"],"")

    def test_alias_cycle_is_finite_and_same_entity(self):
        m=memory("Aster is in the desk drawer.","I renamed Aster to Birch.",
                 "I renamed Birch to Aster.")
        self.assertEqual(m.aliases["aster"],m.aliases["birch"])
        self.assertIn("desk drawer",m.query("Where is Birch?")["supported_answer"])

    def test_unique_identifier_bridges_container_location(self):
        m=memory("The amber seal is packed in parcel Q6.",
                 "Q6 rests on the east inspection shelf.")
        result=m.query("Where is the container holding the amber seal?")
        self.assertIn("east inspection shelf",result["supported_answer"])
        self.assertEqual({s["id"] for s in result["sources"]},{1,2})

    def test_identifier_collision_requires_clarification(self):
        m=memory("Parcel Q6 is in the west room.","Case Q6 is in the east room.",
                 "Q6 is now in the archive.")
        self.assertTrue(any(i["kind"]=="ambiguous_entity" for i in m.issues))
        self.assertEqual(m.query("Where is Q6?")["supported_answer"],"")
        self.assertNotIn("archive",[f.value for f in m.facts])

    def test_owner_location_does_not_imply_item_location(self):
        m=memory("The archive badge belongs to Lera.","Lera is in the west room.")
        self.assertEqual(m.query("Where is the archive badge?")["supported_answer"],"")

    def test_explicit_universal_storage_supports_item_location(self):
        m=memory("The archive badge belongs to Lera.",
                 "All of Lera's belongings are stored in the pine locker.")
        result=m.query("Where is the archive badge?")
        self.assertIn("pine locker",result["supported_answer"])
        self.assertEqual({s["id"] for s in result["sources"]},{1,2})

    def test_transfer_reference_resolves_unique_holder_and_cites_both(self):
        m=memory("Toma handed the brass key to Niva.",
                 "Niva passed the received item to Ralo.",
                 "Toma kept the reserve key.")
        result=m.query("Who holds the brass key?")
        self.assertEqual(result["supported_answer"],"Ralo")
        fact=m.current(m.aliases["brass key"],"held_by")
        self.assertEqual(fact.support_ids,(1,2))

    def test_ambiguous_transfer_never_selects_an_arbitrary_item(self):
        m=memory("Toma handed the brass key to Niva.",
                 "Toma handed the copper badge to Niva.",
                 "Niva passed it to Ralo.")
        self.assertTrue(any(i["kind"]=="ambiguous_reference" for i in m.issues))
        self.assertIsNone(m.current(m.aliases["brass key"],"held_by"))
        self.assertIsNone(m.current(m.aliases["copper badge"],"held_by"))
        self.assertFalse(any(f.value=="Ralo" for f in m.facts))

    def test_explicit_transfer_after_ambiguity_restores_known_holder(self):
        m=memory("Toma handed the brass key to Niva.",
                 "Toma handed the copper badge to Niva.",
                 "Niva passed it to Ralo.",
                 "Niva handed the brass key to Sena.")
        self.assertEqual(m.current(m.aliases["brass key"],"held_by").value,"Sena")

    def test_same_event_explicit_transfers_follow_event_order(self):
        m=memory("Toma handed the brass key to Niva. Niva passed it to Ralo.")
        self.assertEqual(m.current(m.aliases["brass key"],"held_by").value,"Ralo")

    def test_negated_location_invalidates_old_value_without_inventing_new(self):
        m=memory("The probe is in case A1.","The probe is not in case A1.")
        self.assertIsNone(m.current(m.aliases["probe"],"located_in"))
        self.assertEqual(m.query("Where is the probe?")["supported_answer"],"")

    def test_same_event_conflict_is_not_last_statement_wins(self):
        m=memory("The probe is in case A1. The probe is in case B2.")
        self.assertIsNone(m.current(m.aliases["probe"],"located_in"))
        self.assertEqual(m.query("Where is the probe?")["supported_answer"],"")

    def test_explicit_correction_inside_event_can_advance_state(self):
        m=memory("The probe is in case A1. Correction: the probe is now in case B2.")
        self.assertEqual(m.current(m.aliases["probe"],"located_in").value,"case B2")

    def test_tentative_statements_are_not_certain_facts(self):
        m=memory("The probe might be in the archive.")
        self.assertEqual(m.facts,[])
        self.assertTrue(any(i["kind"]=="uncertain_source" for i in m.issues))

    def test_questions_do_not_become_facts(self):
        m=memory("Is the probe in the archive?")
        self.assertEqual(m.facts,[])
        self.assertEqual(m.eligible,set())

    def test_negative_prerequisite_is_not_a_prohibition(self):
        m=memory("The console does not require a network.",
                 "The exhibit must not use a network.")
        negative=next(f for f in m.facts if f.predicate=="requires")
        prohibition=next(f for f in m.facts if f.predicate=="forbids")
        self.assertFalse(negative.positive)
        self.assertTrue(prohibition.positive)

    def test_quantity_bounds_keep_strictness_and_units(self):
        m=memory("The exhibit must spend under thirty euros.",
                 "Option Cedar costs twenty-one euros.",
                 "The trip must spend no more than twenty dollars.")
        values={(f.predicate,f.value) for f in m.facts}
        self.assertIn(("max_cost","< 30 euro"),values)
        self.assertIn(("cost","21 euro"),values)
        self.assertIn(("max_cost","<= 20 dollar"),values)

    def test_location_cycle_has_no_terminal_answer(self):
        m=memory("The seal is in the parcel.","The parcel is in the seal.")
        result=m.query("Where is the seal?")
        self.assertEqual(result["supported_answer"],"")
        self.assertTrue(any(i["kind"]=="location_cycle" for i in result["issues"]))

    def test_duplicate_source_id_is_idempotent_but_not_reassignable(self):
        m=memory("The probe is in the archive.")
        before=copy.deepcopy(m.facts)
        m.add_event(1,"The probe is in the archive.")
        self.assertEqual(before,m.facts)
        with self.assertRaises(ValueError):m.add_event(1,"The probe is in the garage.")

    def test_out_of_order_replay_and_snapshot_are_consistent(self):
        chronological=memory("The probe is in the archive.","The probe is now in the garage.")
        reordered=RelationMemory([SourceEvent(2,"The probe is now in the garage."),
                                  SourceEvent(1,"The probe is in the archive.")],"test")
        self.assertEqual(chronological.facts,reordered.facts)
        snapshot=chronological.snapshot(1)
        self.assertEqual(snapshot.current(snapshot.aliases["probe"],"located_in").value,"archive")

    def test_snapshot_does_not_learn_future_alias(self):
        m=memory("The probe is in the archive.","I renamed the probe to Beacon.")
        self.assertNotIn("beacon",m.snapshot(1).aliases)

    def test_scopes_do_not_share_identity_or_state(self):
        first=memory("The probe is in the archive.",scope="alpha")
        second=memory("The probe is in the garage.",scope="beta")
        self.assertNotEqual(first.aliases["probe"],second.aliases["probe"])

    def test_event_log_roundtrip_and_corruption_detection(self):
        m=memory("The probe is in the archive.","I renamed the probe to Beacon.",
                 "Beacon is in the garage.")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"memory.json"
            m.save(path)
            loaded=RelationMemory.load(path)
            self.assertEqual(m.facts,loaded.facts)
            document=json.loads(path.read_text())
            document["events"][0]["text"]="The probe is in the attic."
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):RelationMemory.load(path)

    def test_atomic_budget_rejection_exposes_no_partial_claims(self):
        bundle={"evidence_bundle_version":1,"sources":[{"id":1,"text":"x"*500},{"id":2,"text":"y"*500}],
                "supported_answer":"secret destination","current_facts":[{"value":"secret destination"}]}
        fitted,meta=fit_atomic_bundle(bundle,lambda obj:len(json.dumps(obj)),300)
        self.assertTrue(meta["bundle_rejected"])
        self.assertNotIn("secret destination",json.dumps(fitted))
        self.assertEqual(meta["required_source_ids"],[1,2])

if __name__=="__main__": unittest.main()
