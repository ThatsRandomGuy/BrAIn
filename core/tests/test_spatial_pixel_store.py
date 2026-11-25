import tempfile,unittest
from pathlib import Path
import brain_spatial_pixel_store as p
class SpatialPixelTests(unittest.TestCase):
    def events(self,*texts):return [{"source_id":i+1,"text":t} for i,t in enumerate(texts)]
    def pair(self,events,question,**limits):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/"pixels";p.save(events,folder)
            pixel=p.PixelStore(folder);self.assertEqual(pixel.stats()["text_rows_materialized"],0)
            a=p.retrieve(p.StructuredStore(events),question,**limits);b=p.retrieve(pixel,question,**limits)
            self.assertEqual(a["context"],b["context"]);self.assertEqual(a["selected_ids"],b["selected_ids"])
            return b
    def test_color_lookup_matches_structured_index_and_only_decodes_selected(self):
        events=self.events("The amber pin is in the blue cup.",*(f"Archive sample {i} remained unchanged." for i in range(32)))
        result=self.pair(events,"Where is the amber pin?")
        self.assertIn("blue cup",result["supported_answer"])
        self.assertLess(result["stats"]["text_rows_materialized"],len(events))
    def test_multi_hop_and_alias_expansion(self):
        result=self.pair(self.events("The amber pin is in parcel X9.","X9 is in the blue cupboard.","I renamed the amber pin to Night Dial."),"Where is Night Dial?")
        self.assertTrue(result["complete"]);self.assertIn("blue cupboard",result["supported_answer"])
    def test_custody_received_item_is_reached_through_holder(self):
        result=self.pair(self.events("Ivo handed the amber pin to Tera.","Tera passed the received item to Oren."),"Who holds the amber pin?")
        self.assertEqual(result["supported_answer"],"Oren")
    def test_unbound_pronoun_change_is_not_dropped(self):
        result=self.pair(self.events("The amber pin is in the blue cup.","I carted it somewhere else."),"Where is the amber pin?")
        self.assertEqual(result["context"]["answer_status"],"unresolved")
        self.assertFalse(result["supported_answer"]);self.assertIn(2,result["selected_ids"])
    def test_later_explicit_observation_recovers(self):
        result=self.pair(self.events("The amber pin is in the blue cup.","I carted it somewhere else.","The amber pin is now in the red basket."),"Where is the amber pin?")
        self.assertIn("red basket",result["supported_answer"])
    def test_limits_never_expose_a_partial_supported_state(self):
        for limits in ({"max_rounds":1},{"max_rows":1}):
            with self.subTest(limits=limits):
                result=self.pair(self.events("The amber pin is in parcel X9.","X9 is in the blue cupboard."),"Where is the amber pin?",**limits)
                self.assertFalse(result["complete"]);self.assertFalse(result["supported_answer"])
                self.assertNotIn("current_facts",result["context"])
    def test_row_overflow_is_rejected_not_truncated(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):p.save(self.events("word "*100),Path(tmp)/"pixels")
    def test_changed_or_missing_assets_fail_closed(self):
        for name in ("memory.png","legend.json","store.json"):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)/"pixels";p.save(self.events("The amber pin is in the blue cup."),folder)
                (folder/name).write_bytes(b"bad")
                with self.assertRaises((ValueError,OSError)):p.PixelStore(folder)
    def test_structured_scan_and_index_are_equivalent(self):
        events=self.events("The amber pin is in the blue cup.","The tall ruler is in the red basket.")
        a=p.retrieve(p.StructuredStore(events,indexed=True),"Where is the amber pin?")
        b=p.retrieve(p.StructuredStore(events,indexed=False),"Where is the amber pin?")
        self.assertEqual(a["context"],b["context"])

    def test_invalid_source_ids_and_record_shapes_are_rejected(self):
        for events in ([{"source_id":True,"text":"A source."}],[{"source_id":2,"text":"A source."}],[{"source_id":1,"text":" "}],["not a record"]):
            with self.subTest(events=events),tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):p.save(events,Path(tmp)/"pixels")
