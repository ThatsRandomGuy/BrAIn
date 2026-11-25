import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import brain_pixel_codec as p
class PixelCodecTests(unittest.TestCase):
    def doc(self):return {"version":1,"scope":"test","events":[{"source_id":1,"text":"The cobalt lens is in the linen pouch."},{"source_id":2,"text":"I put the cobalt lens into the brass case."}]}
    def test_exact_unicode_roundtrip(self):
        doc=self.doc();doc["unicode"]=chr(0x1f511)+chr(0x6f22)+chr(0xe9)+" quoted "+chr(34)
        png,legend=p.encode(doc);self.assertEqual(p.decode(png,legend),doc)
    def test_deterministic_token_only_dictionary(self):
        png,legend=p.encode(self.doc());self.assertEqual((png,legend),p.encode(self.doc()))
        self.assertTrue(all(p.TOKEN.fullmatch(t) for t in legend["colors"].values()))
        self.assertNotIn("The cobalt lens is in the linen pouch.",json.dumps(legend))
        self.assertEqual(list(legend["colors"]),sorted(legend["colors"]))
    def test_only_two_files_and_total_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/"pixels";metrics=p.save(self.doc(),folder)
            self.assertEqual({f.name for f in folder.iterdir()},{"memory.png","legend.json"})
            self.assertEqual(p.load(folder),self.doc())
            self.assertEqual(metrics["total_bytes"],sum(f.stat().st_size for f in folder.iterdir()))
    def test_legend_cannot_replace_erased_image(self):
        png,legend=p.encode(self.doc());w,h,rgb=p.read_rgb_png(png)
        with self.assertRaises(ValueError):p.decode(p.rgb_png(w,h,bytes(len(rgb))),legend)
    def test_reordered_pixels_fail_with_unchanged_vocabulary(self):
        png,legend=p.encode(self.doc());w,h,rgb=p.read_rgb_png(png);n=legend["token_count"]
        pixels=[rgb[i:i+3] for i in range(0,n*3,3)]
        with self.assertRaises(ValueError):p.decode(p.rgb_png(w,h,b"".join(reversed(pixels))+rgb[n*3:]),legend)
    def test_modified_dictionary_fails(self):
        png,legend=p.encode(self.doc());other=copy.deepcopy(legend)
        key=next(k for k,v in other["colors"].items() if v=="cobalt");other["colors"][key]="amber"
        with self.assertRaises(ValueError):p.decode(png,other)
    def test_bad_schema_or_full_record_dictionary_fails(self):
        png,legend=p.encode(self.doc())
        for key,value in (("version",True),("version",2),("token_count",0)):
            other=copy.deepcopy(legend);other[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):p.decode(png,other)
        other=copy.deepcopy(legend);other["colors"][next(iter(other["colors"]))]="hidden source record"
        with self.assertRaises(ValueError):p.decode(png,other)
    def test_png_integrity_and_truncation(self):
        png,legend=p.encode(self.doc())
        for bad in (png[:-1],png+b"extra",png[:40]+bytes([png[40]^1])+png[41:]):
            with self.subTest(length=len(bad)),self.assertRaises(ValueError):p.decode(bad,legend)
    def test_missing_files_never_fall_back(self):
        for missing in ("memory.png","legend.json"):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)/"pixels";p.save(self.doc(),folder);(folder/missing).unlink()
                with self.assertRaises(OSError):p.load(folder)
    def test_color_collision_resolution(self):
        original=p.hashlib.sha256
        class Collision:
            def __init__(self,data):self.real=original(data)
            def digest(self):return bytes(32)
            def hexdigest(self):return self.real.hexdigest()
        with patch.object(p.hashlib,"sha256",side_effect=Collision):
            png,legend=p.encode(self.doc());self.assertEqual(p.decode(png,legend),self.doc())
        self.assertEqual(len(set(legend["colors"].values())),len(legend["colors"]))

    def test_small_file_load_does_not_reserve_maximum_file_capacity(self):
        import tracemalloc
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/"pixels";p.save(self.doc(),folder)
            tracemalloc.start()
            try:
                result=p.load(folder);_,peak=tracemalloc.get_traced_memory()
            finally:tracemalloc.stop()
            self.assertEqual(result,self.doc())
            self.assertLess(peak,2_000_000)
