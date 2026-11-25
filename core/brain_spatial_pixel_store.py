"""Addressable source rows; color postings select records before lexical decoding.

All PNG bytes are still read/decompressed/validated. Geometry is chronological;
this is indexed selective readback, not a vision model or semantic color layout.
"""
from __future__ import annotations
import hashlib,json,re
from collections import defaultdict
from pathlib import Path
import brain_pixel_codec as codec
from brain_relation_memory import RelationMemory,words
CAPACITY=128
STOP={"currently","holds","hold","stored","ultimately","compatible","device","rules","rule","restrictions","restriction","satisfies","requirements","constraints"}
CHANGE={"moved","relocated","put","placed","carted","carried","removed","handed","passed","gave","transferred","changed","stashed","tucked"}
UNCERTAIN={"maybe","might","perhaps","probably","possibly"}
REFERENCE={"it","them","received"}
FEATURE={"must","requires","needs","costs","crosses","avoids","works","dependency"}
def raw_terms(text):return set(re.findall(r"[a-z0-9]+",str(text).casefold()))
def terms(text):return words(text)-STOP
def digest(data):return hashlib.sha256(data).hexdigest()
def unavailable(reason):return {"evidence_status":"unavailable","policy":"Complete relevant evidence could not be established. Answer UNKNOWN.","reason":reason}

def save(events,folder):
    if not events or len(events)>4096:raise ValueError("Unsupported row count")
    if any(not isinstance(e,dict) or set(e)!={"source_id","text"} or type(e["source_id"]) is not int or e["source_id"]!=i+1 or not isinstance(e["text"],str) or not e["text"].strip() for i,e in enumerate(events)):raise ValueError("Require contiguous source records")
    tokens=[codec.TOKEN.findall(e["text"]) for e in events]
    if any(len(row)>CAPACITY for row in tokens):raise ValueError("Source exceeds row capacity; never truncate")
    colors={};inverse={}
    for token in sorted({t for row in tokens for t in row}):
        code=int.from_bytes(hashlib.sha256(token.encode()).digest()[:3],"big") or 1
        while code in inverse:code=code%0xffffff+1
        colors[token]=code;inverse[code]=token
    rgb=b"".join(len(row).to_bytes(3,"big")+b"".join(colors[t].to_bytes(3,"big") for t in row)+bytes((CAPACITY-len(row))*3) for row in tokens)
    png=codec.rgb_png(CAPACITY+1,len(tokens),rgb)
    legend=codec.canonical_bytes({"version":2,"colors":{f"{k:06x}":v for k,v in sorted(inverse.items())}})
    descriptor=codec.canonical_bytes({"version":2,"capacity":CAPACITY,"rows":len(tokens),"image_sha256":digest(png),"legend_sha256":digest(legend)})
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    for name,data in (("memory.png",png),("legend.json",legend),("store.json",descriptor)):(folder/name).write_bytes(data)
    return {"png_bytes":len(png),"legend_bytes":len(legend),"descriptor_bytes":len(descriptor),"total_bytes":len(png)+len(legend)+len(descriptor),"raw_rgb_bytes":len(rgb),"word_pixels":sum(map(len,tokens)),"capacity_pixels":len(tokens)*(CAPACITY+1)}

class StructuredStore:
    def __init__(self,events,indexed=True):
        self.events={e["source_id"]:e["text"] for e in events};self.rows=set(self.events)
        self.row_terms={sid:raw_terms(text) for sid,text in self.events.items()};self.indexed=indexed
        self.postings=defaultdict(set);self.decoded=set();self.total_source_tokens=sum(len(codec.TOKEN.findall(t)) for t in self.events.values())
        if indexed:
            for sid,items in self.row_terms.items():
                for word in items:self.postings[word].add(sid)
    def match(self,items):
        if self.indexed:return set().union(*(self.postings.get(w,set()) for w in items)) if items else set()
        return {sid for sid,tokens in self.row_terms.items() if tokens&items}
    def get(self,sid):self.decoded.add(sid);return {"source_id":sid,"text":self.events[sid]}
    def stats(self):return {"text_rows_materialized":len(self.events),"selected_rows":len(self.decoded),"text_tokens_materialized":self.total_source_tokens}

class PixelStore:
    def __init__(self,folder,indexed=True):
        folder=Path(folder)
        descriptor=json.loads(codec._bounded_read(folder/"store.json"));png=codec._bounded_read(folder/"memory.png");legend_bytes=codec._bounded_read(folder/"legend.json")
        if set(descriptor)!={"version","capacity","rows","image_sha256","legend_sha256"} or type(descriptor["version"]) is not int or descriptor["version"]!=2 or descriptor["capacity"]!=CAPACITY:raise ValueError("Invalid spatial descriptor")
        if digest(png)!=descriptor["image_sha256"] or digest(legend_bytes)!=descriptor["legend_sha256"]:raise ValueError("Spatial store integrity failure")
        legend=json.loads(legend_bytes)
        if set(legend)!={"version","colors"} or type(legend["version"]) is not int or legend["version"]!=2 or not isinstance(legend["colors"],dict):raise ValueError("Invalid color legend")
        self.inverse={};self.word_colors=defaultdict(set)
        for color,token in legend["colors"].items():
            if not re.fullmatch(r"[0-9a-f]{6}",color) or color=="000000" or not isinstance(token,str) or codec.TOKEN.fullmatch(token) is None:raise ValueError("Invalid lexical palette")
            code=int(color,16);self.inverse[code]=token
            for word in raw_terms(token):self.word_colors[word].add(code)
        if len(set(self.inverse.values()))!=len(self.inverse):raise ValueError("Duplicate token")
        width,height,rgb=codec.read_rgb_png(png)
        if width!=CAPACITY+1 or type(descriptor["rows"]) is not int or height!=descriptor["rows"]:raise ValueError("Image dimensions disagree")
        self.rgb=rgb;self.rows=set(range(1,height+1));self.counts={};self.postings=defaultdict(set);self.cache={};self.indexed=indexed;used=set()
        for sid in sorted(self.rows):
            offset=(sid-1)*width*3;count=int.from_bytes(rgb[offset:offset+3],"big")
            if not 1<=count<=CAPACITY:raise ValueError("Invalid row length")
            self.counts[sid]=count
            for pos in range(CAPACITY):
                code=int.from_bytes(rgb[offset+3+pos*3:offset+6+pos*3],"big")
                if pos>=count:
                    if code:raise ValueError("Invalid row padding")
                elif code not in self.inverse:raise ValueError("Unknown word color")
                else:
                    if indexed:self.postings[code].add(sid)
                    used.add(code)
        if used!=self.inverse.keys():raise ValueError("Unused color vocabulary")
    def match(self,items):
        if not self.indexed:raise RuntimeError("Color index not built")
        colors=set().union(*(self.word_colors.get(w,set()) for w in items)) if items else set()
        return set().union(*(self.postings.get(c,set()) for c in colors)) if colors else set()
    def get(self,sid):
        if sid not in self.cache:
            offset=(sid-1)*(CAPACITY+1)*3+3
            text="".join(self.inverse[int.from_bytes(self.rgb[offset+3*i:offset+3*i+3],"big")] for i in range(self.counts[sid]))
            self.cache[sid]={"source_id":sid,"text":text}
        return self.cache[sid]
    def stats(self):return {"text_rows_materialized":len(self.cache),"selected_rows":len(self.cache),"text_tokens_materialized":sum(self.counts[sid] for sid in self.cache),"total_word_pixels":sum(self.counts.values()),"rgb_bytes_read":len(self.rgb),"palette_tokens":len(self.inverse)}

def retrieve(store,question,max_rounds=8,max_rows=64):
    if max_rounds<1 or max_rows<1:raise ValueError("Invalid retrieval limits")
    # Global guards keep unresolved references/tentative reports from silently
    # leaving an earlier named state authoritative in a selectively decoded subset.
    safety=(store.match(REFERENCE)&store.match(CHANGE))|store.match(UNCERTAIN)
    if re.search(r"\b(compatible|satisfies|rules|restrictions|constraints|requirements)\b",question,re.I):safety|=store.match(FEATURE)
    search=terms(question);selected=set();index=None;completed=False
    for round_number in range(1,max_rounds+1):
        found=store.match(search)|safety|selected
        if len(found)>max_rows:return {"context":unavailable("selected-row limit exceeded"),"complete":False,"rounds":round_number,"selected_ids":sorted(selected),"safety_ids":sorted(safety),"supported_answer":"","stats":store.stats()}
        if found==selected:
            completed=True;break
        selected=found
        index=RelationMemory([store.get(sid) for sid in sorted(selected)],"spatial_pixels")
        expansion=set()
        for alias in index.aliases:expansion|=terms(alias)
        for fact in index.facts:
            if fact.predicate in {"requires","forbids","operates_from","independent_of","uses"}:expansion|=terms(fact.value)
        search|=expansion
    if not completed:return {"context":unavailable("closure round limit exceeded"),"complete":False,"rounds":max_rounds,"selected_ids":sorted(selected),"safety_ids":sorted(safety),"supported_answer":"","stats":store.stats()}
    view=index.query(question) if index is not None else {"active":False}
    answer=view.pop("supported_answer","")
    if not view.get("active"):view=unavailable("no matching recorded evidence")
    return {"context":view,"complete":True,"rounds":round_number,"selected_ids":sorted(selected),"safety_ids":sorted(safety),"supported_answer":answer,"stats":store.stats()}
