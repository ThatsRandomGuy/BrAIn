"""Versioned lossless word/color transport. No retained facts or activation metadata.

This deliberately small PNG reader accepts only this writer's RGB8/filter-0 format.
The inverse vocabulary is required and counted as storage; checksums detect damage,
not malicious editing. Decoding is software, not a vision model.
"""
from __future__ import annotations
import hashlib,json,math,os,re,struct,zlib
from pathlib import Path
VERSION=1
TOKEN=re.compile(r"\w+|[^\w]",re.UNICODE)
SIGNATURE=b"\x89PNG\r\n\x1a\n"
MAX_PIXELS=8_000_000
MAX_DOCUMENT=8_000_000
MAX_FILE=32_000_000

def canonical_bytes(document):
    return json.dumps(document,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode("utf-8")

def _chunk(kind,payload):
    return struct.pack(">I",len(payload))+kind+payload+struct.pack(">I",zlib.crc32(kind+payload)&0xffffffff)

def rgb_png(width,height,rgb):
    if type(width) is not int or type(height) is not int or not 1<=width<=4096 or not 1<=height<=4096 or width*height>MAX_PIXELS:
        raise ValueError("Unsupported image dimensions")
    if len(rgb)!=width*height*3:raise ValueError("RGB byte count mismatch")
    scan=b"".join(b"\x00"+rgb[y*width*3:(y+1)*width*3] for y in range(height))
    return SIGNATURE+_chunk(b"IHDR",struct.pack(">IIBBBBB",width,height,8,2,0,0,0))+_chunk(b"IDAT",zlib.compress(scan,9))+_chunk(b"IEND",b"")

def read_rgb_png(data):
    if len(data)>MAX_FILE or not data.startswith(SIGNATURE):raise ValueError("Invalid PNG signature or size")
    offset=8;header=None;compressed=[];ended=False
    while offset<len(data):
        if offset+12>len(data):raise ValueError("Truncated PNG chunk")
        length=struct.unpack(">I",data[offset:offset+4])[0];kind=data[offset+4:offset+8]
        end=offset+12+length
        if end>len(data):raise ValueError("Truncated PNG data")
        payload=data[offset+8:end-4];crc=struct.unpack(">I",data[end-4:end])[0]
        if zlib.crc32(kind+payload)&0xffffffff!=crc:raise ValueError("PNG checksum mismatch")
        if kind==b"IHDR":
            if header is not None or offset!=8 or len(payload)!=13:raise ValueError("Invalid PNG header")
            header=struct.unpack(">IIBBBBB",payload)
        elif kind==b"IDAT":
            if header is None:raise ValueError("Image data without header")
            compressed.append(payload)
        elif kind==b"IEND":
            if payload or end!=len(data):raise ValueError("Trailing PNG data")
            ended=True;break
        else:raise ValueError("Unsupported PNG chunk")
        offset=end
    if header is None or not ended or not compressed:raise ValueError("Incomplete PNG")
    width,height,*flags=header
    if flags!=[8,2,0,0,0] or not 1<=width<=4096 or not 1<=height<=4096 or width*height>MAX_PIXELS:
        raise ValueError("Unsupported PNG format")
    expected=height*(1+width*3);decoder=zlib.decompressobj()
    try:scan=decoder.decompress(b"".join(compressed),expected+1)
    except zlib.error as exc:raise ValueError("Invalid compressed pixels") from exc
    if len(scan)!=expected or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:raise ValueError("Invalid decompressed image size")
    stride=width*3+1
    if any(scan[y*stride]!=0 for y in range(height)):raise ValueError("Unsupported PNG filter")
    return width,height,b"".join(scan[y*stride+1:(y+1)*stride] for y in range(height))

def encode(document,width=64):
    source=canonical_bytes(document)
    if len(source)>MAX_DOCUMENT:raise ValueError("Memory document too large")
    tokens=TOKEN.findall(source.decode("utf-8"));vocabulary=sorted(set(tokens));colors={};inverse={}
    if len(vocabulary)>0xffffff:raise ValueError("Color vocabulary exhausted")
    for token in vocabulary:
        code=int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:3],"big") or 1
        while code in inverse:code=code%0xffffff+1
        colors[token]=code;inverse[code]=token
    if type(width) is not int or not 1<=width<=4096:raise ValueError("Invalid image width")
    height=max(1,math.ceil(len(tokens)/width))
    rgb=b"".join(colors[token].to_bytes(3,"big") for token in tokens)+bytes((width*height-len(tokens))*3)
    legend={"version":VERSION,"width":width,"height":height,"token_count":len(tokens),
            "content_sha256":hashlib.sha256(source).hexdigest(),
            "colors":{f"{code:06x}":token for code,token in sorted(inverse.items())}}
    if len(canonical_bytes(legend))>MAX_FILE:raise ValueError("Legend too large")
    png=rgb_png(width,height,rgb)
    return png,legend

def decode(png,legend):
    if not isinstance(legend,dict) or set(legend)!={"version","width","height","token_count","content_sha256","colors"} or type(legend["version"]) is not int or legend["version"]!=VERSION:
        raise ValueError("Invalid pixel legend schema/version")
    width,height,rgb=read_rgb_png(png)
    if any(type(legend[k]) is not int for k in ("width","height","token_count")) or (width,height)!=(legend["width"],legend["height"]):raise ValueError("Image/legend dimensions disagree")
    count=legend["token_count"]
    if not 1<=count<=width*height or not isinstance(legend["colors"],dict):raise ValueError("Invalid token count/vocabulary")
    inverse={}
    for color,token in legend["colors"].items():
        if not isinstance(color,str) or not re.fullmatch(r"[0-9a-f]{6}",color) or color=="000000":raise ValueError("Invalid color key")
        if not isinstance(token,str) or not token or len(token)>MAX_DOCUMENT or TOKEN.fullmatch(token) is None:raise ValueError("Legend must contain lexical tokens, not source records")
        inverse[int(color,16)]=token
    if len(set(inverse.values()))!=len(inverse):raise ValueError("Duplicate vocabulary token")
    recovered=[];used=set()
    for i in range(width*height):
        code=int.from_bytes(rgb[3*i:3*i+3],"big")
        if i>=count:
            if code:raise ValueError("Nonzero image padding")
        else:
            if code not in inverse:raise ValueError("Unknown or erased word color")
            recovered.append(inverse[code]);used.add(code)
    if used!=inverse.keys():raise ValueError("Unused legend entry")
    text="".join(recovered);source=text.encode("utf-8")
    if len(source)>MAX_DOCUMENT or hashlib.sha256(source).hexdigest()!=legend["content_sha256"]:raise ValueError("Recovered memory checksum mismatch")
    try:document=json.loads(text)
    except (ValueError,TypeError) as exc:raise ValueError("Recovered memory is not JSON") from exc
    if canonical_bytes(document)!=source:raise ValueError("Recovered memory is not canonical")
    return document

def save(document,folder,width=64):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    png,legend=encode(document,width)
    (folder/"memory.png").write_bytes(png)
    (folder/"legend.json").write_bytes(canonical_bytes(legend))
    return {"png_bytes":len(png),"legend_bytes":len(canonical_bytes(legend)),"total_bytes":len(png)+len(canonical_bytes(legend)),
            "raw_rgb_bytes":legend["width"]*legend["height"]*3,"word_pixels":legend["token_count"],"vocabulary_tokens":len(legend["colors"]),
            "width":legend["width"],"height":legend["height"]}

def _bounded_read(path):
    with path.open("rb") as stream:
        size=os.fstat(stream.fileno()).st_size
        if size>MAX_FILE:raise ValueError("Memory file too large")
        data=stream.read(size+1)
        if len(data)!=size:raise ValueError("Memory file changed while reading")
        return data

def load(folder):
    folder=Path(folder)
    # These are the only memory files read by this decoder. No graph/label registry.
    png=_bounded_read(folder/"memory.png")
    raw=_bounded_read(folder/"legend.json")
    try:legend=json.loads(raw)
    except ValueError as exc:raise ValueError("Invalid legend JSON") from exc
    return decode(png,legend)
