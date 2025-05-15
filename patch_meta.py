#!/usr/bin/env python3
"""
Patch existing transcript JSON files with basic metadata
extracted from filenames
"""
import json, glob, os

for jf in glob.glob("data/transcripts/*.json"):
    data = json.load(open(jf))
    if "meta" in data:
        continue
    fname = os.path.basename(jf).replace(".json","")
    pieces = fname.split("_")
    meta = {
        "podcast": pieces[0].replace("-", " "),
        "episode": " ".join(pieces[1:])[:120],
        "date": "",
        "author": ""
    }
    data["meta"] = meta
    json.dump(data, open(jf,"w"))
    print("patched", jf)
