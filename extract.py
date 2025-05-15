#!/usr/bin/env python3
"""
Enhanced extractor:
- infers ideas from pain points
- processes up to 3 transcript chunks
- stores meta + first-chunk timestamp
Usage: python extract.py path/to/transcript.json
"""

import sys, os, json, re, yaml, tiktoken
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from openai import OpenAI

# ── config ──────────────────────────────────────────────────────────────
MODEL  = "gpt-4o-mini"           # swap to "gpt-4o" or gpt-3.5 as you like
MAX_CHUNKS = 3                   # how many 3k-token slices to scan
TEMP   = 0.2

load_dotenv(); api = os.getenv("OPENAI_API_KEY")
if not api: sys.exit("❌  OPENAI_API_KEY missing in .env")
client = OpenAI(api_key=api)
enc = tiktoken.encoding_for_model("gpt-4o-mini")
tk   = lambda s: len(enc.encode(s))

PROMPT = """\
You are a startup analyst. From the transcript slice below extract:

1. **STARTUP IDEAS** – business opportunities mentioned *explicitly* **OR** implied by the problems discussed.
2. **FOUNDER PAIN POINTS** – specific challenges entrepreneurs or businesses face.

Rules:
- Max **3** bullets per list.  
- 1 bullet ≤ 20 words.  
- If none, write "None".

Return exactly:

IDEAS:
- idea 1
- idea 2

PAINS:
- pain 1
- pain 2

TRANSCRIPT ({} – {}):
```text
{}
```"""

# ── helpers ─────────────────────────────────────────────────────────────
def chunks(segments, tok_cap=3000):
    buf, buf_tok, start_t = [], 0, segments[0]["start"]
    for seg in segments:
        t = tk(seg["text"] + " ")
        if buf_tok + t > tok_cap and buf:
            yield start_t, segments[seg_i]["end"], " ".join(buf)
            buf, buf_tok, start_t = [], 0, seg["start"]
        buf.append(seg["text"]); buf_tok += t; seg_i = segments.index(seg)
    yield start_t, segments[-1]["end"], " ".join(buf)

def parse_lists(raw):
    sec = lambda tag: re.search(fr"{tag}:\s*(.+?)(?:\n[A-Z]|$)", raw, re.S|re.I)
    clean = lambda l: re.sub(r"^[\-\*\•]\s*", "", l).strip()
    ideas = [clean(x) for x in (sec("IDEAS").group(1).splitlines() if sec("IDEAS") else []) if clean(x) and clean(x)!="**"]
    pains = [clean(x) for x in (sec("PAINS").group(1).splitlines() if sec("PAINS") else []) if clean(x) and clean(x)!="**"]
    return ideas, pains

def mins(sec): return str(timedelta(seconds=int(sec)))[:-3]

# ── main ────────────────────────────────────────────────────────────────
def main(path):
    data = json.load(open(path))
    segs = data["segments"]
    meta = data.get("meta", {})    # may be empty on first runs
    
    # Add a nice header for the user (moved inside the function where meta is defined)
    print("="*71)
    print(f"ANALYZING: {meta.get('podcast', 'Unknown Podcast')} – {meta.get('episode', 'Unknown Episode')}")
    if meta.get("author"):
        print(f"SPEAKERS: {meta.get('author')}")
    print("="*71)

    ideas, pains, used_time = set(), set(), None

    for i, (start, end, txt) in enumerate(chunks(segs)):
        if i >= MAX_CHUNKS: break
        prompt = PROMPT.format(mins(start), mins(end), txt[:14000])
        raw = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=TEMP,
        ).choices[0].message.content
        I, P = parse_lists(raw)
        if I: ideas.update(I)
        if P: pains.update(P)
        if (ideas or pains) and used_time is None:
            used_time = f"{mins(start)}–{mins(end)}"
        if ideas and pains: break

    # second pass: transform pains → ideas if ideas still empty
    if not ideas and pains:
        trans_prompt = "Transform these pain points into startup ideas:\n" + \
                       "\n".join(f"- {p}" for p in pains)
        add_raw = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":trans_prompt}],
            temperature=0.3
        ).choices[0].message.content
        more = [re.sub(r"^[\-\*\•]\s*", "", l).strip()
                for l in add_raw.splitlines() if l.strip().startswith(("-", "•"))]
        ideas.update(more[:3])

    out = {
        "podcast" : meta.get("podcast"),
        "episode" : meta.get("episode"),
        "date"    : meta.get("pub_date"),
        "slice"   : used_time,
        "ideas"   : sorted(ideas) if ideas else ["None"],
        "pains"   : sorted(pains) if pains else ["None"],
    }

    dst = Path("data/insights"); dst.mkdir(parents=True, exist_ok=True)
    out_file = dst / (Path(path).stem + ".yaml")
    yaml.safe_dump(out, open(out_file,"w"))

    print("\n=== IDEAS ===")
    for i in out["ideas"]: print("•", i)
    print("\n=== PAINS ===")
    for p in out["pains"]: print("•", p)
    print(f"\nSaved → {out_file}")

if __name__ == "__main__":
    if len(sys.argv)!=2: sys.exit("Usage: python extract.py transcript.json")
    main(sys.argv[1])