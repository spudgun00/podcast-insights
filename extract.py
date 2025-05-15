#!/usr/bin/env python3
"""
Usage: python extract.py path/to/transcript.json
Outputs YAML to data/insights/<episode>.yaml
"""

import sys, os, json, re, yaml, textwrap, tiktoken
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)                               # loads .env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY missing in .env"); sys.exit(1)

client = OpenAI(api_key=api_key)
MODEL = "gpt-4o-mini"                      # or gpt-3.5-turbo-1106

PROMPT = """\
You are a startup analyst. From the transcript below,
write two bullet lists:

IDEAS:
- <startup idea>  (max 20 words)

PAINS:
- <founder pain point>  (max 20 words)

Max 3 bullets each. If none, write "None".

TRANSCRIPT:
```text
{chunk}
```"""

enc = tiktoken.encoding_for_model("gpt-4o-mini")
def tk(txt): return len(enc.encode(txt))
def chunks(txt, lim=3000):
    buf, count = [], 0
    for w in txt.split():
        t = tk(w + " ")
        if count + t > lim and buf:
            yield " ".join(buf); buf, count = [], 0
        buf.append(w); count += t
    if buf: yield " ".join(buf)

def parse(raw):
    ideam = re.search(r"IDEAS:\s*(.*?)\nPAINS:", raw, re.S|re.I)
    painm = re.search(r"PAINS:\s*(.*)", raw, re.S|re.I)
    ideas = [re.sub(r"^- ", "", l).strip() for l in (ideam.group(1).splitlines() if ideam else []) if l.strip()]
    pains = [re.sub(r"^- ", "", l).strip() for l in (painm.group(1).splitlines() if painm else []) if l.strip()]
    return ideas, pains

def main(jpath):
    with open(jpath) as f:
        text = json.load(f)["text"]

    ideas, pains = set(), set()
    for chunk in chunks(text):
        msg = PROMPT.format(chunk=chunk[:15000])
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":msg}],
            temperature=0.2,
        )
        i, p = parse(resp.choices[0].message.content)
        ideas.update(i); pains.update(p)
        if ideas or pains: break         # first chunk good enough

    out = {"ideas": sorted(ideas), "pains": sorted(pains)}
    out_path = Path("data/insights") / (Path(jpath).stem + ".yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f: yaml.safe_dump(out, f)

    print("\n=== IDEAS ===")
    print(*(["None"] if not ideas else ["• "+x for x in ideas]), sep="\n")
    print("\n=== PAINS ===")
    print(*(["None"] if not pains else ["• "+x for x in pains]), sep="\n")
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract.py transcript.json"); sys.exit(1)
    main(sys.argv[1])

