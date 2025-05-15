from pathlib import Path
import yaml, jinja2

INSIGHTS = list(Path("data/insights").glob("*.yaml"))

items = []
for p in INSIGHTS:
    data = yaml.safe_load(open(p))
    for idea in data["ideas"]:
        if idea == "None": continue
        items.append({
            "type": "idea",
            "text": idea,
            "source": f"{data['podcast']} – {data['episode']} ({data['slice']})"
        })
    for pain in data["pains"]:
        if pain == "None": continue
        items.append({
            "type": "pain",
            "text": pain,
            "source": f"{data['podcast']} – {data['episode']} ({data['slice']})"
        })

# simple de-dupe by text
seen = set(); unique=[]
for i in items:
    if i["text"] in seen: continue
    seen.add(i["text"]); unique.append(i)

env = jinja2.Environment()
tpl = env.from_string("""
# StartupAudio.ai — Sample Digest

{% for i in unique if i.type == 'idea' %}
💡 **{{ i.text }}**  
 ▸ {{ i.source }}
{% endfor %}

---

{% for i in unique if i.type == 'pain' %}
⚠️ {{ i.text }}  
 ▸ {{ i.source }}
{% endfor %}
""")

print(tpl.render(unique=unique))
