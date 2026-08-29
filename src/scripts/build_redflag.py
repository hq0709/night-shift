"""RedFlag candidate extraction: filter AgentClinic + MedAgentsBench for can't-miss diagnoses.

Stage 1 (this script, free): string-match gold diagnoses against the can't-miss alias list.
Stage 2 (llm): propose the per-case sentinel feature + differential aliases.
Stage 3 (human): verify all candidates. Only verified cases enter RedFlag-N.
"""
from __future__ import annotations
import sys, json, pathlib, collections
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from taxonomy.redflag_list import RED_FLAGS, match  # noqa: E402
from envs import qa                                  # noqa: E402

out = []

# --- AgentClinic OSCE cases (interactive portion) -------------------------------
for fn, tag in [("agentclinic_medqa_extended.jsonl", "agentclinic_medqa"),
                ("agentclinic_nejm_extended.jsonl", "agentclinic_nejm")]:
    p = _ROOT / "external/AgentClinic" / fn
    if not p.exists():
        continue
    for i, line in enumerate(open(p)):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "OSCE_Examination" in d:
            o = d["OSCE_Examination"]
            gold = str(o.get("Correct_Diagnosis", ""))
            obj = str(o.get("Objective_for_Doctor", ""))
            hist = str(o.get("Patient_Actor", {}).get("History", ""))
        else:
            gold = str(d.get("answers", d.get("question", "")))
            obj = str(d.get("question", "")); hist = str(d.get("patient_info", ""))[:600]
        c = match(gold)
        if c:
            out.append(dict(uid=f"{tag}::{i}", mode="interactive", source=tag,
                            gold=gold, canonical=c,
                            sentinel_ref=RED_FLAGS[c]["sentinel"], context=(obj + " " + hist)[:600]))

# --- MedAgentsBench static items ------------------------------------------------
for it in qa.load_hard():
    if not it["clinical"]:
        continue
    gold_text = it["options"].get(it["answer_idx"], "") or it["answer_text"]
    c = match(gold_text)
    if c:
        out.append(dict(uid=it["uid"], mode="static", source=it["source"],
                        gold=gold_text, canonical=c,
                        sentinel_ref=RED_FLAGS[c]["sentinel"], context=it["question"][:600]))

p = _ROOT / "data/redflag_candidates.jsonl"
p.parent.mkdir(exist_ok=True)
with open(p, "w") as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"candidates: {len(out)}  -> {p}")
print("\nby mode:", dict(collections.Counter(r["mode"] for r in out)))
print("\nby canonical diagnosis:")
for k, v in collections.Counter(r["canonical"] for r in out).most_common():
    print(f"  {k:42s} {v}")
