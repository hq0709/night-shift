"""RedFlag-probe stage 3: canonicalize condition labels and emit the hand-verification worksheet.

The annotator returns free-text condition names at inconsistent granularity ("Aortic dissection" vs
"Acute aortic dissection"; "Acute coronary syndrome (STEMI/myocardial infarction)" vs "... (ST-
elevation myocardial infarction)"), which inflated 105 cases to 79 distinct labels. Scoring needs a
closed vocabulary, so each label is mapped back onto the can't-miss list; unmapped labels are kept
verbatim and surfaced for manual assignment. Also flags cases whose sentinel quote is too weak to
carry a safety claim, which is what hand verification should reject.
"""
from __future__ import annotations
import sys, json, pathlib, collections, re
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from taxonomy.redflag_list import RED_FLAGS, match   # noqa: E402

rows = [json.loads(l) for l in open(_ROOT / "data/redflag_probe.jsonl")]

# A sentinel quote must be a specific clinical feature, not a vague complaint.
WEAK = re.compile(r"^(difficulty|trouble|problem|issue|pain|discomfort|feeling|not feeling|unwell|"
                  r"symptoms?|complaint)s?\b.{0,25}$", re.I)

out, unmapped = [], collections.Counter()
for r in rows:
    canon = match(r["condition"]) or match(" ".join(r.get("aliases", [])))
    if canon is None:
        unmapped[r["condition"]] += 1
    weak = bool(WEAK.match(r.get("sentinel_quote", "").strip())) or \
        len(r.get("sentinel_quote", "").split()) < 3
    out.append({**r, "canonical": canon or r["condition"],
                "mapped": canon is not None, "weak_sentinel": weak,
                "verified": None})            # filled by hand

with open(_ROOT / "data/redflag_probe_canon.jsonl", "w") as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

mapped = sum(r["mapped"] for r in out)
weak = sum(r["weak_sentinel"] for r in out)
print(f"RedFlag-probe: {len(out)} cases")
print(f"  mapped to the can't-miss vocabulary : {mapped} ({mapped/len(out):.0%})")
print(f"  distinct canonical conditions       : {len(set(r['canonical'] for r in out))} "
      f"(was {len(set(r['condition'] for r in out))} free-text)")
print(f"  flagged: weak sentinel quote        : {weak}")
print(f"  source concentration                : {dict(collections.Counter(r['source'] for r in out).most_common(4))}")
if unmapped:
    print(f"\nunmapped labels needing manual assignment ({len(unmapped)}):")
    for k, v in unmapped.most_common(12):
        print(f"    {k[:60]:62s} {v}")

# Human verification worksheet: one line per case, decision column blank.
ws = _ROOT / "data/redflag_verification_sheet.tsv"
with open(ws, "w") as fh:
    fh.write("uid\tcanonical\tmapped\tweak\tsentinel_quote\tgold\tdecision(keep/drop)\tnote\n")
    for r in sorted(out, key=lambda x: (not x["weak_sentinel"], x["canonical"])):
        fh.write(f"{r['uid']}\t{r['canonical']}\t{int(r['mapped'])}\t{int(r['weak_sentinel'])}\t"
                 f"{r['sentinel_quote'][:110]}\t{r['gold'][:60]}\t\t\n")
print(f"\nverification worksheet -> {ws}")
print("  weak-sentinel cases are sorted to the top for review first.")
