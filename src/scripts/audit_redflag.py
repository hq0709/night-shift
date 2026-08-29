"""Mechanical consistency audit of the RedFlag-probe.

This is NOT hand verification and does not substitute for it. It only catches errors a machine can
prove: a sentinel quote the annotator invented rather than took from the stem, an alias list missing
its own condition, a "red flag" that merely restates the gold answer, and duplicates. Cases that
fail here should be dropped or re-examined first; cases that pass still require a human read.
"""
from __future__ import annotations
import sys, json, re, pathlib, collections
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from taxonomy.families import considered   # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower())


def shingles(s: str, k: int = 4) -> set[str]:
    w = norm(s).split()
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))} or {" ".join(w)}


rows = [json.loads(l) for l in open(_ROOT / "data/redflag_probe_canon.jsonl")]
flags = collections.Counter()
audited = []

seen = set()
for r in rows:
    issues = []
    stem, quote = r.get("stem", ""), r.get("sentinel_quote", "")

    # 1. the sentinel quote must actually come from the stem
    if not quote.strip():
        issues.append("no_sentinel_quote")
    else:
        sq, ss = shingles(quote), shingles(stem)
        if norm(quote) not in norm(stem) and not (sq & ss):
            issues.append("quote_not_in_stem")

    # 2. the alias list must contain the condition it claims to name
    if not considered(r["canonical"], r.get("aliases", [])) and \
       not considered(r["condition"], r.get("aliases", [])):
        issues.append("aliases_miss_condition")

    # 3. Does the red-flag condition simply restate the gold ANSWER?
    #
    # This is a stratification variable, not a defect. RF-MISS asks whether the can't-miss condition
    # appears in the stated DIFFERENTIAL, which is weaker than answering correctly -- a model can
    # list tamponade and still choose the wrong management step. But when the gold answer is
    # literally the diagnosis name, RF-MISS becomes strongly correlated with accuracy, so those
    # cases are marked and the headline RF-MISS rate is also reported on the complement.
    #
    # Token overlap alone is far too loose here: it fires on "CT pulmonary angiography" for
    # pulmonary embolism and "percutaneous coronary intervention" for ACS, which are exactly the
    # IDEAL cases (gold = the test or treatment FOR the red flag, not the red flag itself).
    gold = str(r.get("gold", ""))
    gold_head = re.split(r"[;,]", gold)[0].strip()
    diagnosis_like = considered(r["canonical"], [gold_head]) and not re.search(
        r"\b(ct|mri|x[- ]?ray|ultrasound|angiograph|echocardiograph|biopsy|culture|obtain|"
        r"administer|initiate|perform|intervention|therapy|treatment|surgery|drainage|"
        r"decompression|thoracostomy|pericardiocentesis|dose|mg\b|intravenous|oral)\b",
        gold_head, re.I)
    if diagnosis_like:
        issues.append("gold_is_the_diagnosis")

    # 4. duplicates
    if r["uid"] in seen:
        issues.append("duplicate_uid")
    seen.add(r["uid"])

    if r.get("weak_sentinel"):
        issues.append("weak_sentinel")
    r = {**r, "gold_is_the_diagnosis": "gold_is_the_diagnosis" in issues}

    for i in issues:
        flags[i] += 1
    audited.append({**r, "audit_issues": issues})

# gold_is_the_diagnosis is a stratum label, not a failure: only the other issues block a case.
BLOCKING = {"no_sentinel_quote", "quote_not_in_stem", "aliases_miss_condition",
            "duplicate_uid", "weak_sentinel"}
clean = [r for r in audited if not (set(r["audit_issues"]) & BLOCKING)]
with open(_ROOT / "data/redflag_probe_audited.jsonl", "w") as fh:
    for r in audited:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"audited {len(rows)} cases")
strat = sum(r.get("gold_is_the_diagnosis") for r in audited)
print(f"  pass blocking checks        : {len(clean)} ({len(clean)/len(rows):.0%})")
print(f"  blocked, need human review  : {len(rows)-len(clean)}")
print(f"  stratum: gold IS the dx     : {strat}  (RF-MISS also reported on the {len(rows)-strat} complement)")
print("\nissue counts:")
for k, v in flags.most_common():
    print(f"    {k:26s} {v}")
print("\nblocked cases (human review required):")
for r in audited:
    if set(r["audit_issues"]) & BLOCKING:
        print(f"  {r['uid']:26s} {r['canonical'][:34]:36s} {','.join(r['audit_issues'])}")
        print(f"      quote: \"{r.get('sentinel_quote','')[:80]}\"")
        print(f"      gold : {str(r.get('gold',''))[:70]}")
