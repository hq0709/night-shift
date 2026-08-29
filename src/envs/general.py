"""Independent (non-MedAgentsBench) datasets, for the cross-dataset generalisation test.

All prior evidence came from one benchmark family, so "it also holds on the non-clinical subsets"
was never a real generalisation claim — those subsets ship inside MedAgentsBench. These three are
separate corpora chosen to span a wide accuracy range, which is what the paper's scope claim
(confidence discriminates when the model is competent, and collapses when it is not) requires.
"""
from __future__ import annotations
import re, random, functools


@functools.lru_cache(maxsize=8)
def _ds(name, cfg=None, split="test"):
    from datasets import load_dataset
    return load_dataset(*(x for x in (name, cfg) if x), split=split)


def load_gsm8k(n=300, seed=0):
    d = _ds("openai/gsm8k", "main", "test")
    idx = list(range(len(d))); random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        r = d[i]
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(dict(uid=f"gsm8k::{i}", source="GSM8K", clinical=False, free_form=True,
                        question=r["question"], options={}, answer_idx=gold, answer_text=gold))
    return out


def load_arc(n=300, seed=0):
    d = _ds("allenai/ai2_arc", "ARC-Challenge", "test")
    idx = list(range(len(d))); random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        r = d[i]
        labs = r["choices"]["label"]; txts = r["choices"]["text"]
        # some items use numeric labels; normalise to letters so one schema serves every dataset
        letters = [chr(ord("A") + j) for j in range(len(labs))]
        opts = dict(zip(letters, txts))
        key = r["answerKey"]
        gold = letters[labs.index(key)] if key in labs else None
        if gold is None:
            continue
        out.append(dict(uid=f"arc::{i}", source="ARC-Challenge", clinical=False, free_form=False,
                        question=r["question"], options=opts, answer_idx=gold,
                        answer_text=opts[gold]))
    return out


def load_mmlu_pro(n=300, seed=0):
    d = _ds("TIGER-Lab/MMLU-Pro", None, "test")
    idx = list(range(len(d))); random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        r = d[i]
        opts = {chr(ord("A") + j): t for j, t in enumerate(r["options"])}
        gold = r["answer"]
        if gold not in opts:
            continue
        out.append(dict(uid=f"mmlupro::{i}", source="MMLU-Pro-std", clinical=False,
                        free_form=False, question=r["question"], options=opts,
                        answer_idx=gold, answer_text=opts[gold]))
    return out


LOADERS = {"gsm8k": load_gsm8k, "arc": load_arc, "mmlupro": load_mmlu_pro}

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def score_free(gold: str, answer: str) -> bool:
    """Numeric tasks compare the last number. A gold label containing letters is a free-text
    diagnosis (AgentClinic), scored by the same alias-aware consideration match used elsewhere."""
    if gold and any(ch.isalpha() for ch in str(gold)):
        from taxonomy.families import considered
        return considered(str(gold), [str(answer or "")])
    if answer is None:
        return False
    m = _NUM.findall(str(answer).replace(",", ""))
    if not m:
        return False
    try:
        return abs(float(m[-1]) - float(gold)) < 1e-6
    except ValueError:
        return False


def load_redflag(n=200, seed=0):
    """RedFlag probe: items whose stem carries a can't-miss (time-critical) red flag.

    Built tonight (taxonomy/sentinels.py -> LLM confirmation -> canonicalisation -> mechanical
    audit). These come from the ORDINARY-difficulty `test` split, so they are difficulty-matched to
    the ordinary-item control by construction -- which is what makes the high-stakes contrast
    interpretable rather than a difficulty effect in disguise.
    """
    import json, pathlib
    _R = pathlib.Path(__file__).resolve().parents[1]
    probe = [json.loads(l) for l in open(_R / "data/redflag_probe_canon.jsonl")]
    keep = {r["uid"]: r for r in probe if r["mode"] == "static"}
    from envs import qa
    out = []
    for it in qa.load_split(split="test"):
        if it["uid"] in keep:
            r = keep[it["uid"]]
            out.append({**it, "source": "RedFlag", "clinical": True, "free_form": False,
                        "redflag": r["canonical"], "rf_aliases": r.get("aliases", []),
                        "gold_is_dx": bool(r.get("gold_is_the_diagnosis"))})
    return out[:n]


LOADERS["redflag"] = load_redflag


def load_agentclinic(n=250, seed=0):
    """AgentClinic OSCE vignettes as a second, independent CLINICAL source.

    Different from MedAgentsBench on three axes that matter for a journal reviewer: cases are
    OSCE-style vignettes rather than exam items, the answer is a FREE-TEXT diagnosis rather than a
    letter, and the gold label is the diagnosis itself. The full case (history, exam, tests) is
    presented at once — the interactive protocol is a separate question and is not what the
    confidence endpoint needs.
    """
    import json
    from envs import clinic
    from taxonomy.redflag_list import match as rf_match
    out = []
    for c in clinic.load_cases("medqa")[:n]:
        findings = clinic.available_tests(c)
        body = (f"{c['objective']}\n\n"
                f"Patient: {json.dumps(c['patient'], ensure_ascii=False)[:2500]}\n\n"
                f"Available findings: {', '.join(findings[:25])}")
        out.append(dict(uid=f"ac::{c['uid']}", source="AgentClinic-OSCE", clinical=True,
                        free_form=True, question=body, options={},
                        answer_idx=c["gold"], answer_text=c["gold"],
                        redflag=rf_match(c["gold"])))
    return out


LOADERS["agentclinic"] = load_agentclinic
