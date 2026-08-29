"""Ablations required to bring the manuscript to journal standard.

A1  differential cap K   -- does the elicited-differential ceiling drive the breadth results?
A2  elicitation position -- confidence produced before vs after the answer. Joint generation is the
                           deployment-realistic protocol but confounds elicitation with generation;
                           this separates them without leaving the single-call setting.
A3  scale ceiling        -- does a 1-10 integer scale compress the usable range? 1-5 vs 1-10 vs 1-100.
"""
from __future__ import annotations
import sys, json, copy, argparse, pathlib, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
_R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R))
from envs import qa, schema as S      # noqa: E402
from budgets import dial              # noqa: E402
from common import llm                # noqa: E402


def variant(kind, val):
    s = copy.deepcopy(S.EPISODE_SCHEMA)
    p = s["properties"]
    if kind == "K":
        p["differential"]["description"] = (
            f"Every diagnosis or option you actually considered, most likely first, at most {val}. "
            f"List only what you genuinely weighed.")
    elif kind == "pos":
        if val == "conf_first":
            s["required"] = ["confidence", "differential", "red_flag_considered",
                             "key_findings_used", "final_answer"]
            p["confidence"]["description"] = ("Before committing: your confidence that the answer "
                                              "you are about to give is correct, 1 to 10.")
    elif kind == "scale":
        hi = int(val)
        p["confidence"]["description"] = (
            f"Confidence that final_answer is correct, 1 (guess) to {hi} (certain).")
    return s


def run(kind, vals, model, n, ops, workers, out):
    items = qa.stratified_sample(qa.load_hard(), n, seed=0) + \
            qa.stratified_sample(qa.load_split(split="test"), n, seed=0)
    for it in items[:n]:
        it["split"] = "hard"
    for it in items[n:]:
        it["split"] = "std"
    jobs = [(v, o, it) for v in vals for o in ops for it in items]
    print(f"{kind}: {len(vals)} variants x {len(ops)} ops x {len(items)} items -> {len(jobs)} calls")

    def one(j):
        v, o, it = j
        orig = S.EPISODE_SCHEMA
        S.EPISODE_SCHEMA = variant(kind, v)
        try:
            ep = dial.ask(model, qa.prompt_structured(it), axis="A_effort", setting=o, seed=0,
                          system=qa.DOCTOR_SYSTEM, item_id=f"{it['uid']}|{kind}{v}",
                          max_tokens=64000, structured="letter")
        finally:
            S.EPISODE_SCHEMA = orig
        r = S.parse(ep.text, it["options"])
        c = r["confidence"]
        if kind == "scale" and c is not None:      # rescale onto 0-10 for comparability
            c = c * 10.0 / int(v)
        return dict(ablation=kind, variant=str(v), operating_point=o, uid=it["uid"],
                    split=it["split"], letter=r["letter"], gold=it["answer_idx"],
                    correct=(r["letter"] == it["answer_idx"]) if r["letter"] else False,
                    confidence=c, breadth=r["breadth"], parsed=r["parsed"],
                    reasoning_tokens=ep.reasoning_tokens)

    rows, errs = [], collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                errs[str(e)[:70]] += 1
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  ${llm.global_spend_usd():.2f}", flush=True)
    p = _R / out
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {p}")
    if errs:
        [print("  ERR", k, v) for k, v in errs.most_common(4)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["K", "pos", "scale"])
    ap.add_argument("--vals", required=True)
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--ops", default="high,none")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.kind, a.vals.split(","), a.model, a.n, a.ops.split(","), a.workers, a.out)
