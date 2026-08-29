"""Sensitivity analysis on the free-text scorer.

The alias-aware match of (4) scores a free-text diagnosis by token overlap with the reference. That
rule is strict: it marks "Pancoast tumor" wrong against "apical lung tumor", and "Jersey finger"
wrong against "rupture of the flexor digitorum profundus tendon". Both are the same diagnosis.

The strictness is applied identically in every condition, so it cannot bias a between-condition
contrast, but it does understate absolute accuracy and could in principle move AUROC, since the
mis-scored answers tend to be confident ones. We therefore re-adjudicate every episode the string
rule marked wrong, using a judge model that is not among the evaluated models, and report every
principal quantity under both scorers.
"""
from __future__ import annotations
import sys, json, argparse, pathlib, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
_R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R))
from common import llm                                     # noqa: E402

JUDGE = "gpt-4o-mini"
SYS = ("You are a board-certified physician adjudicating whether two diagnosis strings name the same "
       "clinical entity. Answer EQUIVALENT if a clinician would accept the candidate as the same "
       "diagnosis as the reference (including eponyms, renamed classifications, and standard "
       "synonyms). Answer DIFFERENT if it names a different entity, a different level of the "
       "diagnostic hierarchy that changes management, or only a related finding. Reply with one word.")


def judge(gold, dx):
    r = llm.chat(JUDGE, [{"role": "user", "content":
                          f"Reference diagnosis: {gold}\nCandidate diagnosis: {dx}\n\nEQUIVALENT or DIFFERENT?"}],
                 system=SYS, max_tokens=4, temperature=0.0, seed=0)
    return int("EQUIV" in (r.get("text") or "").upper())


def main(workers, out):
    pairs = sorted({(d["gold"].strip(), d["dx"].strip())
                    for f in _R.glob("results/*.jsonl")
                    for d in map(json.loads, open(f))
                    if d.get("correct") == 0 and d.get("dx") and d.get("gold")})
    print(f"{len(pairs)} unique (reference, candidate) pairs to adjudicate")
    verdict, errs = {}, collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(judge, g, d): (g, d) for g, d in pairs}
        for i, f in enumerate(as_completed(futs), 1):
            g, d = futs[f]
            try:
                verdict[f"{g}||{d}"] = f.result()
            except Exception as e:
                errs[str(e)[:60]] += 1
            if i % 100 == 0:
                print(f"  {i}/{len(pairs)}  ${llm.global_spend_usd():.2f}", flush=True)
    json.dump(verdict, open(_R / out, "w"), indent=0)
    n_eq = sum(verdict.values())
    print(f"wrote {len(verdict)} -> {out};  {n_eq} ({n_eq/max(1,len(verdict)):.1%}) judged equivalent")
    for k, v in errs.most_common(3):
        print("  ERR", k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/synonym_verdicts.json")
    a = ap.parse_args()
    main(a.workers, a.out)
