"""Confidence in the actual deployment scenario: the model gathers its own information.

Every confidence result so far comes from a fully-specified vignette. In deployment a clinical agent
decides what to ask and what to order, then commits. If confidence is a readout of answer
convergence rather than of correctness, an agent that also controls its own evidence has one more
way to be confidently wrong -- it can converge on a diagnosis it never tested.
"""
from __future__ import annotations
import sys, json, argparse, pathlib, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from envs import clinic                    # noqa: E402
from taxonomy.families import considered   # noqa: E402
from taxonomy.redflag_list import match as rf_match, RED_FLAGS   # noqa: E402
from common import llm                     # noqa: E402


def main(models, n, turn_cap, effort, seeds, workers, out):
    cases = clinic.load_cases("medqa")[:n]
    jobs = [(m, c, s) for m in models for s in range(seeds) for c in cases]
    print(f"{len(cases)} cases x {len(models)} models x {seeds} seeds -> {len(jobs)} consultations")

    def one(j):
        m, c, s = j
        con = clinic.run_case(c, m, turn_cap=turn_cap, effort=effort, seed=s)
        r = con.readout or {}
        dx = r.get("letter") or ""
        correct = considered(c["gold"], [dx])
        rf = rf_match(c["gold"])
        rf_hit = None
        if rf:
            names = [rf] + list(RED_FLAGS[rf]["aliases"])
            cand = list(r.get("differential") or []) + [r.get("red_flag_considered", ""), dx]
            rf_hit = int(any(considered(nm, cand) for nm in names if nm))
        return dict(model=m, uid=c["uid"], seed=s, gold=c["gold"], dx=dx, correct=int(correct),
                    confidence=r.get("confidence"), breadth=r.get("breadth"),
                    n_actions=con.n_asked + con.n_tested, committed_early=con.committed_early,
                    redflag=rf, rf_hit=rf_hit, parsed=bool(r.get("parsed")),
                    patient_separation=con.patient_separation,
                    reasoning_tokens=con.reasoning_tokens)

    rows, errs = [], collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                errs[str(e)[:80]] += 1
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}  ${llm.global_spend_usd():.2f}", flush=True)
    p = pathlib.Path(out)
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} -> {p}")
    if errs:
        [print("  ERR", k, v) for k, v in errs.most_common(5)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-5.4-mini")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--turn_cap", type=int, default=8)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/interactive_conf.jsonl")
    a = ap.parse_args()
    main(a.models.split(","), a.n, a.turn_cap, a.effort, a.seeds, a.workers, a.out)
