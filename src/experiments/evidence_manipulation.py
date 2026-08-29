"""Causal test of the mechanism: does elicited confidence become informative *because* the
discriminating evidence is present?

Everything is held fixed except what the model has in hand:

  WITHHELD  the vignette with the key discriminating finding removed
  GIVEN     the same vignette with that finding restored
  ACQUIRED  the model must request it itself (interactive), and it is returned on request

WITHHELD vs GIVEN isolates evidence *availability*. GIVEN vs ACQUIRED isolates evidence *agency* --
whether the model obtained the finding through its own action. If confidence reports convergence,
availability should raise discrimination; if it additionally reports something about the model's own
epistemic process, agency should raise it further.
"""
from __future__ import annotations
import sys, json, argparse, pathlib, collections, re
from concurrent.futures import ThreadPoolExecutor, as_completed
_R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R))
from envs import clinic, schema as S        # noqa: E402
from budgets import dial                     # noqa: E402
from taxonomy.families import considered     # noqa: E402
from common import llm                       # noqa: E402

KT = {r["uid"]: r for r in map(json.loads, open(_R / "data/key_tests_medqa.jsonl"))}


def vignette(case, include_key: bool):
    """Full case as text, with the key discriminating finding present or removed."""
    key = (KT.get(case["uid"], {}).get("key_findings") or [None])[0]
    lines = []
    for label, val in clinic._flatten(case.get("tests") or {}) + clinic._flatten(case.get("exams") or {}):
        pretty = label.replace("_", " ")
        if (not include_key) and key and _same(pretty, key):
            continue                              # withhold exactly the key finding
        lines.append(f"- {pretty}: {val}")
    return (f"{case['objective']}\n\n"
            f"Patient: {json.dumps(case['patient'], ensure_ascii=False)[:2200]}\n\n"
            f"Findings:\n" + "\n".join(lines))


def _same(a, b):
    ta = {w for w in re.sub(r"[^a-z0-9 ]", " ", a.lower()).split() if len(w) > 3}
    tb = {w for w in re.sub(r"[^a-z0-9 ]", " ", b.lower()).split() if len(w) > 3}
    return bool(ta) and len(ta & tb) >= max(1, len(tb) // 2)


def main(models, n, seeds, effort, workers, out):
    cases = [c for c in clinic.load_cases("medqa")[:n] if KT.get(c["uid"], {}).get("key_findings")]
    jobs = [(m, cond, s, c) for m in models for cond in ("withheld", "given", "acquired")
            for s in range(seeds) for c in cases]
    print(f"{len(cases)} cases x 3 conditions x {seeds} seeds x {len(models)} models "
          f"-> {len(jobs)} episodes")

    def one(j):
        m, cond, s, c = j
        key = (KT[c["uid"]]["key_findings"] or [None])[0]
        if cond == "acquired":
            con = clinic.run_case(c, m, turn_cap=8, effort=effort, seed=s)
            r = con.readout or {}
            got = any(_same(a["request"] + " " + str(a["response"]), key or "")
                      for a in con.actions if a.get("response")
                      and "not available" not in str(a["response"]).lower())
            n_act = con.n_asked + con.n_tested
            rtok = con.reasoning_tokens
        else:
            prompt = (vignette(c, include_key=(cond == "given")) +
                      "\n\nGive your single most likely diagnosis. " + S.CONTRACT_NOTE)
            ep = dial.ask(m, prompt, axis="A_effort", setting=effort, seed=s,
                          system="You are an experienced attending physician.",
                          item_id=f"{c['uid']}|{cond}|{s}", max_tokens=32000,
                          structured="diagnosis")
            r = S.parse(ep.text, None)
            got = (cond == "given")
            n_act = 0
            rtok = ep.reasoning_tokens
        dx = r.get("letter") or ""
        return dict(model=m, condition=cond, seed=s, uid=c["uid"], gold=c["gold"],
                    dx=dx, correct=int(considered(c["gold"], [dx])),
                    confidence=r.get("confidence"), breadth=r.get("breadth"),
                    differential=r.get("differential", []),
                    key_finding=key, key_obtained=int(bool(got)),
                    n_actions=n_act, reasoning_tokens=rtok,
                    parsed=bool(r.get("parsed")))

    rows, errs = [], collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                errs[str(e)[:80]] += 1
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
    ap.add_argument("--models", default="gpt-5.4-mini")
    ap.add_argument("--n", type=int, default=214)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="results/evidence_manip.jsonl")
    a = ap.parse_args()
    main(a.models.split(","), a.n, a.seeds, a.effort, a.workers, a.out)
