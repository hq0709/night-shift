"""Dose-response for the central claim.

The manipulation in Section V-A is binary: the required item is present or absent. If elicited
confidence reports an information state, the relation should be graded -- removing *more* of what
the calculation requires should depress stated confidence further, monotonically, while leaving
discrimination flat at every dose. A monotone confidence response with a flat AUROC response is
much harder to explain by anything other than the account we propose; a two-point contrast is not.

Items are restricted to those carrying at least K removable required entities so that every dose
level is evaluated on the *same* items, making the comparison within-item.
"""
from __future__ import annotations
import sys, json, argparse, pathlib, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
_R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R))
sys.path.insert(0, str(_R / "experiments"))
from evidence_manip_calc import removable_entity          # noqa: E402
from envs import medcalc, schema as S                     # noqa: E402
from budgets import dial                                  # noqa: E402
from common import llm                                    # noqa: E402
import ast, re                                            # noqa: E402

PLACE = " [value not recorded]"


def removable_set(note, entities, k):
    """Up to k disjoint (entity, fragment) removals, found greedily on the shrinking note."""
    out, work = [], note
    for _ in range(k):
        sp = removable_entity(work, entities)
        if not sp:
            break
        ent, frag = sp
        if frag not in work:
            break
        out.append((ent, frag))
        work = work.replace(frag, PLACE, 1)
        entities = {a: b for a, b in entities.items() if a != ent}
    return out


def main(models, n, seeds, effort, workers, out, kmax, dry):
    items = medcalc.load(n, seed=0)
    prepared = []
    for it in items:
        try:
            ents = ast.literal_eval(str(it.get("entities") or "{}"))
        except Exception:
            ents = {}
        rs = removable_set(it["note"], ents, kmax)
        if len(rs) >= kmax:                      # same items at every dose
            prepared.append((it, rs))
    print(f"{len(prepared)}/{len(items)} items carry >= {kmax} removable required entities")
    jobs = [(m, k, s, it, rs) for m in models for k in range(kmax + 1)
            for s in range(seeds) for it, rs in prepared]
    print(f"-> {len(jobs)} episodes ({kmax+1} dose levels)")
    if dry:
        return

    def one(j):
        m, k, s, it, rs = j
        note = it["note"]
        for _, frag in rs[:k]:
            note = note.replace(frag, PLACE, 1)
        prompt = f"{note}\n\n{it['question']}\n\n" + medcalc.CONTRACT
        ep = dial.ask(m, prompt, axis="A_effort", setting=effort, seed=s,
                      system="You are an experienced attending physician.",
                      item_id=f"{it['uid']}|dose{k}|{s}", max_tokens=32000, structured="diagnosis")
        r = S.parse(ep.text, None)
        sc = medcalc.score(it, r.get("letter") or "")
        return dict(model=m, dose=k, seed=s, uid=it["uid"], calculator=it["calculator"],
                    removed=[e for e, _ in rs[:k]], correct=int(sc["correct"]),
                    confidence=r.get("confidence"), parsed=bool(r.get("parsed")),
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
    with open(_R / out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {out}")
    for k, v in errs.most_common(4):
        print("  ERR", k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-5.4-mini")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--out", default="results/evidence_dose.jsonl")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    main(a.models.split(","), a.n, a.seeds, a.effort, a.workers, a.out, a.kmax, a.dry)
