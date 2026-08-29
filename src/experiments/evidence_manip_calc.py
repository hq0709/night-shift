"""Second causal test of the same mechanism, on a task where the missing item is a *computational
input* rather than a diagnostic clue.

MedCalc-Bench annotates the entities each calculation requires. Removing one from the note produces
a case that is genuinely unanswerable rather than merely harder, which is the sharpest possible form
of "insufficient information". If elicited confidence reports an information state, its
discrimination should be unchanged by the removal (as in the diagnostic case), while its ability to
flag the missing input should be present.
"""
from __future__ import annotations
import sys, json, re, argparse, pathlib, collections, random
from concurrent.futures import ThreadPoolExecutor, as_completed
_R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R))
from envs import medcalc, schema as S    # noqa: E402
from budgets import dial                  # noqa: E402
from common import llm                    # noqa: E402


def removable_entity(note, entities):
    """Find one numerically-stated required entity and the exact substring carrying its value.

    Removing the whole sentence would also remove co-reported laboratory values, confounding the
    manipulation. We therefore excise only the value and its immediate label, leaving the rest of
    the note intact, so the case differs in exactly one required input.
    """
    for k, v in (entities or {}).items():
        if k in ("sex", "age") or not isinstance(v, list) or not v:
            continue
        val = v[0]
        if not isinstance(val, (int, float)):
            continue
        cands = {str(val), f"{val:.1f}", f"{val:.2f}"}
        if float(val) == int(val):
            cands.add(str(int(val)))
        for c in filter(None, cands):
            m = re.search(r"(?<![\d.])" + re.escape(c) + r"(?![\d])", note)
            if not m:
                continue
            # widen to the surrounding "label value unit" fragment between commas/semicolons
            st = max(note.rfind(",", 0, m.start()), note.rfind(";", 0, m.start()),
                     note.rfind(".", 0, m.start())) + 1
            en = min([x for x in (note.find(",", m.end()), note.find(";", m.end()),
                                  note.find(".", m.end())) if x != -1] or [len(note)])
            frag = note[st:en]
            if 3 < len(frag) < 90:
                return k, frag
    return None


def main(models, n, seeds, effort, workers, out):
    import ast
    items = medcalc.load(n, seed=0)
    prepared = []
    for it in items:
        try:
            ents = ast.literal_eval(str(it.get("entities") or "{}"))
        except Exception:
            ents = {}
        sp = removable_entity(it["note"], ents)
        if sp:
            prepared.append((it, sp))
    print(f"{len(prepared)}/{len(items)} items have a removable required entity")
    jobs = [(m, cond, s, it, sp) for m in models for cond in ("withheld", "given")
            for s in range(seeds) for it, sp in prepared]
    print(f"-> {len(jobs)} episodes")

    def one(j):
        m, cond, s, it, (ent, sent) = j
        note = it["note"] if cond == "given" else it["note"].replace(sent, " [value not recorded]")
        prompt = (f"{note}\n\n{it['question']}\n\n" + medcalc.CONTRACT)
        ep = dial.ask(m, prompt, axis="A_effort", setting=effort, seed=s,
                      system="You are an experienced attending physician.",
                      item_id=f"{it['uid']}|{cond}|{s}", max_tokens=32000, structured="diagnosis")
        r = S.parse(ep.text, None)
        sc = medcalc.score(it, r.get("letter") or "")
        return dict(model=m, condition=cond, seed=s, uid=it["uid"], calculator=it["calculator"],
                    removed_entity=ent, correct=int(sc["correct"]), value=sc["value"],
                    confidence=r.get("confidence"), breadth=r.get("breadth"),
                    parsed=bool(r.get("parsed")), reasoning_tokens=ep.reasoning_tokens)

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
    ap.add_argument("--models", default="gpt-5.4-mini")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="results/evmanip_calc.jsonl")
    a = ap.parse_args()
    main(a.models.split(","), a.n, a.seeds, a.effort, a.workers, a.out)
