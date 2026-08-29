"""Claim 2 confirmatory: the ordinal operating-point ladder on static task families.

Within-item randomised assignment of operating point, structured measurement contract at every
level, family classification with no LLM in the loop. Analysis (ITT with item fixed effects) is
done downstream from the jsonl this writes.

  python3 experiments/run_axisA.py --task mab --n 300 --models gpt-5.4-mini --seeds 3
"""
from __future__ import annotations
import sys, json, random, argparse, pathlib, collections
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from envs import qa, schema, medcalc, general   # noqa: E402
from budgets import dial         # noqa: E402
from taxonomy import families    # noqa: E402
from common import llm           # noqa: E402

# Ladders are model-family specific and are pruned by the separation gate before this runs.
LADDER = {"openai_reasoning": ["xhigh", "high", "medium", "low", "none"],
          "anthropic_c5": ["max", "xhigh", "high", "medium", "low", "off"],
          "anthropic_legacy": ["8192", "4096", "2048", "1024", "0"],
          "openai_chat": ["1024", "512", "256", "128", "64", "0"]}

# Generous ceilings: a ceiling near the top-of-ladder reasoning draw censors the answer and records
# a NON-RESPONSE caused by HIGH effort -- an artifact that points opposite to the hypothesis.
MAX_TOKENS = 64000


def surviving_ladder(model: str) -> list[str]:
    gate = _ROOT / "results/separation_gate.json"
    if gate.exists():
        d = json.loads(gate.read_text())
        if model in d and d[model].get("points"):
            return d[model]["points"]
    return LADDER[dial.family(model)]


def main(task, n, models, seeds, workers, out_path):
    if task == "mab":
        items = qa.stratified_sample(qa.load_hard(), n, seed=0)
    elif task in general.LOADERS:
        items = general.LOADERS[task](n, seed=0)
    elif task == "mab_easy":
        # ordinary-difficulty split — the difficulty control
        items = qa.stratified_sample(qa.load_split(split="test"), n, seed=0)
    elif task == "medcalc":
        items = medcalc.load(n, seed=0)
    else:
        raise SystemExit(f"unknown task {task}")

    jobs = []
    for m in models:
        ladder = surviving_ladder(m)
        print(f"{m}: operating points = {ladder}")
        for s in range(seeds):
            for it in items:
                # Within-item randomisation of assignment order; every item sees every point.
                order = list(ladder)
                random.Random(hash((it["uid"], m, s)) & 0xffffffff).shuffle(order)
                for pos, e in enumerate(order):
                    jobs.append((m, e, s, pos, it))
    print(f"{len(items)} items x {len(models)} models x {seeds} seeds -> {len(jobs)} calls")

    def one(j):
        m, e, s, pos, it = j
        is_calc = task == "medcalc"
        free = bool(it.get("free_form"))
        if is_calc:
            prompt = medcalc.prompt(it)
        elif free:
            # free-form numeric (GSM8K): same fixed contract, `differential` = approaches weighed
            prompt = (f"{it['question']}\n\nGive the final numeric answer only in "
                      f"`final_answer`.\n\n{schema.CONTRACT_NOTE}")
        else:
            prompt = qa.prompt_structured(it)
        ep = dial.ask(m, prompt, axis="A_effort", setting=e, seed=s,
                      system=qa.DOCTOR_SYSTEM, item_id=it["uid"], max_tokens=MAX_TOKENS,
                      structured="diagnosis" if (is_calc or free) else "letter")
        r = schema.parse(ep.text, None if (is_calc or free) else it["options"])
        extra = {}
        if is_calc:
            # The procedural family scores against MedCalc's own acceptance band, and splits
            # commission into slip (right formula, wrong arithmetic) vs knowledge (wrong formula).
            sc = medcalc.score(it, r["letter"] or "")
            correct = sc["correct"]
            gold_text = it["calculator"]
            split = medcalc.slip_vs_knowledge(it, r["letter"] or "",
                                              r.get("red_flag_considered", ""), r["differential"])
            extra = dict(calculator=it["calculator"], category=it["category"],
                         value=sc["value"], rel_err=sc["rel_err"], proc_split=split,
                         SLIP=(split == "slip"), KNOW=(split == "knowledge"))
            fam = families.classify(correct=correct, no_answer=r["no_answer"], gold=gold_text,
                                    differential=r["differential"],
                                    red_flag_considered=r.get("red_flag_considered", ""),
                                    confidence=r["confidence"], interactive=False)
        elif free:
            correct = general.score_free(it["answer_idx"], r["letter"])
            gold_text = it["answer_text"]
            fam = families.classify(correct=correct, no_answer=r["no_answer"], gold=gold_text,
                                    differential=r["differential"],
                                    red_flag_considered=r.get("red_flag_considered", ""),
                                    confidence=r["confidence"], interactive=False)
        else:
            correct = (r["letter"] == it["answer_idx"]) if r["letter"] else False
            gold_text = it["options"].get(it["answer_idx"], "") or it["answer_text"]
            fam = families.classify(correct=correct, no_answer=r["no_answer"], gold=gold_text,
                                    differential=r["differential"],
                                    red_flag_considered=r.get("red_flag_considered", ""),
                                    confidence=r["confidence"], interactive=False)
        return dict(model=m, operating_point=e, seed=s, assign_pos=pos, uid=it["uid"],
                    source=it["source"], clinical=it["clinical"],
                    gold=it.get("answer_idx", it.get("gold", "")),
                    letter=r["letter"], correct=correct, parsed=r["parsed"], **extra,
                    breadth=r["breadth"], confidence=r["confidence"],
                    n_findings=r.get("n_findings", 0),
                    red_flag_considered=r.get("red_flag_considered", ""),
                    reasoning_tokens=ep.reasoning_tokens, reasoning_exact=ep.reasoning_exact,
                    output_tokens=ep.output_tokens, input_tokens=ep.input_tokens,
                    answer_chars=len(ep.text), truncated=ep.truncated, finish=ep.finish,
                    **fam)

    rows, errs = [], collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as ex:
                errs[f"{futs[f][0]}|{futs[f][1]}|{str(ex)[:70]}"] += 1
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  ${llm.LEDGER.total_cost():.2f}", flush=True)

    p = pathlib.Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} -> {p}")
    if errs:
        print("ERRORS:"); [print("  ", k, v) for k, v in errs.most_common(8)]
    print("\n" + llm.LEDGER.report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mab")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--models", default="gpt-5.4-mini")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/axisA.jsonl")
    a = ap.parse_args()
    main(a.task, a.n, a.models.split(","), a.seeds, a.workers, a.out)
