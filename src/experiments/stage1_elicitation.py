"""Stage 1 / Q2: is the null discrimination an artefact of HOW confidence is asked?

The first reviewer question will be "your prompt was bad". Four elicitation formats on identical
items and identical operating points; if AUROC stays at chance across all of them, the finding is
about the model, not the prompt. If one format works, the paper becomes "how to ask" instead.

  A  integer   : the format used so far, 1-10
  B  probability: "probability this answer is correct", 0-100
  C  verbal    : certain / fairly confident / uncertain / guessing
  D  post-hoc  : answer first, then a SEPARATE call judging that answer  (2 calls/item)
"""
from __future__ import annotations
import sys, json, argparse, pathlib, collections
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from envs import qa                 # noqa: E402
from budgets import dial            # noqa: E402
from common import llm              # noqa: E402

BASE = {"type": "object", "additionalProperties": False,
        "properties": {
            "differential": {"type": "array", "items": {"type": "string"},
                             "description": "Options you actually considered, at most 5."},
            "final_answer": {"type": "string", "description": "The single option letter."}},
        "required": ["differential", "final_answer"]}


def schema_for(fmt: str) -> dict:
    s = json.loads(json.dumps(BASE))
    p, r = s["properties"], s["required"]
    if fmt == "A_integer":
        p["confidence"] = {"type": "integer", "description":
                           "Confidence that final_answer is correct, 1 (guess) to 10 (certain)."}
        r.append("confidence")
    elif fmt == "B_probability":
        p["confidence"] = {"type": "integer", "description":
                           "The probability, 0 to 100, that final_answer is correct. Treat this as "
                           "a calibrated probability: of all answers you label 70, about 70% should "
                           "be right."}
        r.append("confidence")
    elif fmt == "C_verbal":
        p["confidence"] = {"type": "string", "enum": ["guessing", "uncertain",
                                                      "fairly confident", "certain"],
                           "description": "How sure you are that final_answer is correct."}
        r.append("confidence")
    return s


VERBAL = {"guessing": 1, "uncertain": 4, "fairly confident": 7, "certain": 10}
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False,
                "properties": {"correct": {"type": "boolean",
                               "description": "Is the proposed answer correct?"},
                               "confidence": {"type": "integer",
                               "description": "Confidence in that judgement, 1-10."}},
                "required": ["correct", "confidence"]}


def norm_conf(fmt: str, raw) -> float | None:
    if raw is None:
        return None
    try:
        if fmt == "C_verbal":
            return float(VERBAL.get(str(raw).strip().lower(), float("nan")))
        v = float(raw)
        return v / 10.0 if fmt == "B_probability" else v   # both onto a 0-10 scale
    except Exception:
        return None


def run(models, n, ops, fmts, workers, out):
    items = qa.stratified_sample(qa.load_hard(), n, seed=0)
    jobs = [(m, o, f, it) for m in models for o in ops for f in fmts for it in items]
    print(f"{len(items)} items x {len(ops)} ops x {len(fmts)} formats -> {len(jobs)} episodes")

    def one(j):
        m, o, fmt, it = j
        if fmt == "D_posthoc":
            # answer with NO confidence field, then judge it in a separate call
            ep = dial.ask(m, qa.prompt_structured(it), axis="A_effort", setting=o, seed=0,
                          system=qa.DOCTOR_SYSTEM, item_id=f"{it['uid']}|D|ans",
                          max_tokens=64000, structured=False)
            import re
            mm = re.findall(r"\b([A-Z])\b", (ep.text or "")[-200:])
            letter = mm[-1] if mm and mm[-1] in it["options"] else None
            jp = (f"{it['question']}\n\n{qa.fmt_options(it['options'])}\n\n"
                  f"A model answered: {letter}\n\nIs that answer correct? Return JSON.")
            jep = dial.ask(m, jp, axis="A_effort", setting=o, seed=0,
                           item_id=f"{it['uid']}|D|judge", max_tokens=32000, structured=False)
            try:
                jo = json.loads(jep.text[jep.text.index("{"):jep.text.rindex("}") + 1])
                conf = 10.0 if jo.get("correct") else 1.0
                conf = conf if jo.get("confidence") is None else (
                    float(jo["confidence"]) if jo.get("correct") else 11 - float(jo["confidence"]))
            except Exception:
                conf = None
            rt, ot = ep.reasoning_tokens + jep.reasoning_tokens, ep.output_tokens + jep.output_tokens
        else:
            from envs import schema as _s
            orig = _s.EPISODE_SCHEMA
            _s.EPISODE_SCHEMA = schema_for(fmt)
            try:
                ep = dial.ask(m, qa.prompt_structured(it), axis="A_effort", setting=o, seed=0,
                              system=qa.DOCTOR_SYSTEM, item_id=f"{it['uid']}|{fmt}",
                              max_tokens=64000, structured="letter")
            finally:
                _s.EPISODE_SCHEMA = orig
            try:
                obj = json.loads(ep.text[ep.text.index("{"):ep.text.rindex("}") + 1])
            except Exception:
                obj = {}
            fa = str(obj.get("final_answer", "")).strip().upper()
            letter = next((c for c in fa if c in it["options"]), None)
            conf = norm_conf(fmt, obj.get("confidence"))
            rt, ot = ep.reasoning_tokens, ep.output_tokens
        return dict(model=m, operating_point=o, fmt=fmt, uid=it["uid"], source=it["source"],
                    letter=letter, gold=it["answer_idx"],
                    correct=(letter == it["answer_idx"]) if letter else False,
                    confidence=conf, reasoning_tokens=rt, output_tokens=ot)

    rows, errs = [], collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                errs[f"{futs[f][2]}|{str(e)[:70]}"] += 1
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  ${llm.global_spend_usd():.2f}", flush=True)

    p = pathlib.Path(out); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} -> {p}")
    if errs:
        [print("  ERR", k, v) for k, v in errs.most_common(6)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-5.4-mini")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ops", default="high,none")
    ap.add_argument("--fmts", default="A_integer,B_probability,C_verbal,D_posthoc")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/stage1_elicitation.jsonl")
    a = ap.parse_args()
    run(a.models.split(","), a.n, a.ops.split(","), a.fmts.split(","), a.workers, a.out)
