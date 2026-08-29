"""Per-case key discriminating findings — the ground truth for F1b (acquisition omission).

This is the ONLY LLM-derived input to the family classifier, and it is deliberately built offline
from the case file plus the gold diagnosis alone. The annotator model never sees a doctor
transcript, so it cannot advantage or disadvantage any condition: it is case annotation, not
episode judging. Output is hand-spot-checked before use.

  python3 scripts/build_key_tests.py --which medqa --model claude-opus-5
"""
from __future__ import annotations
import sys, json, argparse, pathlib, collections
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from envs import clinic          # noqa: E402
from budgets import dial         # noqa: E402
from common import llm           # noqa: E402

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "key_findings": {
            "type": "array",
            "description": ("The findings from the available list that a competent clinician must "
                            "obtain to establish or exclude the correct diagnosis. Copy the "
                            "available-finding labels verbatim. Between 1 and 3 of them."),
            "items": {"type": "string"}},
        "key_question": {
            "type": "string",
            "description": ("The single most discriminating question to ask the patient for this "
                            "diagnosis, in plain clinical language.")},
    },
    "required": ["key_findings", "key_question"],
}

PROMPT = """You are annotating an OSCE case for a study of clinical information-seeking.

Correct diagnosis: {gold}

Presenting objective: {obj}

Findings that CAN be obtained in this case (verbatim labels):
{avail}

Task: identify which of those findings are the KEY DISCRIMINATORS for the correct diagnosis — the
ones whose absence would leave a competent clinician unable to establish or exclude it. Copy the
labels verbatim from the list. Also give the single most discriminating history question.

Do not list vital signs unless they are genuinely discriminating for this diagnosis.
Return JSON matching the required schema."""


def main(which, model, workers, limit):
    cases = clinic.load_cases(which)
    if limit:
        cases = cases[:limit]
    out_p = _ROOT / f"data/key_tests_{which}.jsonl"

    def one(c):
        avail = clinic.available_tests(c)
        if not avail:
            return dict(uid=c["uid"], gold=c["gold"], key_findings=[], key_question="", n_avail=0)
        p = PROMPT.format(gold=c["gold"], obj=c["objective"],
                          avail="\n".join(f"- {a}" for a in avail))
        if model.startswith("claude"):
            r = llm._anthropic().messages.create(
                model=model, max_tokens=8000, thinking={"type": "adaptive"},
                extra_body={"output_config": {"effort": "medium",
                                              "format": {"type": "json_schema", "schema": SCHEMA}}},
                messages=[{"role": "user", "content": p}])
            txt = "".join(b.text for b in r.content if b.type == "text")
            llm.LEDGER.add(model, r.usage.input_tokens, r.usage.output_tokens, cached=False)
        else:
            r = llm._openai().chat.completions.create(
                model=model, messages=[{"role": "user", "content": p}],
                max_completion_tokens=8000, reasoning_effort="medium",
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "key_tests", "strict": True,
                                                 "schema": SCHEMA}})
            txt = r.choices[0].message.content or ""
            llm.LEDGER.add(model, r.usage.prompt_tokens, r.usage.completion_tokens, cached=False)
        try:
            o = json.loads(txt)
        except Exception:
            o = {"key_findings": [], "key_question": ""}
        return dict(uid=c["uid"], gold=c["gold"], n_avail=len(avail),
                    key_findings=o.get("key_findings", []), key_question=o.get("key_question", ""))

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, c): c for c in cases}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                print("  ERR", str(e)[:120])
            if i % 25 == 0:
                print(f"  {i}/{len(cases)}  ${llm.LEDGER.total_cost():.2f}", flush=True)

    rows.sort(key=lambda r: r["uid"])
    with open(out_p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ok = sum(1 for r in rows if r["key_findings"])
    print(f"\nwrote {len(rows)} -> {out_p}   ({n_ok} with >=1 key finding)")
    print("key-finding count distribution:",
          dict(collections.Counter(len(r["key_findings"]) for r in rows)))
    print(llm.LEDGER.report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="medqa")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.which, a.model, a.workers, a.limit)
