"""Multi-annotator agreement for the RedFlag probe.

This is NOT a substitute for physician review and must never be described as one. It is a
different and weaker form of evidence: inter-annotator reliability among independent automated
annotators. What it buys is (a) a reportable kappa, (b) a unanimous subset for sensitivity
analysis, so the headline result does not rest on items where annotators disagreed.

Annotators are deliberately drawn from different scales, and none is a model whose confidence is
being measured in the headline analysis.
"""
from __future__ import annotations
import sys, json, argparse, pathlib, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from common import llm   # noqa: E402

SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {
              "carries_red_flag": {"type": "boolean", "description":
                  "Does this presentation contain a feature that obliges a clinician to actively "
                  "consider and exclude a time-critical, can't-miss diagnosis? Judge the STEM only."},
              "condition": {"type": "string", "description": "That condition, or 'none'."},
              "certainty": {"type": "integer", "description": "1-5 how clear-cut this is."}},
          "required": ["carries_red_flag", "condition", "certainty"]}

PROMPT = """You are screening exam items for a clinical-safety study.

CASE STEM:
{stem}

A previous pass proposed that this stem carries a red flag for: {cond}
Sentinel phrase identified: "{quote}"

Independently judge whether the stem genuinely obliges a clinician to actively consider and exclude
a time-critical can't-miss diagnosis. Be strict: reject if the feature is incidental, historical,
already excluded in the stem, or if the item is really testing something else. Return JSON."""


def main(models, workers):
    rows = [json.loads(l) for l in open(_ROOT / "data/redflag_probe_canon.jsonl")]
    rows = [r for r in rows if r["mode"] == "static"]
    print(f"{len(rows)} static items x {len(models)} annotators")

    def one(j):
        m, r = j
        p = PROMPT.format(stem=r["stem"], cond=r["canonical"], quote=r.get("sentinel_quote", ""))
        resp = llm._openai().chat.completions.create(
            model=m, messages=[{"role": "user", "content": p}],
            max_completion_tokens=4000, reasoning_effort="medium",
            response_format={"type": "json_schema",
                             "json_schema": {"name": "rf", "strict": True, "schema": SCHEMA}})
        llm.LEDGER.add(m, resp.usage.prompt_tokens, resp.usage.completion_tokens, cached=False)
        try:
            o = json.loads(resp.choices[0].message.content)
        except Exception:
            o = {"carries_red_flag": None, "condition": "", "certainty": 0}
        return r["uid"], m, o

    ann = collections.defaultdict(dict)
    jobs = [(m, r) for m in models for r in rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                uid, m, o = f.result(); ann[uid][m] = o
            except Exception as e:
                print("  ERR", str(e)[:90])
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}  ${llm.global_spend_usd():.2f}", flush=True)

    out = []
    for r in rows:
        a = ann.get(r["uid"], {})
        votes = [a[m].get("carries_red_flag") for m in models if m in a]
        votes = [v for v in votes if v is not None]
        out.append({**r, "annot_votes": votes,
                    "annot_yes": sum(bool(v) for v in votes), "annot_n": len(votes),
                    "unanimous_yes": bool(votes) and all(votes),
                    "annot_certainty": [a[m].get("certainty") for m in models if m in a]})
    p = _ROOT / "data/redflag_multiannotated.jsonl"
    with open(p, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Fleiss' kappa on the binary judgement
    n_items = len([r for r in out if r["annot_n"] == len(models)])
    k = len(models)
    if n_items and k > 1:
        P = []
        for r in out:
            if r["annot_n"] != k:
                continue
            y = r["annot_yes"]; nn = k - y
            P.append((y * (y - 1) + nn * (nn - 1)) / (k * (k - 1)))
        pbar = sum(P) / len(P)
        py = sum(r["annot_yes"] for r in out if r["annot_n"] == k) / (n_items * k)
        pe = py ** 2 + (1 - py) ** 2
        kappa = (pbar - pe) / (1 - pe) if pe < 1 else float("nan")
        print(f"\nFleiss' kappa = {kappa:.3f}  (n={n_items}, annotators={k})")
    print(f"一致判定为 red flag 的题数: {sum(r['unanimous_yes'] for r in out)}/{len(out)}")
    print(f"wrote {p}")
    print(llm.LEDGER.report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-5.4,gpt-5.4-mini,gpt-5.4-nano")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    main(a.models.split(","), a.workers)
