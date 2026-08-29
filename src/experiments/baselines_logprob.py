"""Head-to-head comparison of three confidence estimators on identical answers.

The paper's claim is about elicited confidence. A reviewer will ask whether the failure is a
property of *verbalization* or of the model's uncertainty as such. That is answerable only by
measuring alternatives on the same committed answer, so each item produces one answer and three
confidences attached to it:

  verbalized  -- the 1--10 value the model states when asked about that answer
  logprob     -- the token probability the model assigned to the answer letter it chose
  P(True)     -- the probability it assigns to the token "True" when asked whether that answer is right
                 (the self-evaluation estimator of Kadavath et al.)

All three are scored against the same correctness label, so their AUROCs are directly comparable.
"""
from __future__ import annotations
import sys, json, math, time, argparse, pathlib, collections, re
from concurrent.futures import ThreadPoolExecutor, as_completed
_R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R))
from envs import qa                                       # noqa: E402
from common import llm                                    # noqa: E402

LETTER = re.compile(r"\b([A-J])\b")


# Reasoning models spend the completion budget on reasoning before emitting anything, so a
# ceiling sized for a one-letter answer returns an empty message with a 400. Reasoning tokens do
# not appear in the logprobs payload, so a generous ceiling costs nothing for this measurement.
REASONING = ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "o1", "o3", "o4")
_CEIL = lambda m, want: 4000 if m.startswith(REASONING) else want


def _raw(model, messages, max_tok=6, logprobs=True, tries=6):
    """Rate limits get their own retry budget; without it, attrition is concentrated in whichever
    model is slowest, which is not random with respect to the comparison being made."""
    c = llm._openai()
    kw = dict(model=model, messages=messages, max_completion_tokens=_CEIL(model, max_tok))
    if logprobs:
        kw.update(logprobs=True, top_logprobs=5)
    delay = 2.0
    for k in range(tries):
        try:
            r = c.chat.completions.create(**kw)
            break
        except Exception as e:
            msg = str(e)
            if k == tries - 1:
                raise
            if "429" in msg or "rate limit" in msg.lower():
                wait = delay
                mm = re.search(r"try again in ([\d.]+)s", msg)
                if mm:
                    wait = float(mm.group(1)) + 0.5
                time.sleep(min(wait, 30)); delay = min(delay * 1.7, 30)
            elif "400" in msg:
                raise
            else:
                time.sleep(delay); delay = min(delay * 1.7, 30)
    txt = r.choices[0].message.content or ""
    lp = r.choices[0].logprobs
    llm.LEDGER.add(model, r.usage.prompt_tokens, r.usage.completion_tokens, cached=False)
    return txt, (lp.content if lp and lp.content else None), r.usage


def _norm(t):
    """Tokenizers emit "(A" or " True" as single tokens; compare on the bare content."""
    return re.sub(r"[^A-Za-z]", "", t or "").upper()


def first_token_prob(lp_content, want):
    """Probability mass the model put on `want` at the first generated token position.

    Summed over every top-k alternative that normalises to the same string, so the estimate is
    not split across "(A" and "A".
    """
    if not lp_content:
        return None
    w = _norm(want)
    tot = 0.0
    for t in (lp_content[0].top_logprobs or [lp_content[0]]):
        if _norm(t.token) == w:
            tot += math.exp(t.logprob)
    return tot


def one(model, item, split):
    q = f"{item['question']}\n\n{qa.fmt_options(item['options'])}"
    sysmsg = "You are an experienced attending physician answering a board-style medical question."

    # 1. the committed answer, constrained to a single token so the logprob is the answer's
    txt, lp, _ = _raw(model, [{"role": "system", "content": sysmsg},
                              {"role": "user", "content": q + "\n\nRespond with the option letter only."}])
    m = LETTER.search((txt or "").upper())
    if not m:
        return None
    letter = m.group(1)
    p_ans = first_token_prob(lp, letter)

    # 2. verbalized confidence about that same answer
    vtxt, _, _ = _raw(model, [{"role": "system", "content": sysmsg},
                              {"role": "user", "content":
                               f"{q}\n\nA physician answered ({letter}). On a scale of 1 to 10, how "
                               f"confident are you that ({letter}) is correct? Reply with the number only."}],
                      max_tok=5, logprobs=False)
    vm = re.search(r"\b(10|[1-9])\b", vtxt or "")
    verbal = int(vm.group(1)) if vm else None

    # 3. P(True): the self-evaluation estimator, read off the token distribution
    ttxt, tlp, _ = _raw(model, [{"role": "system", "content": sysmsg},
                                {"role": "user", "content":
                                 f"{q}\n\nProposed answer: ({letter}).\n\nIs the proposed answer "
                                 f"correct? Reply True or False."}], max_tok=4)
    p_true = first_token_prob(tlp, "True")
    if p_true is None:
        p_true = 1.0 if (ttxt or "").strip().lower().startswith("t") else 0.0

    return dict(model=model, split=split, uid=item["uid"], source=item.get("source"),
                letter=letter, correct=int(letter == item["answer_idx"]),
                verbal=verbal, p_answer=p_ans, p_true=p_true)


def main(models, n, workers, out):
    sets = {"frontier": qa.stratified_sample(qa.load_hard(split="test_hard"), n, seed=0),
            "standard": qa.stratified_sample(qa.load_hard(split="test"), n, seed=0)}
    jobs = [(m, it, s) for m in models for s, items in sets.items() for it in items]
    print(f"{len(jobs)} items x 3 calls = {len(jobs)*3} calls")
    rows, errs = [], collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, m, it, s): (m, s) for m, it, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                if r:
                    rows.append(r)
            except Exception as e:
                errs[str(e)[:70]] += 1
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  ${llm.global_spend_usd():.2f}", flush=True)
    with open(_R / out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {out}")
    for k, v in errs.most_common(4):
        print("  ERR", k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-4o-mini,gpt-5.4-mini")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="results/baselines_logprob.jsonl")
    a = ap.parse_args()
    main(a.models.split(","), a.n, a.workers, a.out)
