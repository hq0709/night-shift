"""MedCalc-Bench: the procedural task family.

This is the family where COMMISSION splits cleanly into two mechanisms the diagnostic benchmarks
cannot separate: choosing the wrong formula (a knowledge error) versus executing the right formula
badly (a slip). MedCalc ships the ground-truth explanation naming the formula, and an acceptance
band (Lower/Upper Limit) rather than an exact value, so both are checkable without a judge.
"""
from __future__ import annotations
import re, random, pathlib
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = _ROOT / "external/MedCalc-Bench/datasets/test_data.csv"


def load(n: int | None = None, seed: int = 0) -> list[dict]:
    df = pd.read_csv(CSV)
    items = []
    for _, r in df.iterrows():
        items.append({
            "uid": f"medcalc::{r['Row Number']}",
            "source": "MedCalc",
            "clinical": True,
            "calculator": str(r["Calculator Name"]),
            "category": str(r["Category"]),
            "output_type": str(r["Output Type"]),
            "note": str(r["Patient Note"]),
            "question": str(r["Question"]),
            "gold": str(r["Ground Truth Answer"]),
            "lo": r.get("Lower Limit"), "hi": r.get("Upper Limit"),
            "explanation": str(r.get("Ground Truth Explanation", "")),
        })
    if n:
        rng = random.Random(seed)
        by = {}
        for it in items:
            by.setdefault(it["category"], []).append(it)
        out, cats = [], sorted(by)
        per = max(1, n // len(cats))
        for c in cats:
            pool = sorted(by[c], key=lambda x: x["uid"])
            rng.shuffle(pool)
            out += pool[:per]
        rng.shuffle(out)
        items = out[:n]
    return items


# Same fixed measurement contract as the diagnostic families: identical elicitation at every
# operating point, with `differential` repurposed as the candidate formulas considered so the
# consideration channel stays measurable here too.
CONTRACT = (
    "Return your answer as JSON matching the required schema. Put the formula or rule you used in "
    "`red_flag_considered`, any other formulas you weighed in `differential`, the numeric result in "
    "`final_answer` (digits only, no units), and your confidence in `confidence`. Fill every field."
)


def prompt(item: dict) -> str:
    return f"{item['note']}\n\n{item['question']}\n\n{CONTRACT}"


_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def parse_value(s: str) -> float | None:
    if s is None:
        return None
    m = _NUM.findall(str(s).replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def score(item: dict, answer: str) -> dict:
    """Correct = inside MedCalc's own acceptance band (it ships one because these are continuous
    quantities; an exact-match rule would score rounding as a clinical error)."""
    v = parse_value(answer)
    if v is None:
        return {"correct": False, "value": None, "in_band": False, "rel_err": None}
    lo, hi = item.get("lo"), item.get("hi")
    gold = parse_value(item["gold"])
    try:
        in_band = (lo is not None and hi is not None and not pd.isna(lo) and not pd.isna(hi)
                   and float(lo) <= v <= float(hi))
    except Exception:
        in_band = False
    if not in_band and gold not in (None, 0):
        in_band = abs(v - gold) / abs(gold) <= 0.01
    rel = abs(v - gold) / abs(gold) if (gold not in (None, 0) and v is not None) else None
    return {"correct": bool(in_band), "value": v, "in_band": bool(in_band), "rel_err": rel}


_FORMULA_STOP = {"the", "for", "and", "with", "using", "patient", "formula", "equation",
                 "computing", "calculated", "given", "score", "index", "value"}


def _toks(s):
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()
            if len(t) > 3 and t not in _FORMULA_STOP}


def right_formula(item: dict, stated: str, differential: list[str]) -> bool:
    """Did the model name the correct calculator? Matched against the calculator name and the
    ground-truth explanation, which states the formula explicitly."""
    target = _toks(item["calculator"])
    if not target:
        return False
    need = max(1, (len(target) + 1) // 2)
    for cand in [stated] + list(differential or []):
        if len(target & _toks(cand)) >= need:
            return True
    return False


def slip_vs_knowledge(item: dict, answer: str, stated: str, differential: list[str]) -> str:
    """The procedural split. 'ok' | 'slip' (right formula, wrong arithmetic) |
    'knowledge' (wrong formula) | 'unparsed'."""
    sc = score(item, answer)
    if sc["value"] is None:
        return "unparsed"
    if sc["correct"]:
        return "ok"
    return "slip" if right_formula(item, stated, differential) else "knowledge"
