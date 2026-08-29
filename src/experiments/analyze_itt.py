"""Claim 2 analysis: ITT effect of assigned operating point on each error family.

Design (round-1 CRITICAL fix): the independent variable is the ASSIGNED operating point, not
realized reasoning tokens. Item fixed effects absorb item difficulty, so every contrast is paired
on identical items. Realized tokens appear only as (a) the first-stage compliance check and
(b) a labelled descriptive mediator on exactly-measured models.

  python3 experiments/analyze_itt.py --path results/axisA.jsonl
"""
from __future__ import annotations
import sys, json, argparse, pathlib, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

FAMILIES = ["F1a", "F1b", "F2", "F3", "NR"]
LADDER_ORDER = ["max", "xhigh", "high", "medium", "low", "none", "off"]


def _rank(op):
    return LADDER_ORDER.index(op) if op in LADDER_ORDER else 99


def first_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Does the assigned operating point actually move the resource? Reported as a result."""
    g = df.groupby(["model", "operating_point"]).agg(
        n=("reasoning_tokens", "size"),
        rtok_median=("reasoning_tokens", "median"),
        rtok_mean=("reasoning_tokens", "mean"),
        rtok_p90=("reasoning_tokens", lambda s: float(np.percentile(s, 90))),
        out_median=("output_tokens", "median"),
        truncated=("truncated", "mean"),
        parsed=("parsed", "mean")).reset_index()
    g["op_rank"] = g["operating_point"].map(_rank)
    return g.sort_values(["model", "op_rank"])


def itt(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Within-item paired ITT: for each operating point, the family rate difference from the
    top-of-ladder reference, computed on the items where BOTH conditions were observed.
    Bootstrap CI over items (the unit of pairing), not over episodes."""
    rows = []
    for model, dm in df.groupby("model"):
        ops = sorted(dm["operating_point"].unique(), key=_rank)
        ref = ops[0]
        wide = dm.pivot_table(index="uid", columns="operating_point", values=family,
                              aggfunc="mean")
        if ref not in wide:
            continue
        for op in ops:
            if op not in wide:
                continue
            if op == ref:
                # The reference contrasted with itself is identically zero; selecting [ref, ref]
                # also yields duplicate columns, so pair[op] returns a DataFrame, not a Series.
                col = wide[ref].dropna()
                rows.append(dict(model=model, family=family, operating_point=op, ref=ref,
                                 n_items=len(col), rate=float(col.mean()),
                                 rate_ref=float(col.mean()), delta=0.0, lo=0.0, hi=0.0))
                continue
            pair = wide[[ref, op]].dropna()
            if len(pair) < 5:
                continue
            d = (pair[op] - pair[ref]).to_numpy()
            rng = np.random.default_rng(0)
            boot = rng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
            rows.append(dict(model=model, family=family, operating_point=op, ref=ref,
                             n_items=len(pair), rate=float(pair[op].mean()),
                             rate_ref=float(pair[ref].mean()), delta=float(d.mean()),
                             lo=float(np.percentile(boot, 2.5)),
                             hi=float(np.percentile(boot, 97.5))))
    return pd.DataFrame(rows)


def onset_ranks(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Bootstrap rank of onset (round-2 fix: a rank distribution, not a fixed +5% threshold).

    Onset = the highest-resource operating point at which the family's paired delta from the
    reference first exceeds zero. Resampling items gives the rank distribution across families.
    """
    out = []
    for model, dm in df.groupby("model"):
        ops = sorted(dm["operating_point"].unique(), key=_rank)
        ref = ops[0]
        uids = dm["uid"].unique()
        wides = {}
        for f in FAMILIES:
            w = dm.pivot_table(index="uid", columns="operating_point", values=f, aggfunc="mean")
            wides[f] = w.reindex(columns=ops)
        rng = np.random.default_rng(0)
        counts = {f: np.zeros(len(ops)) for f in FAMILIES}
        for _ in range(n_boot):
            samp = rng.choice(uids, size=len(uids), replace=True)
            for f in FAMILIES:
                w = wides[f].reindex(samp)
                if ref not in w:
                    continue
                base = w[ref].mean()
                idx = len(ops) - 1
                for j, op in enumerate(ops):
                    if op in w and (w[op].mean() - base) > 0:
                        idx = j
                        break
                counts[f][idx] += 1
        for f in FAMILIES:
            p = counts[f] / n_boot
            exp_rank = float(np.sum(p * np.arange(len(ops))))
            out.append(dict(model=model, family=f, expected_onset_index=exp_rank,
                            modal_onset=ops[int(np.argmax(p))],
                            dist={o: round(float(x), 3) for o, x in zip(ops, p)}))
    return pd.DataFrame(out)


def main(path):
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    for f in FAMILIES:
        if f not in df:
            df[f] = False
        df[f] = df[f].astype(float)
    print("=" * 100); print("FIRST STAGE — does assigned operating point move the resource?")
    print("=" * 100)
    print(first_stage(df).to_string(index=False))

    print("\n" + "=" * 100)
    print("ITT — paired within-item delta from top-of-ladder reference (bootstrap 95% CI over items)")
    print("=" * 100)
    allitt = pd.concat([itt(df, f) for f in FAMILIES], ignore_index=True)
    if not allitt.empty:
        allitt["sig"] = np.where((allitt.lo > 0) | (allitt.hi < 0), "*", "")
        print(allitt.sort_values(["model", "family", "operating_point"],
              key=lambda s: s.map(_rank) if s.name == "operating_point" else s).to_string(index=False))

    print("\n" + "=" * 100)
    print("ONSET RANK — bootstrap distribution (lower index = degrades at higher resource)")
    print("=" * 100)
    orank = onset_ranks(df)
    print(orank.to_string(index=False))

    out = _ROOT / "results"
    allitt.to_csv(out / "itt_effects.csv", index=False)
    orank.to_csv(out / "onset_ranks.csv", index=False)
    first_stage(df).to_csv(out / "first_stage.csv", index=False)
    print(f"\nwrote itt_effects.csv, onset_ranks.csv, first_stage.csv -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results/axisA.jsonl")
    main(ap.parse_args().path)
