"""Claim 1 analysis: diagonal dominance at pre-calibrated iso-cost.

Pre-registered endpoints (round-3 ruling — dominance, NOT off-diagonal nulls):
    effort-compressed side (A) :  F1a > F1b
    turn-compressed  side (B) :  F1b > F1a
Estimated as a within-item paired contrast on the SAME cases under both assigned conditions, with
bootstrap CIs over items (the unit of pairing). A null is interpretable because equivalence bounds
are pre-specified rather than inferred after the fact.

Also reports the two anti-tautology instruments and the calibration check.

  python3 experiments/analyze_dominance.py --path results/claim1_gpt54mini.jsonl
"""
from __future__ import annotations
import sys, json, argparse, pathlib, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
_ROOT = pathlib.Path(__file__).resolve().parents[1]

EQUIV = 0.05          # pre-specified equivalence bound on the dominance difference
NBOOT = 5000


def boot_ci(x, n=NBOOT, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main(path):
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    for c in ("F1a", "F1b", "F2", "F3", "NR"):
        if c not in df:
            df[c] = False
        df[c] = df[c].astype(float)
    df["dom"] = df["F1a"] - df["F1b"]

    print("=" * 96)
    print("CALIBRATION CHECK — did the frozen iso-cost pair actually cost the same on held-out data?")
    print("=" * 96)
    cc = df.groupby(["model", "pair_index", "side"]).agg(
        n=("uid", "size"), doctor_out_tokens=("doctor_output_tokens", "mean"),
        reasoning_tokens=("reasoning_tokens", "mean"), actions=("n_actions", "mean")).reset_index()
    print(cc.to_string(index=False))
    for (m, pi), g in cc.groupby(["model", "pair_index"]):
        if len(g) == 2:
            a, b = g["doctor_out_tokens"].tolist()
            gap = abs(a - b) / max(a, b) if max(a, b) else float("nan")
            print(f"\n  realized output-token gap for {m} pair {pi}: {gap:.1%} "
                  f"({'within' if gap <= .15 else 'OUTSIDE'} the ±15% design band)")

    print("\n" + "=" * 96)
    print("PRE-REGISTERED DOMINANCE — within-item paired, bootstrap 95% CI over items")
    print("=" * 96)
    print(f"{'model':16s} {'pair':>5s} {'side':13s} {'n':>5s} {'F1a':>6s} {'F1b':>6s} "
          f"{'dom':>7s} {'lo':>7s} {'hi':>7s}  verdict")
    verdicts = []
    for (m, pi, side), g in df.groupby(["model", "pair_index", "side"]):
        per_item = g.groupby("uid")["dom"].mean()
        mean, lo, hi = boot_ci(per_item.to_numpy())
        want_pos = side.startswith("A")
        if lo > 0:
            v = "POSITIVE" + (" ✓" if want_pos else " ✗ (wrong sign)")
        elif hi < 0:
            v = "NEGATIVE" + (" ✓" if not want_pos else " ✗ (wrong sign)")
        elif abs(mean) < EQUIV and max(abs(lo), abs(hi)) < 2 * EQUIV:
            v = "NULL (equivalent)"
        else:
            v = "INCONCLUSIVE"
        verdicts.append((m, pi, side, v))
        print(f"{m:16s} {pi:5d} {side:13s} {len(per_item):5d} "
              f"{g['F1a'].mean():6.3f} {g['F1b'].mean():6.3f} "
              f"{mean:7.3f} {lo:7.3f} {hi:7.3f}  {v}")

    print("\n  Claim 1 is supported only when A_effortcut is POSITIVE and B_turncut is NEGATIVE.")
    ok = {(m, pi): [] for m, pi, _, _ in verdicts}
    for m, pi, side, v in verdicts:
        ok[(m, pi)].append(v.endswith("✓"))
    for k, vs in ok.items():
        print(f"    {k[0]} pair {k[1]}: {'SUPPORTED' if len(vs) == 2 and all(vs) else 'NOT supported'}")

    print("\n" + "=" * 96)
    print("ANTI-TAUTOLOGY INSTRUMENTS")
    print("=" * 96)
    print("(i) acquisition efficiency = key findings obtained per action spent.")
    print("    Turn compression shrinks the denominator by construction; effort compression should")
    print("    shrink the NUMERATOR at a fixed denominator. A tautology cannot produce the latter.")
    for (m, pi, side), g in df.groupby(["model", "pair_index", "side"]):
        ae = g["acq_eff"].dropna()
        mean, lo, hi = boot_ci(ae.to_numpy())
        print(f"    {m:16s} pair {pi} {side:13s} acq_eff={mean:.3f} [{lo:.3f},{hi:.3f}]  "
              f"actions/case={g['n_actions'].mean():.2f}")

    print("\n(ii) off-diagonal magnitudes (reported, not required to be zero — the channels are")
    print("     mediated, so the pre-registered test is dominance, not nullity):")
    for (m, pi, side), g in df.groupby(["model", "pair_index", "side"]):
        print(f"    {m:16s} pair {pi} {side:13s} F1a={g['F1a'].mean():.3f} F1b={g['F1b'].mean():.3f} "
              f"F2={g['F2'].mean():.3f} F3={g['F3'].mean():.3f} NR={g['NR'].mean():.3f} "
              f"acc={g['correct'].mean():.3f}")

    out = _ROOT / "results/claim1_dominance.csv"
    df.groupby(["model", "pair_index", "side"]).agg(
        n=("uid", "size"), F1a=("F1a", "mean"), F1b=("F1b", "mean"), dom=("dom", "mean"),
        F2=("F2", "mean"), F3=("F3", "mean"), NR=("NR", "mean"), acc=("correct", "mean"),
        acq_eff=("acq_eff", "mean"), actions=("n_actions", "mean"),
        out_tokens=("doctor_output_tokens", "mean")).to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results/claim1_gpt54mini.jsonl")
    main(ap.parse_args().path)
