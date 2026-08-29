"""B4 separation gate: is the effort ladder an ordinal RESOURCE manipulation, or noise?

Round-2 review: "If the first-stage distributions separate in expectation, it is defensible. If
adjacent levels heavily overlap, collapse them and stop calling it a compression curve."

Pre-specified gate, per model, for each adjacent pair of operating points:
    Cliff's delta >= 0.33  AND  Mann-Whitney rank-sum survives Holm correction at alpha=0.05
Levels failing the gate are merged. A model whose ladder collapses below 3 distinct points is
dropped from the confirmatory layer and reported as a negative result about the knob.
"""
from __future__ import annotations
import sys, json, argparse, pathlib, collections, statistics as st

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# Ladder order, most resource first. Used to define adjacency.
ORDER = {
    "openai_reasoning": ["xhigh", "high", "medium", "low", "none"],
    # Round-3 ruling: `off` (thinking disabled) is a separate GENERATION MODE, not the bottom rung.
    # Measured: disabling thinking RAISES sonnet-5's visible output (median 460 vs 99 at low) --
    # the model relocates reasoning into the answer instead of eliminating it.
    "anthropic_c5": ["max", "xhigh", "high", "medium", "low"],
    "anthropic_legacy": ["8192", "4096", "2048", "1024", "0"],
    "openai_chat": ["1024", "512", "256", "128", "64", "0"],
}
ALPHA = 0.05
DELTA_MIN = 0.33


def cliffs_delta(a, b):
    """P(x>y) - P(x<y). O(n log n) via sorting."""
    if not a or not b:
        return 0.0
    b_sorted = sorted(b)
    import bisect
    gt = lt = 0
    for x in a:
        lt += bisect.bisect_left(b_sorted, x)          # b < x
        gt += len(b_sorted) - bisect.bisect_right(b_sorted, x)  # b > x
    n = len(a) * len(b)
    return (lt - gt) / n           # positive => a stochastically larger


def ranksum_p(a, b):
    """Two-sided Mann-Whitney U with normal approximation and tie correction."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 1.0
    comb = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(comb)
    i = 0
    ties = []
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and comb[j + 1][0] == comb[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        ties.append(j - i + 1)
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, comb) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    N = n1 + n2
    tie_term = sum(t ** 3 - t for t in ties)
    var = n1 * n2 / 12 * ((N + 1) - tie_term / (N * (N - 1))) if N > 1 else 0
    if var <= 0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / var ** 0.5
    # two-sided normal tail
    import math
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2))))


def holm(pvals):
    """Holm-Bonferroni: returns adjusted p-values in the original order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(idx):
        v = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, v)
        adj[i] = prev
    return adj


def main(path):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("setting") != "off"]      # scored separately, see OFF_MODE
    from budgets.dial import family
    by_model = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_model[r["model"]][r["setting"]].append(r)

    verdicts = {}
    for m, cells in by_model.items():
        fam = family(m)
        order = [s for s in ORDER[fam] if s in cells]
        print("\n" + "=" * 92)
        print(f"MODEL {m}   family={fam}   levels present: {order}")
        print("=" * 92)
        print(f"{'level':8s} {'n':>4s} {'rtok_med':>9s} {'rtok_IQR':>15s} {'out_med':>8s} "
              f"{'acc':>6s} {'noans':>6s} {'breadth':>8s} {'conf':>5s} {'trunc':>6s}")
        for s in order:
            v = cells[s]
            rt = sorted(r["reasoning_tokens"] for r in v)
            q1, q3 = rt[len(rt) // 4], rt[3 * len(rt) // 4]
            cf = [r["confidence"] for r in v if r.get("confidence") is not None]
            print(f"{s:8s} {len(v):4d} {st.median(rt):9.0f} {f'[{q1},{q3}]':>15s} "
                  f"{st.median([r['output_tokens'] for r in v]):8.0f} "
                  f"{sum(r['correct'] for r in v)/len(v):6.3f} "
                  f"{sum(r['no_answer'] for r in v)/len(v):6.3f} "
                  f"{st.mean([r['breadth'] for r in v]):8.2f} "
                  f"{(st.mean(cf) if cf else float('nan')):5.2f} "
                  f"{sum(r.get('truncated',False) for r in v)/len(v):6.3f}")

        pairs, ds, ps = [], [], []
        for a, b in zip(order, order[1:]):
            xa = [r["reasoning_tokens"] for r in cells[a]]
            xb = [r["reasoning_tokens"] for r in cells[b]]
            pairs.append((a, b)); ds.append(cliffs_delta(xa, xb)); ps.append(ranksum_p(xa, xb))
        padj = holm(ps) if ps else []
        print(f"\n  {'adjacent pair':22s} {'cliffs_d':>9s} {'p_raw':>10s} {'p_holm':>10s}  gate")
        keep = [order[0]] if order else []
        merges = []
        for (a, b), d, p, pa in zip(pairs, ds, ps, padj):
            ok = (abs(d) >= DELTA_MIN) and (pa < ALPHA)
            label = f"{a} vs {b}"
            print(f"  {label:22s} {d:9.3f} {p:10.2e} {pa:10.2e}  {'PASS' if ok else 'MERGE'}")
            if ok:
                keep.append(b)
            else:
                merges.append((a, b))
        n_pts = len(keep)
        verdict = "USABLE" if n_pts >= 3 else "DROP (ladder collapsed below 3 operating points)"
        verdicts[m] = dict(points=keep, merges=merges, n_points=n_pts, verdict=verdict)
        print(f"\n  surviving operating points: {keep}   ->  {verdict}")

    # Merge, never overwrite: each model is gated in its own run, and run_axisA reads this file to
    # pick every model's surviving ladder. Overwriting silently reverts earlier models to the
    # unpruned default ladder.
    out = _ROOT / "results/separation_gate.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(verdicts)
    out.write_text(json.dumps(prev, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results/pilot_dial_range.jsonl")
    main(ap.parse_args().path)
