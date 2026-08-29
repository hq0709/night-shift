"""Significance and equivalence tests for the principal contrasts.

Two of the paper's three claims are claims of *difference* (confidence and accuracy respond to the
manipulation) and one is a claim of *no difference* (discrimination does not). A confidence interval
containing zero is not evidence of absence, so the null claim is tested by two one-sided tests
against a pre-specified equivalence margin: an AUROC shift of 0.10, which is roughly a third of the
range this paper reports across deployment conditions and below any difference that would change a
deferral policy.
"""
from __future__ import annotations
import sys, json, pathlib, itertools, collections
import numpy as np
from scipy.stats import rankdata, wilcoxon, spearmanr
_R = pathlib.Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
B = 4000
MARGIN = 0.10


def auroc(c, y):
    c = np.asarray(c, float); y = np.asarray(y, int)
    if y.sum() in (0, len(y)): return np.nan
    r = rankdata(c); n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def L(p):
    return [json.loads(l) for l in open(_R / p) if json.loads(l).get("confidence") is not None]


def holm(pvals):
    idx = np.argsort(pvals); m = len(pvals); out = np.empty(m)
    run = 0.0
    for k, i in enumerate(idx):
        run = max(run, (m - k) * pvals[i])
        out[i] = min(1.0, run)
    return out


def perm_auroc(a, b, n=B):
    """Two-sided permutation test on the AUROC difference between two independent samples."""
    obs = auroc([x[0] for x in b], [x[1] for x in b]) - auroc([x[0] for x in a], [x[1] for x in a])
    pool = a + b; na = len(a); cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        d = auroc([x[0] for x in pool[na:]], [x[1] for x in pool[na:]]) - \
            auroc([x[0] for x in pool[:na]], [x[1] for x in pool[:na]])
        if not np.isnan(d) and abs(d) >= abs(obs): cnt += 1
    return obs, (cnt + 1) / (n + 1)


rows = []

# --- Claim 1: the manipulation moves confidence and accuracy (paired, by item) ---
CELLS = [("MedCalc", "gpt-5.4-mini", "results/evmanip_calc.jsonl"),
         ("MedCalc", "gpt-5.4", "results/evmanip_calc_gpt54.jsonl"),
         ("AgentClinic", "gpt-5.4-mini", "results/evidence_manip.jsonl"),
         ("AgentClinic", "gpt-5.4", "results/evidence_manip_gpt54.jsonl"),
         ("AgentClinic", "gpt-5.4-nano", "results/evidence_manip_nano.jsonl")]
for task, m, p in CELLS:
    R = L(p)
    pair = collections.defaultdict(dict)
    for r in R:
        pair[(r["uid"], r["seed"])][r["condition"]] = r
    both = [v for v in pair.values() if "withheld" in v and "given" in v]
    for what, f in (("confidence", lambda r: r["confidence"]), ("accuracy", lambda r: r["correct"])):
        w = np.array([f(v["withheld"]) for v in both], float)
        g = np.array([f(v["given"]) for v in both], float)
        d = g - w
        stat, pv = wilcoxon(d) if np.any(d != 0) else (np.nan, 1.0)
        rows.append((f"removal moves {what}", f"{task} / {m}", f"{d.mean():+.3f}",
                     "Wilcoxon signed-rank", pv, "difference"))

# --- Claim 2 (the null): the manipulation does not move discrimination. TOST. ---
for task, m, p in CELLS:
    R = L(p)
    W = [r for r in R if r["condition"] == "withheld"]
    G = [r for r in R if r["condition"] == "given"]
    d = []
    for _ in range(B):
        ia = rng.integers(0, len(W), len(W)); ib = rng.integers(0, len(G), len(G))
        a = auroc([W[i]["confidence"] for i in ia], [W[i]["correct"] for i in ia])
        b = auroc([G[i]["confidence"] for i in ib], [G[i]["correct"] for i in ib])
        if not (np.isnan(a) or np.isnan(b)): d.append(b - a)
    d = np.array(d)
    # TOST: reject non-equivalence if the 90% interval lies inside +-MARGIN
    lo90, hi90 = np.percentile(d, [5, 95])
    p_low = (d <= -MARGIN).mean(); p_high = (d >= MARGIN).mean()
    pv = max(p_low, p_high)
    rows.append(("removal does NOT move AUROC", f"{task} / {m}",
                 f"{d.mean():+.3f} [{lo90:+.3f},{hi90:+.3f}]",
                 f"TOST, margin {MARGIN}", pv, "equivalence"))

# --- Claim 3: discrimination differs between deployment regimes (independent samples) ---
REG = [("frontier vs standard", "gpt-5.4-mini", "results/stage2_gpt-5.4-mini.jsonl", "results/easy_gpt-5.4-mini.jsonl"),
       ("frontier vs standard", "gpt-5.4", "results/stage2_gpt-5.4.jsonl", "results/easy_gpt-5.4.jsonl"),
       ("static vs interactive", "gpt-5.4-mini", "results/ac_gpt-5.4-mini.jsonl", "results/inter_full_gpt-5.4-mini.jsonl"),
       ("static vs interactive", "gpt-5.4", "results/ac_gpt-5.4.jsonl", "results/inter_full_gpt-5.4.jsonl")]
for name, m, pa, pb in REG:
    a = [(r["confidence"], r["correct"]) for r in L(pa)]
    b = [(r["confidence"], r["correct"]) for r in L(pb)]
    obs, pv = perm_auroc(a, b)
    rows.append((name, m, f"{obs:+.3f}", "permutation", pv, "difference"))

# --- Claim 4: dose-response ---
D = L("results/evidence_dose.jsonl")
sp = spearmanr([r["dose"] for r in D], [r["confidence"] for r in D])
rows.append(("dose lowers confidence", "gpt-5.4-mini", f"rho {sp.statistic:+.3f}", "Spearman", sp.pvalue, "difference"))
sub = [r for r in D if r["dose"] >= 1]
byd = {k: [(r["confidence"], r["correct"]) for r in sub if r["dose"] == k] for k in (1, 3)}
obs, pv = perm_auroc(byd[1], byd[3])
rows.append(("dose does NOT move AUROC (1 vs 3)", "gpt-5.4-mini", f"{obs:+.3f}", "permutation", pv, "difference"))

# Holm within each family separately: difference claims and equivalence claims are separate families
out = []
for fam in ("difference", "equivalence"):
    grp = [r for r in rows if r[5] == fam]
    adj = holm([r[4] for r in grp])
    out += [(*r[:5], a) for r, a in zip(grp, adj)]

print(f"{'contrast':34s} {'model / cell':24s} {'estimate':24s} {'test':22s} {'p':>9s} {'p_Holm':>9s}")
for c, m, e, t, pv, pa in out:
    print(f"{c:34s} {m:24s} {e:24s} {t:22s} {pv:9.2e} {pa:9.2e}")
json.dump([dict(contrast=c, cell=m, est=e, test=t, p=pv, p_holm=pa) for c, m, e, t, pv, pa in out],
          open(_R / "results/significance.json", "w"), indent=1)
