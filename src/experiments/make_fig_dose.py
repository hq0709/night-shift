"""Dose-response figure: the central contrast, graded rather than binary."""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as S
import matplotlib.pyplot as plt
from scipy.stats import rankdata
_R = pathlib.Path(__file__).resolve().parents[1]
rng = np.random.default_rng(1)

def auroc(c, y):
    c = np.asarray(c, float); y = np.asarray(y, int)
    if y.sum() in (0, len(y)): return np.nan
    r = rankdata(c); n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))

R = [json.loads(l) for l in open(_R / "results/evidence_dose.jsonl")
     if json.loads(l).get("confidence") is not None]
DOSES = sorted({r["dose"] for r in R})
items = sorted({r["uid"] for r in R})

fig, axes = plt.subplots(1, 3, figsize=(S.TXT, 2.15))

# (a) confidence, with one faint line per item so the within-item response is visible
by = {}
for r in R:
    by.setdefault(r["uid"], {}).setdefault(r["dose"], []).append(r["confidence"])
for uid, d in by.items():
    if len(d) == len(DOSES):
        axes[0].plot(DOSES, [np.mean(d[k]) for k in DOSES], color=S.BLUE, lw=.35, alpha=.13,
                     solid_capstyle="round")
mu = [np.mean([r["confidence"] for r in R if r["dose"] == k]) for k in DOSES]
se = [np.std([r["confidence"] for r in R if r["dose"] == k], ddof=1) /
      np.sqrt(sum(r["dose"] == k for r in R)) for k in DOSES]
axes[0].errorbar(DOSES, mu, yerr=[1.96 * x for x in se], color=S.RED, lw=1.6, marker="o",
                 capsize=2, zorder=5)
axes[0].set_ylabel("stated confidence"); axes[0].set_ylim(0, 10.5)

# (b) accuracy
acc = [np.mean([r["correct"] for r in R if r["dose"] == k]) for k in DOSES]
lo, hi = [], []
for k in DOSES:
    v = np.array([r["correct"] for r in R if r["dose"] == k], float)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(3000)]
    lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
axes[1].errorbar(DOSES, acc, yerr=[np.array(acc) - lo, np.array(hi) - np.array(acc)],
                 color=S.RED, lw=1.6, marker="o", capsize=2)
axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0, 1)

# (c) discrimination, items resampled so the interval respects the repeated-measures design
au, alo, ahi = [], [], []
for k in DOSES:
    g = [r for r in R if r["dose"] == k]
    au.append(auroc([r["confidence"] for r in g], [r["correct"] for r in g]))
    b = []
    for _ in range(3000):
        pick = set(rng.choice(items, len(items), replace=True))
        sub = [r for r in g if r["uid"] in pick]
        v = auroc([r["confidence"] for r in sub], [r["correct"] for r in sub])
        if not np.isnan(v): b.append(v)
    alo.append(np.percentile(b, 2.5)); ahi.append(np.percentile(b, 97.5))
axes[2].errorbar(DOSES, au, yerr=[np.array(au) - alo, np.array(ahi) - np.array(au)],
                 color=S.RED, lw=1.6, marker="o", capsize=2)
axes[2].axhline(.5, color="0.55", ls="--", lw=.6)
axes[2].set_ylabel("AUROC"); axes[2].set_ylim(.42, 1.0)
axes[2].axvspan(.65, 3.35, color=S.BLUE, alpha=.07, lw=0, zorder=0)
axes[2].annotate("flat across\nremoval amount", xy=(2.05, .93), fontsize=6.2, ha="center",
                 color=S.BLUE)

for j, ax in enumerate(axes):
    ax.set_xlabel("required inputs removed"); ax.set_xticks(DOSES); ax.set_xlim(-.25, 3.25)
    S.panel(ax, "abc"[j], dx=-0.26)
fig.subplots_adjust(wspace=.36)
S.save(fig, "fig_dose")
print(f"{'dose':>4} {'conf':>6} {'acc':>6} {'AUROC':>6}  95% CI")
for i, k in enumerate(DOSES):
    print(f"{k:>4} {mu[i]:>6.2f} {acc[i]:>6.3f} {au[i]:>6.3f}  [{alo[i]:.3f}, {ahi[i]:.3f}]")
