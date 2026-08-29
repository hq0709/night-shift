"""Three confidence estimators on the same committed answers."""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as S
import matplotlib.pyplot as plt
from scipy.stats import rankdata
_R = pathlib.Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)

def auroc(c, y):
    c = np.asarray(c, float); y = np.asarray(y, int)
    if y.sum() in (0, len(y)): return np.nan
    r = rankdata(c); n1 = y.sum()
    return (r[y == 1].sum() - n1*(n1+1)/2) / (n1*(len(y)-n1))

def boot(c, y, n=4000):
    c = np.asarray(c, float); y = np.asarray(y, int); v = []
    for _ in range(n):
        i = rng.integers(0, len(c), len(c)); a = auroc(c[i], y[i])
        if not np.isnan(a): v.append(a)
    return auroc(c, y), np.percentile(v, 2.5), np.percentile(v, 97.5)

R = [json.loads(l) for l in open(_R / "results/baselines_logprob.jsonl")]
R = [r for r in R if r["verbal"] is not None and r["p_answer"] is not None and r["p_true"] is not None]
EST = [("verbalized 1\u201310", "verbal", S.BLUE),
       ("answer token probability", "p_answer", S.ORANGE),
       ("P(True) self-evaluation", "p_true", S.GREEN)]
MODELS = ["gpt-4o-mini", "gpt-5.4-mini"]

fig, axes = plt.subplots(1, 2, figsize=(S.TXT, 2.15), sharey=True)
W = 0.24
for ax, m in zip(axes, MODELS):
    for j, (lbl, key, col) in enumerate(EST):
        xs, mus, los, his = [], [], [], []
        for i, sp in enumerate(("standard", "frontier")):
            g = [r for r in R if r["model"] == m and r["split"] == sp]
            a, lo, hi = boot([r[key] for r in g], [r["correct"] for r in g])
            xs.append(i + (j - 1) * W); mus.append(a); los.append(a - lo); his.append(hi - a)
        ax.bar(xs, mus, W * .88, color=col, label=lbl if m == MODELS[0] else None)
        ax.errorbar(xs, mus, yerr=[los, his], fmt="none", ecolor="0.2", elinewidth=.8, capsize=1.8)
    ax.axhline(.5, color="0.35", ls="--", lw=.7, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["standard\ndifficulty", "capability\nfrontier"])
    ax.set_title(m, pad=4); ax.set_ylim(0.25, 0.88)
axes[0].set_ylabel("AUROC")
axes[0].text(-0.34, .508, "chance", fontsize=6, color="0.35", va="bottom", ha="left")
S.panel(axes[0], "a", dx=-0.20); S.panel(axes[1], "b", dx=-0.08)
fig.legend(loc="lower center", bbox_to_anchor=(.53, -.20), ncol=3, handlelength=1.2, handleheight=.8)
fig.subplots_adjust(wspace=.08)
S.save(fig, "fig_baselines")

print(f"{'model':14s} {'split':10s} " + "  ".join(f"{l:>26s}" for l, _, _ in EST))
for m in MODELS:
    for sp in ("standard", "frontier"):
        g = [r for r in R if r["model"] == m and r["split"] == sp]
        cells = []
        for _, k, _ in EST:
            a, lo, hi = boot([r[k] for r in g], [r["correct"] for r in g])
            cells.append(f"{a:.3f} [{lo:.3f},{hi:.3f}]")
        print(f"{m:14s} {sp:10s} " + "  ".join(f"{c:>26s}" for c in cells))
