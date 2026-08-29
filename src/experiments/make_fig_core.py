"""Figure 1, the paper's central result: what the removal manipulation moves and what it does not."""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as S
import matplotlib.pyplot as plt
from scipy.stats import rankdata
_R = pathlib.Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
B = 4000

def auroc(c, y):
    c = np.asarray(c, float); y = np.asarray(y, int)
    if y.sum() in (0, len(y)): return np.nan
    r = rankdata(c); n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))

def L(p):
    return [json.loads(l) for l in open(_R / p) if json.loads(l).get("confidence") is not None]

# Red = the case was rendered unanswerable by the removal; blue = it stayed answerable.
CELLS = [("MedCalc",     "gpt-5.4-mini", "results/evmanip_calc.jsonl",        S.RED,  1),
         ("MedCalc",     "gpt-5.4",      "results/evmanip_calc_gpt54.jsonl",  S.RED,  0),
         ("AgentClinic", "gpt-5.4-mini", "results/evidence_manip.jsonl",      S.BLUE, 1),
         ("AgentClinic", "gpt-5.4",      "results/evidence_manip_gpt54.jsonl",S.BLUE, 0),
         ("AgentClinic", "gpt-5.4-nano", "results/evidence_manip_nano.jsonl", S.BLUE, 0)]

fig, axes = plt.subplots(1, 3, figsize=(S.TXT, 2.35))
labels, rows = [], []
for k, (task, m, p, col, dark) in enumerate(CELLS):
    r = L(p)
    W = [x for x in r if x["condition"] == "withheld"]
    G = [x for x in r if x["condition"] == "given"]
    wa = np.array([x["correct"] for x in W], float); ga = np.array([x["correct"] for x in G], float)
    wc = np.array([x["confidence"] for x in W], float); gc = np.array([x["confidence"] for x in G], float)
    # One bootstrap loop resamples both arms and yields all three contrasts, so the
    # intervals in the three panels come from the same resampling and are comparable.
    da, dc, du = [], [], []
    for _ in range(B):
        ia = rng.integers(0, len(W), len(W)); ib = rng.integers(0, len(G), len(G))
        da.append(ga[ib].mean() - wa[ia].mean())
        dc.append(gc[ib].mean() - wc[ia].mean())
        a = auroc(wc[ia], wa[ia]); b = auroc(gc[ib], ga[ib])
        if not (np.isnan(a) or np.isnan(b)): du.append(b - a)
    y = -k
    for ax, obs, dist in ((axes[0], ga.mean() - wa.mean(), da),
                          (axes[1], gc.mean() - wc.mean(), dc),
                          (axes[2], float(np.mean(du)), du)):
        lo, hi = np.percentile(dist, [2.5, 97.5])
        if ax is axes[2]:
            ax.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="butt")
            ax.plot(obs, y, "o", color=col, mec="white", mew=.5, zorder=3)
        else:
            ax.barh(y, obs, color=col, height=.58, alpha=1.0 if dark else .62,
                    edgecolor=col, lw=.5)
            ax.plot([lo, hi], [y, y], color="0.15", lw=.9, solid_capstyle="butt", zorder=3)
    labels.append(f"{task} $\\cdot$ {m}")
    rows.append((task, m, ga.mean() - wa.mean(), gc.mean() - wc.mean(), float(np.mean(du)),
                 *np.percentile(du, [2.5, 97.5])))

for j, (ax, xl) in enumerate(zip(axes, [r"$\Delta$ accuracy",
                                        "$\\Delta$ confidence (0\u201310 scale)",
                                        r"$\Delta$ AUROC"])):
    ax.set_yticks(range(0, -5, -1)); ax.set_ylim(-4.6, .6)
    ax.set_xlabel(xl); ax.axvline(0, color="0.55", lw=.6, zorder=0)
    S.panel(ax, "abc"[j], dx=-0.62 if j == 0 else -0.10)
axes[0].set_yticklabels(labels, fontsize=6.5)
for ax in axes[1:]: ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
axes[2].set_xlim(-.20, .20); axes[2].set_xticks([-.2, -.1, 0, .1, .2])
axes[0].set_xlim(0, .46); axes[1].set_xlim(0, 4.0)

fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=S.RED, ec="none"),
                    plt.Rectangle((0, 0), 1, 1, fc=S.BLUE, ec="none")],
           labels=["removal makes the case unanswerable",
                   "case remains answerable without the removed item"],
           loc="lower center", bbox_to_anchor=(.55, -.13), ncol=2, handlelength=1.1,
           handleheight=.8, columnspacing=1.4)
fig.subplots_adjust(wspace=.13)
S.save(fig, "fig0_core")
print(f"{'cell':30s} {'dAcc':>7s} {'dConf':>7s} {'dAUROC':>8s}  95% CI")
for t, m, a, c, u, lo, hi in rows:
    print(f"{t+' / '+m:30s} {a:+7.3f} {c:+7.2f} {u:+8.3f}  [{lo:+.3f}, {hi:+.3f}]")
