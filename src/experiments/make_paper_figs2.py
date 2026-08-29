"""Figures 3-6: the diagnostics a journal reviewer expects and the manuscript lacked."""
from __future__ import annotations
import sys, json, pathlib, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, pearsonr
_R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R))
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "legend.frameon": False})
rng = np.random.default_rng(0)

def L(p):
    p = _R / p
    return [json.loads(l) for l in open(p) if json.loads(l).get("confidence") is not None] if p.exists() else []

# ---- Fig 3: confidence distributions by correctness, per condition ----------
CONDS = [("Capability frontier", "results/stage2_gpt-5.4-mini.jsonl"),
         ("High acuity", "results/redflag_gpt-5.4-mini.jsonl"),
         ("Standard difficulty", "results/easy_gpt-5.4-mini.jsonl"),
         ("Interactive", "results/inter_full_gpt-5.4-mini.jsonl")]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9), sharey=True)
for ax, (lbl, p) in zip(axes, CONDS):
    r = L(p)
    if not r: continue
    bins = np.arange(0.5, 11.5, 1)
    cor = [x["confidence"] for x in r if x["correct"]]
    wrg = [x["confidence"] for x in r if not x["correct"]]
    ax.hist(cor, bins=bins, density=True, alpha=.62, color="#2980b9", label="correct")
    ax.hist(wrg, bins=bins, density=True, alpha=.62, color="#c0392b", label="incorrect")
    ax.set_title(lbl, fontsize=7.4); ax.set_xlabel("stated confidence", fontsize=7)
    ax.set_xlim(3.5, 10.5); ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("density", fontsize=7.4); axes[0].legend(fontsize=6.4)
fig.tight_layout(); fig.savefig(_R/"paper/figs/fig3_distributions.pdf", bbox_inches="tight")
print("fig3")

# ---- Fig 4: reliability diagrams -------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=True)
for ax, (lbl, p) in zip(axes, CONDS):
    r = L(p)
    if not r: continue
    xs, ys, ns = [], [], []
    for b in range(1, 11):
        s = [x for x in r if x["confidence"] == b]
        if len(s) < 12: continue
        xs.append(b/10); ys.append(np.mean([x["correct"] for x in s])); ns.append(len(s))
    ax.plot([0,1],[0,1], ls="--", lw=.8, color="#999")
    sz = 6 + 44*np.array(ns)/max(ns) if ns else []
    ax.scatter(xs, ys, s=sz, color="#8e44ad", zorder=3, alpha=.85)
    ax.plot(xs, ys, color="#8e44ad", lw=1.1)
    ece = sum(n/sum(ns)*abs(y-x) for x,y,n in zip(xs,ys,ns)) if ns else float("nan")
    ax.set_title(f"{lbl}\nECE={ece:.3f}", fontsize=7.2)
    ax.set_xlabel("stated confidence", fontsize=7); ax.set_xlim(0.35,1.03); ax.set_ylim(0,1.03)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("observed accuracy", fontsize=7.4)
fig.tight_layout(); fig.savefig(_R/"paper/figs/fig4_reliability.pdf", bbox_inches="tight")
print("fig4")

# ---- Fig 5: risk-coverage ---------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
COL = {"Capability frontier":"#c0392b","High acuity":"#e67e22",
       "Standard difficulty":"#2980b9","Interactive":"#27ae60"}
for lbl, p in CONDS:
    r = L(p)
    if not r: continue
    o = sorted(r, key=lambda x: -x["confidence"])
    cov, risk = [], []
    for k in range(30, len(o)+1, max(1, len(o)//60)):
        cov.append(k/len(o)); risk.append(1-np.mean([x["correct"] for x in o[:k]]))
    ax.plot(np.array(cov)*100, risk, lw=1.4, color=COL[lbl], label=lbl)
ax.set_xlabel("coverage (%) — cases handled autonomously", fontsize=7.4)
ax.set_ylabel("risk (error rate on covered cases)", fontsize=7.4)
ax.legend(fontsize=6.6); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(_R/"paper/figs/fig5_risk_coverage.pdf", bbox_inches="tight")
print("fig5")

# ---- Fig 6: what confidence tracks -----------------------------------------
def agg(p):
    d = collections.defaultdict(list)
    for l in open(_R/p):
        x = json.loads(l)
        if x.get("confidence") is not None: d[x["uid"]].append(x)
    return d
A = {k: agg(v) for k, v in {
    "gpt-5.4-mini":"results/axisA_gpt54mini.jsonl","sonnet-5":"results/axisA_sonnet5.jsonl",
    "gpt-5.4-nano":"results/stage1_gpt-5.4-nano.jsonl","gpt-5.4":"results/stage1_gpt-5.4.jsonl"}.items()}
com = sorted(set.intersection(*[set(v) for v in A.values()]))
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
cx = np.array([np.mean([e["confidence"] for e in A["gpt-5.4-mini"][u]]) for u in com])
cy = np.array([np.mean([e["confidence"] for e in A["gpt-5.4"][u]]) for u in com])
ax = axes[0]; ax.scatter(cx, cy, s=7, alpha=.45, color="#34495e")
ax.set_xlabel("gpt-5.4-mini confidence", fontsize=7); ax.set_ylabel("gpt-5.4 confidence", fontsize=7)
ax.set_title(f"cross-model confidence\nr = {pearsonr(cx,cy)[0]:.3f}", fontsize=7.4)
ax_ = axes[1]
ax_.scatter(cx, [np.mean([e["correct"] for e in A["gpt-5.4-mini"][u]]) for u in com],
            s=7, alpha=.45, color="#c0392b")
ax_.set_xlabel("gpt-5.4-mini confidence", fontsize=7); ax_.set_ylabel("its own accuracy", fontsize=7)
ax_.set_title(f"confidence vs own correctness\nr = "
              f"{pearsonr(cx,[np.mean([e['correct'] for e in A['gpt-5.4-mini'][u]]) for u in com])[0]:+.3f}",
              fontsize=7.4)
def stab(k):
    o=[]
    for u in com:
        ls=[e["letter"] for e in A[k][u] if e["letter"]]
        o.append(collections.Counter(ls).most_common(1)[0][1]/len(ls) if ls else np.nan)
    return np.array(o)
s_other = stab("gpt-5.4")
m = ~np.isnan(s_other)
ax2 = axes[2]; ax2.scatter(s_other[m], cx[m], s=7, alpha=.45, color="#27ae60")
ax2.set_xlabel("another model's answer stability", fontsize=7)
ax2.set_ylabel("gpt-5.4-mini confidence", fontsize=7)
ax2.set_title(f"confidence vs cross-model stability\nr = {pearsonr(s_other[m],cx[m])[0]:+.3f}", fontsize=7.4)
for a in axes: a.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(_R/"paper/figs/fig6_mechanism.pdf", bbox_inches="tight")
print("fig6")
