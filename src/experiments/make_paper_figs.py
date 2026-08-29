"""Figures for the paper. Two figures carry the argument:
Fig 1  discrimination across conditions (the central claim: it is task-dependent)
Fig 2  decision curves, high-acuity static vs interactive (the deployment consequence)
"""
from __future__ import annotations
import sys, json, os, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
_R = pathlib.Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.7, "xtick.major.width": 0.7,
                     "ytick.major.width": 0.7, "legend.frameon": False})


def auroc(c, y):
    c = np.asarray(c, float); y = np.asarray(y, int)
    if y.sum() in (0, len(y)):
        return np.nan
    r = rankdata(c); n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def boot(c, y, n=4000):
    c = np.asarray(c, float); y = np.asarray(y, int); v = []
    for _ in range(n):
        i = rng.integers(0, len(c), len(c)); a = auroc(c[i], y[i])
        if not np.isnan(a):
            v.append(a)
    return (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan,) * 3


def load(p):
    p = _R / p
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if json.loads(l).get("confidence") is not None]


# ---------------------------------------------------------------- Figure 1
GROUPS = [
    ("Capability frontier", "#c0392b", [
        ("MedAgentsBench-hard", "gpt-5.4-mini", "results/stage2_gpt-5.4-mini.jsonl"),
        ("MedAgentsBench-hard", "gpt-5.4", "results/stage2_gpt-5.4.jsonl"),
        ("MedAgentsBench-hard", "sonnet-5", "results/axisA_sonnet5.jsonl"),
        ("MedAgentsBench-hard", "gpt-4o-mini", "results/nr_hard_gpt-4o-mini.jsonl")]),
    ("High acuity", "#e67e22", [
        ("RedFlag-99", "gpt-5.4-mini", "results/redflag_gpt-5.4-mini.jsonl"),
        ("RedFlag-99", "gpt-5.4", "results/redflag_gpt-5.4.jsonl")]),
    ("Standard difficulty", "#2980b9", [
        ("MedAgentsBench-std", "gpt-5.4-mini", "results/easy_gpt-5.4-mini.jsonl"),
        ("MedAgentsBench-std", "gpt-5.4", "results/easy_gpt-5.4.jsonl"),
        ("MedAgentsBench-std", "gpt-4o-mini", "results/nr_easy_gpt-4o-mini.jsonl"),
        ("AgentClinic static", "gpt-5.4-mini", "results/ac_gpt-5.4-mini.jsonl"),
        ("AgentClinic static", "gpt-5.4", "results/ac_gpt-5.4.jsonl")]),
    ("Interactive consultation", "#27ae60", [
        ("AgentClinic interactive", "gpt-5.4-mini", "results/inter_full_gpt-5.4-mini.jsonl"),
        ("AgentClinic interactive", "gpt-5.4", "results/inter_full_gpt-5.4.jsonl")]),
]

fig, ax = plt.subplots(figsize=(7.0, 4.4))
y = 0; ticks = []; labels = []
for gname, color, entries in GROUPS:
    for ds, m, p in entries:
        r = load(p)
        if not r:
            continue
        mu, lo, hi = boot([x["confidence"] for x in r], [x["correct"] for x in r])
        ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="butt")
        ax.plot(mu, y, "o", color=color, ms=4.5, zorder=3)
        ax.text(0.905, y, f"{np.mean([x['correct'] for x in r]):.2f}", va="center",
                ha="center", fontsize=6.6, color="#555")
        ticks.append(y); labels.append(f"{ds}  ·  {m}")
        y -= 1
    y -= 0.55

ax.axvline(0.5, color="#888", ls="--", lw=0.8, zorder=0)
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("AUROC of elicited confidence for discriminating correct vs incorrect")
ax.set_xlim(0.38, 0.93); ax.set_ylim(y + 0.4, 1.2)
ax.text(0.905, 1.0, "acc.", ha="center", fontsize=6.6, color="#555")
ax.text(0.5, 1.0, "chance", ha="center", fontsize=6.6, color="#888")
for gname, color, _ in GROUPS:
    ax.plot([], [], color=color, lw=2.4, label=gname)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4,
          fontsize=7, columnspacing=1.4, handlelength=1.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(_R / "paper/figs/fig1_discrimination.pdf", bbox_inches="tight")
fig.savefig(_R / "paper/figs/fig1_discrimination.png", dpi=220, bbox_inches="tight")
print("fig1 done")

# ---------------------------------------------------------------- Figure 2
from envs import general                     # noqa: E402
from taxonomy.families import considered     # noqa: E402
meta = {it["uid"]: it for it in general.load_redflag()}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax_, (title, kind, sets) in zip(axes, [
        ("High-acuity, static\n(endpoint: red flag considered)", "rf",
         [("gpt-5.4-mini", "results/redflag_gpt-5.4-mini.jsonl", "#e67e22"),
          ("gpt-5.4", "results/redflag_gpt-5.4.jsonl", "#c0392b")]),
        ("Interactive consultation\n(endpoint: diagnosis correct)", "acc",
         [("gpt-5.4-mini", "results/inter_full_gpt-5.4-mini.jsonl", "#27ae60"),
          ("gpt-5.4", "results/inter_full_gpt-5.4.jsonl", "#16a085")])]):
    for m, p, col in sets:
        rows = load(p)
        pts = []
        for r in rows:
            if kind == "rf":
                it = meta.get(r["uid"])
                if not it:
                    continue
                names = [it["redflag"]] + list(it.get("rf_aliases", []))
                cand = list(r.get("differential") or []) + [r.get("red_flag_considered", "")]
                good = int(any(considered(nm, cand) for nm in names if nm))
            else:
                good = int(r["correct"])
            pts.append((r["confidence"], good))
        base = np.mean([g for _, g in pts])
        xs, ys = [], []
        for th in np.arange(0, 11.5, 0.5):
            keep = [g for c, g in pts if c >= th]
            if len(keep) < 20:
                continue
            xs.append(1 - len(keep) / len(pts)); ys.append(np.mean(keep))
        ax_.plot(np.array(xs) * 100, ys, "-o", color=col, ms=2.6, lw=1.3, label=m)
        ax_.axhline(base, color=col, ls=":", lw=0.9)
    ax_.set_title(title, fontsize=7.6)
    ax_.set_xlabel("cases deferred to clinician (%)", fontsize=7.5)
    ax_.legend(fontsize=6.8, loc="best")
    ax_.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("red flag considered\n(retained cases)", fontsize=7.5)
axes[1].set_ylabel("diagnostic accuracy\n(retained cases)", fontsize=7.5)
fig.tight_layout()
fig.savefig(_R / "paper/figs/fig2_decision_curves.pdf", bbox_inches="tight")
fig.savefig(_R / "paper/figs/fig2_decision_curves.png", dpi=220, bbox_inches="tight")
print("fig2 done")
