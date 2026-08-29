"""Figure 1, the paper's central result: what the removal manipulation moves and what it does not."""
import sys, json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
_R = pathlib.Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 8, "axes.linewidth": .7, "legend.frameon": False})
def auroc(c, y):
    c=np.asarray(c,float); y=np.asarray(y,int)
    if y.sum() in (0,len(y)): return np.nan
    r=rankdata(c); n1=y.sum(); return (r[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
def boot(c,y,n=4000):
    c=np.asarray(c,float); y=np.asarray(y,int); v=[]
    for _ in range(n):
        i=rng.integers(0,len(c),len(c)); a=auroc(c[i],y[i])
        if not np.isnan(a): v.append(a)
    return np.mean(v),np.percentile(v,2.5),np.percentile(v,97.5)
def L(p): return [json.loads(l) for l in open(_R/p) if json.loads(l).get("confidence") is not None]

CELLS=[("MedCalc\n(unanswerable)","gpt-5.4-mini","results/evmanip_calc.jsonl","#c0392b"),
       ("MedCalc\n(unanswerable)","gpt-5.4","results/evmanip_calc_gpt54.jsonl","#e74c3c"),
       ("AgentClinic\n(answerable)","gpt-5.4-mini","results/evidence_manip.jsonl","#2980b9"),
       ("AgentClinic\n(answerable)","gpt-5.4","results/evidence_manip_gpt54.jsonl","#3498db"),
       ("AgentClinic\n(answerable)","gpt-5.4-nano","results/evidence_manip_nano.jsonl","#5dade2")]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
labels=[]
for k,(task,m,p,col) in enumerate(CELLS):
    r=L(p); W=[x for x in r if x["condition"]=="withheld"]; G=[x for x in r if x["condition"]=="given"]
    dacc=np.mean([x["correct"] for x in G])-np.mean([x["correct"] for x in W])
    dcon=np.mean([x["confidence"] for x in G])-np.mean([x["confidence"] for x in W])
    d=[]
    for _ in range(4000):
        ia=rng.integers(0,len(W),len(W)); ib=rng.integers(0,len(G),len(G))
        a=auroc([W[i]["confidence"] for i in ia],[W[i]["correct"] for i in ia])
        b=auroc([G[i]["confidence"] for i in ib],[G[i]["correct"] for i in ib])
        if not (np.isnan(a) or np.isnan(b)): d.append(b-a)
    y=-k
    axes[0].barh(y, dacc, color=col, height=.62)
    axes[1].barh(y, dcon, color=col, height=.62)
    axes[2].plot([np.percentile(d,2.5),np.percentile(d,97.5)],[y,y],color=col,lw=1.8)
    axes[2].plot(np.mean(d), y, "o", color=col, ms=4.5)
    labels.append(f"{task.splitlines()[0]} · {m}")
for ax,t,xl in zip(axes,
        ["accuracy restored","stated confidence restored","discrimination restored"],
        ["$\\Delta$ accuracy","$\\Delta$ confidence (0--10)","$\\Delta$ AUROC"]):
    ax.set_yticks(range(0,-5,-1)); ax.set_title(t, fontsize=8)
    ax.set_xlabel(xl, fontsize=7.5); ax.axvline(0,color="#666",lw=.8)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_yticklabels(labels, fontsize=6.6)
for ax in axes[1:]: ax.set_yticklabels([])
axes[2].set_xlim(-0.20,0.20)
fig.tight_layout()
fig.savefig(_R/"paper/figs/fig0_core.pdf", bbox_inches="tight")
fig.savefig(_R/"paper/figs/fig0_core.png", dpi=200, bbox_inches="tight")
print("fig0 core done")
