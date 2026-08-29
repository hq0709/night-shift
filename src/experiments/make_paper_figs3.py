"""Fig 7: the high-acuity contrast at matched difficulty. Fig 8: qualitative episode walk-through."""
import sys, json, pathlib, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
_R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R))
plt.rcParams.update({"font.size": 8, "axes.linewidth": .7, "legend.frameon": False})
rng = np.random.default_rng(0)
def auroc(c, y):
    c=np.asarray(c,float); y=np.asarray(y,int)
    if y.sum() in (0,len(y)): return np.nan
    r=rankdata(c); n1=y.sum(); return (r[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
def L(p):
    p=_R/p
    return [json.loads(l) for l in open(p) if json.loads(l).get("confidence") is not None]

# ---- Fig 7: matched-difficulty comparison ----
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ax, (m, rf, od) in zip(axes, [
        ("gpt-5.4-mini","results/redflag_gpt-5.4-mini.jsonl","results/easy_gpt-5.4-mini.jsonl"),
        ("gpt-5.4","results/redflag_gpt-5.4.jsonl","results/easy_gpt-5.4.jsonl")]):
    R,O=L(rf),L(od)
    acc=np.mean([x["correct"] for x in R])
    a_rf=auroc([x["confidence"] for x in R],[x["correct"] for x in R])
    cor=[x for x in O if x["correct"]]; wr=[x for x in O if not x["correct"]]
    n=int(len(wr)*acc/(1-acc)); vals=[]
    for _ in range(3000):
        s=list(rng.choice(cor,size=min(n,len(cor)),replace=False))+wr
        v=auroc([x["confidence"] for x in s],[x["correct"] for x in s])
        if not np.isnan(v): vals.append(v)
    ax.hist(vals,bins=40,color="#2980b9",alpha=.75,density=True)
    ax.axvline(a_rf,color="#c0392b",lw=2.0)
    ax.axvline(np.percentile(vals,2.5),color="#555",ls="--",lw=.9)
    ax.axvline(np.percentile(vals,97.5),color="#555",ls="--",lw=.9)
    ax.set_title(f"{m}\nRedFlag {a_rf:.3f} vs matched controls",fontsize=7.6)
    ax.set_xlabel("AUROC on accuracy-matched standard items",fontsize=7.3)
    ax.spines[["top","right"]].set_visible(False)
    ax.annotate("RedFlag-99",xy=(a_rf,ax.get_ylim()[1]*.72),xytext=(a_rf-.055,ax.get_ylim()[1]*.88),
                fontsize=6.8,color="#c0392b",
                arrowprops=dict(arrowstyle="->",color="#c0392b",lw=.9))
axes[0].set_ylabel("density over matched resamples",fontsize=7.3)
fig.tight_layout(); fig.savefig(_R/"paper/figs/fig7_matched.pdf",bbox_inches="tight")
print("fig7")

# ---- Fig 8: accuracy vs discrimination across all nine sets ----
SETS=[("MedAgentsBench-hard","results/stage2_gpt-5.4-mini.jsonl","#c0392b","o"),
      ("RedFlag-99","results/redflag_gpt-5.4-mini.jsonl","#e67e22","s"),
      ("MedAgentsBench-std","results/easy_gpt-5.4-mini.jsonl","#2980b9","o"),
      ("AgentClinic static","results/ac_gpt-5.4-mini.jsonl","#2980b9","^"),
      ("AgentClinic interactive","results/inter_full_gpt-5.4-mini.jsonl","#27ae60","^"),
      ("MMLU-Pro","results/gen_mmlupro.jsonl","#7f8c8d","D"),
      ("ARC-Challenge","results/gen_arc.jsonl","#7f8c8d","D"),
      ("MedCalc","results/gen_medcalc.jsonl","#7f8c8d","v"),
      ("GSM8K","results/gen_gsm8k.jsonl","#8e44ad","v")]
fig, ax = plt.subplots(figsize=(3.5,3.1))
xs,ys=[],[]
for lbl,p,col,mk in SETS:
    r=L(p)
    if not r: continue
    a=np.mean([x["correct"] for x in r]); u=auroc([x["confidence"] for x in r],[x["correct"] for x in r])
    xs.append(a); ys.append(u)
    ax.scatter(a,u,color=col,marker=mk,s=42,zorder=3,edgecolor="white",linewidth=.6)
    OFF={"MedAgentsBench-hard":(0,9),"RedFlag-99":(-4,-11),"MedAgentsBench-std":(24,7),
         "AgentClinic static":(30,-4),"AgentClinic interactive":(6,9),"MMLU-Pro":(-20,7),
         "ARC-Challenge":(-6,-12),"MedCalc":(-26,-3),"GSM8K":(0,9)}
    dx,dy=OFF.get(lbl,(0,8))
    ax.annotate(lbl,(a,u),textcoords="offset points",xytext=(dx,dy),fontsize=5.8,ha="center")
ax.axhline(.5,ls="--",lw=.8,color="#999")
z=np.polyfit(xs,ys,1); xx=np.linspace(min(xs),max(xs),50)
ax.plot(xx,np.polyval(z,xx),ls=":",lw=1.0,color="#555")
ax.set_xlabel("task accuracy",fontsize=7.5); ax.set_ylabel("AUROC of elicited confidence",fontsize=7.5)
ax.set_ylim(.43,.88); ax.set_xlim(.30,1.03); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(_R/"paper/figs/fig8_acc_vs_auc.pdf",bbox_inches="tight")
print("fig8")
