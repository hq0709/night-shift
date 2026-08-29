"""Fig 8: task competence against discrimination, across every evaluation set."""
import sys, json, pathlib, collections
import numpy as np
import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import figstyle as S
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
_R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R))
rng = np.random.default_rng(0)
def auroc(c, y):
    c=np.asarray(c,float); y=np.asarray(y,int)
    if y.sum() in (0,len(y)): return np.nan
    r=rankdata(c); n1=y.sum(); return (r[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
def L(p):
    p=_R/p
    return [json.loads(l) for l in open(p) if json.loads(l).get("confidence") is not None]

# ---- Fig 8: accuracy vs discrimination across every evaluation set ----
SETS=[("MedAgentsBench-hard","results/stage2_gpt-5.4-mini.jsonl",S.RED,"o"),
      ("MedAgentsBench-std","results/easy_gpt-5.4-mini.jsonl",S.BLUE,"o"),
      ("AgentClinic static","results/ac_gpt-5.4-mini.jsonl",S.BLUE,"^"),
      ("AgentClinic interactive","results/inter_full_gpt-5.4-mini.jsonl",S.GREEN,"^"),
      ("MMLU-Pro","results/gen_mmlupro.jsonl",S.GREY,"D"),
      ("ARC-Challenge","results/gen_arc.jsonl",S.GREY,"D"),
      ("MedCalc","results/gen_medcalc.jsonl",S.GREY,"v"),
      ]
fig, ax = plt.subplots(figsize=(S.COL,2.9))
xs,ys=[],[]
for lbl,p,col,mk in SETS:
    r=L(p)
    if not r: continue
    a=np.mean([x["correct"] for x in r]); u=auroc([x["confidence"] for x in r],[x["correct"] for x in r])
    xs.append(a); ys.append(u)
    ax.scatter(a,u,color=col,marker=mk,s=42,zorder=3,edgecolor="white",linewidth=.6)
    OFF={"MedAgentsBench-hard":(0,9),"MedAgentsBench-std":(24,7),
         "AgentClinic static":(30,-4),"AgentClinic interactive":(6,9),"MMLU-Pro":(-20,7),
         "ARC-Challenge":(-6,-12),"MedCalc":(-26,-3)}
    dx,dy=OFF.get(lbl,(0,8))
    ax.annotate(lbl,(a,u),textcoords="offset points",xytext=(dx,dy),ha="center")
ax.axhline(.5,ls="--",lw=.8,color="#999")
z=np.polyfit(xs,ys,1); xx=np.linspace(min(xs),max(xs),50)
ax.plot(xx,np.polyval(z,xx),ls=":",lw=1.0,color="#555")
ax.set_xlabel("task accuracy"); ax.set_ylabel("AUROC of elicited confidence")
ax.set_ylim(.43,.88); ax.set_xlim(.30,1.03); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); S.save(fig,"fig8_acc_vs_auc")
print("fig8")
