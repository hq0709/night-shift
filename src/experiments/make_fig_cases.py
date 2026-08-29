"""Qualitative figure: three real episode pairs from the removal manipulation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as S
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import textwrap

# Verbatim from results/evidence_manip.jsonl, gpt-5.4-mini, seed 0.
CASES = [
    dict(tag="Signal reports the gap", uid="medqa::2", gold="Hirschsprung disease",
         held="barium enema findings",
         w=dict(dx="Constipation with fecal impaction", conf=7, ok=False,
                diff=["Constipation with fecal loading", "Hirschsprung disease", "Gas distension"]),
         g=dict(dx="Hirschsprung disease", conf=10, ok=True,
                diff=["Hirschsprung disease", "Mechanical obstruction", "Constipation / impaction"])),
    dict(tag="Signal is blind to the error", uid="medqa::119",
         gold="Thrombotic thrombocytopenic purpura", held="platelet count",
         w=dict(dx="Plasmodium falciparum malaria", conf=9, ok=False,
                diff=["P. falciparum malaria", "Leptospirosis", "Dengue fever"]),
         g=dict(dx="Dengue hemorrhagic fever", conf=9, ok=False,
                diff=["Dengue hemorrhagic fever", "Falciparum malaria", "TTP"])),
    dict(tag="Confidence moves, correctness does not", uid="medqa::10", gold="Hemorrhoids",
         held="anoscopy findings",
         w=dict(dx="Internal hemorrhoids", conf=8, ok=True,
                diff=["Internal hemorrhoids", "Rectal carcinoma", "Rectal prolapse"]),
         g=dict(dx="Prolapsed internal hemorrhoids", conf=10, ok=True,
                diff=["Internal hemorrhoids with prolapse", "Rectal prolapse", "Anal mass"])),
]

fig = plt.figure(figsize=(S.TXT, 3.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ROW_H, TOP = 0.295, 0.905
XC = {"w": 0.315, "g": 0.655}          # left edge of each condition column
CW = 0.325

ax.text(XC["w"], 0.965, "key finding WITHHELD", fontsize=7.2, fontweight="bold", color=S.RED)
ax.text(XC["g"], 0.965, "key finding GIVEN", fontsize=7.2, fontweight="bold", color=S.BLUE)

for i, c in enumerate(CASES):
    y = TOP - i * ROW_H
    ax.text(0.005, y, f"({chr(97+i)})", fontsize=8, fontweight="bold", va="top")
    ax.text(0.042, y, c["tag"], fontsize=7.2, fontweight="bold", va="top")
    ax.text(0.042, y - .052, f"reference: {c['gold']}", fontsize=6.4, va="top", color="0.25")
    ax.text(0.042, y - .096, f"removed: {c['held']}", fontsize=6.4, va="top", color="0.45",
            style="italic")
    for key, col in (("w", S.RED), ("g", S.BLUE)):
        d = c[key]; x = XC[key]
        ax.add_patch(FancyBboxPatch((x - .012, y - .215), CW, .225,
                                    boxstyle="round,pad=0.004,rounding_size=0.008",
                                    fc=col, alpha=.055, ec=col, lw=.5,
                                    transform=ax.transAxes, zorder=0))
        mark = "correct" if d["ok"] else "incorrect"
        ax.text(x, y, d["dx"], fontsize=6.9, va="top", fontweight="bold")
        # matplotlib's wrap= does not respect axes-fraction boxes, so wrap explicitly
        body = textwrap.fill("differential: " + "; ".join(d["diff"]), 52)
        ax.text(x, y - .042, body, fontsize=5.9, va="top", color="0.3", linespacing=1.35)
        # confidence rendered as a bar so the two columns are comparable at a glance
        bx, bw = x, 0.20
        ax.add_patch(plt.Rectangle((bx, y - .175), bw, .028, fc="0.88", ec="none",
                                   transform=ax.transAxes))
        ax.add_patch(plt.Rectangle((bx, y - .175), bw * d["conf"] / 10, .028, fc=col, ec="none",
                                   transform=ax.transAxes))
        ax.text(bx + bw + .012, y - .162, f"confidence {d['conf']}/10", fontsize=6.2, va="center")
        ax.text(bx, y - .205, mark, fontsize=6.2, va="center",
                color=S.GREEN if d["ok"] else S.RED, fontweight="bold")
    if i < len(CASES) - 1:
        ax.plot([0.03, 0.985], [y - .245] * 2, color="0.86", lw=.5, transform=ax.transAxes)

S.save(fig, "fig_cases")
print("fig_cases done")
