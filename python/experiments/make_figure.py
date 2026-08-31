"""Render the headline figure from the saved experiment results."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "docs", "results"))

# Categorical slots 1-3 from the validated palette
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

with open(os.path.join(RESULTS, "exp02_duration_labels.json")) as fh:
    d = json.load(fh)

dur = d["duration"]
lab = d["labels"]
x = [r["duration_s"] for r in dur]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

# ---- Panel A: error does not shrink with observation time ----
series = [
    ("Static channel\n(WASN assumption)", [r["static_rmse"] for r in dur], BLUE),
    ("Signal-dependent channel\n(multimodal reality)", [r["switching_rmse"] for r in dur], ORANGE),
    ("+ class conditioning\n(proposed)", [r["conditioned_rmse"] for r in dur], AQUA),
]
for name, y, c in series:
    ax1.plot(x, y, color=c, linewidth=2, marker="o", markersize=6,
             markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{int(v)}s" for v in x])
ax1.set_xlabel("Observation length")
ax1.set_ylabel("SRO estimation RMSE (ppm)")
ax1.set_title("More data does not rescue the baseline", color=INK,
              fontsize=11, fontweight="bold", loc="left", pad=12)
ax1.set_ylim(0.7, 400)

# Direct labels (required: aqua is sub-3:1 on this surface)
ax1.annotate("Signal-dependent channel\n(multimodal reality)", (x[-1], dur[-1]["switching_rmse"]),
             xytext=(-6, 22), textcoords="offset points", color=ORANGE,
             fontsize=8.5, ha="right", fontweight="bold")
ax1.annotate("+ class conditioning\n(proposed)", (x[2], dur[2]["conditioned_rmse"]),
             xytext=(-4, -40), textcoords="offset points", color=AQUA,
             fontsize=8.5, ha="center", fontweight="bold")
ax1.annotate("Static channel\n(WASN assumption)", (x[0], dur[0]["static_rmse"]),
             xytext=(8, -30), textcoords="offset points", color=BLUE,
             fontsize=8.5, ha="left", fontweight="bold")

ax1.axhline(100, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax1.annotate("true SRO = 100 ppm", (x[0], 100), xytext=(2, 6),
             textcoords="offset points", color=MUTED, fontsize=8)

# ---- Panel B: how good must the classifier be ----
lx = [r["label_error"] * 100 for r in lab]
ly = [r["rmse_ppm"] for r in lab]
baseline = [r for r in dur if r["duration_s"] == 60.0][0]["switching_rmse"]

ax2.axhline(baseline, color=ORANGE, linewidth=2, linestyle=(0, (4, 3)), zorder=2)
ax2.annotate("unconditioned baseline", (100, baseline), xytext=(-4, 8),
             textcoords="offset points", color=ORANGE, fontsize=8.5,
             ha="right", fontweight="bold")
ax2.plot(lx, ly, color=AQUA, linewidth=2, marker="o", markersize=6,
         markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
ax2.annotate("class-conditioned", (lx[2], ly[2]), xytext=(14, -16),
             textcoords="offset points", color=AQUA, fontsize=8.5,
             fontweight="bold")
ax2.annotate("crossover ≈ 45% label error", (45, 84), xytext=(-8, -46),
             textcoords="offset points", color=MUTED, fontsize=8, ha="right",
             arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))

ax2.set_xlabel("Class-label error rate (%)")
ax2.set_ylabel("SRO estimation RMSE (ppm)")
ax2.set_title("A mediocre classifier is enough", color=INK,
              fontsize=11, fontweight="bold", loc="left", pad=12)
ax2.set_ylim(0, 145)

fig.text(0.005, -0.03,
         "Synthetic model, 10 seeds per point, true SRO 100 ppm. "
         "Class labels are oracle in panel A. Lower is better.",
         fontsize=8, color=MUTED)

fig.tight_layout()
out = os.path.join(RESULTS, "fig01_confound.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print("wrote", out)
