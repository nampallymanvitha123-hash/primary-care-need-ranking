"""Exports the seed-42 dataset and ranking, and draws the README figures.
Run from the repository root:  python src/make_figures.py
"""
import contextlib, io, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
with contextlib.redirect_stdout(io.StringIO()):   # silence the 100-seed run
    import ranking_system as rs

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b4b2ab"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})
os.makedirs("figures", exist_ok=True); os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- data exports ---------------------------------------------------------
rs.df.to_csv("data/synthetic_areas_seed42.csv", index=False)
r = rs.ranked_df
r[["priority_rank", "area_id", "region", "rurality", "need_score",
   "provision_gap_score", "priority_score", "is_embedded_gap"]].head(30) \
    .to_csv("results/top30_priority_areas.csv", index=False)
summary = rs.results.groupby(["setting", "method"])["f1"].agg(["mean", "std"]).round(2)
summary.to_csv("results/f1_100_seeds.csv")

# --- Figure 1: need vs provision gap --------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.6))
other, emb = r[~r.is_embedded_gap], r[r.is_embedded_gap]
ax.scatter(other.need_score, other.provision_gap_score, s=18, c=GREY,
           edgecolors=SURFACE, linewidths=0.6, label="Other areas (470)")
ax.scatter(emb.need_score, emb.provision_gap_score, s=46, c=ORANGE,
           edgecolors=SURFACE, linewidths=1, label="Hidden high-need, low-provision areas (30)")
ax.set_xlabel("Need Score (0–100)"); ax.set_ylabel("Provision Gap Score (0–100)")
ax.set_title("Hidden gap areas cluster where need and provision gap are both high",
             loc="left", color=INK, fontsize=12.5, pad=12)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=10)
fig.tight_layout(); fig.savefig("figures/need_vs_provision.png", dpi=160); plt.close(fig)

# --- Figure 2: top 30 ------------------------------------------------------
top = r.head(30).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 8))
colors = [ORANGE if e else GREY for e in top.is_embedded_gap]
ax.barh(top.area_id, top.priority_score, color=colors, height=0.72)
ax.set_xlabel("Priority Score"); ax.grid(axis="y", visible=False)
ax.set_title("Top 30 areas by Priority Score: 23 of 30 are true gap areas",
             loc="left", color=INK, fontsize=12.5, pad=12)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE, label="Hidden gap area (true positive)"),
                   Patch(color=GREY, label="Other area (false positive)")],
          frameon=False, loc="lower right", fontsize=10)
ax.tick_params(axis="y", labelsize=8.5)
fig.tight_layout(); fig.savefig("figures/top30_priority_areas.png", dpi=160); plt.close(fig)

# --- Figure 3: F1 before / after normalisation fix ------------------------
names = {"multiplicative": "Multiplicative\n(recommended)", "equal_sum": "Equal\nweighted sum",
         "need_emphasis": "Need-emphasis\nsum", "provision_emphasis": "Provision-emphasis\nsum"}
mm = summary.loc["Min-max"]; fx = summary.loc["Log + within rurality"]
x = range(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(8.4, 5))
b1 = ax.bar([i - w/2 for i in x], [mm.loc[m, "mean"] for m in names], w,
            yerr=[mm.loc[m, "std"] for m in names], color=GREY, label="Plain min-max",
            error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
b2 = ax.bar([i + w/2 for i in x], [fx.loc[m, "mean"] for m in names], w,
            yerr=[fx.loc[m, "std"] for m in names], color=BLUE,
            label="Log-scaled + normalised within urban/rural",
            error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, 3, f"{b.get_height():.1f}", ha="center",
            va="bottom", color=SURFACE, fontsize=9.5, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(names.values(), fontsize=9.5)
ax.set_ylabel("Mean F1 score, % (100 random seeds)"); ax.set_ylim(0, 100)
ax.grid(axis="x", visible=False)
ax.set_title("The normalisation fix lifts every method to 86–88% F1",
             loc="left", color=INK, fontsize=12.5, pad=12)
ax.legend(frameon=False, loc="upper center", fontsize=10, ncol=2,
          bbox_to_anchor=(0.5, -0.17))
fig.tight_layout(); fig.savefig("figures/f1_by_method.png", dpi=160); plt.close(fig)
print("Saved data/, results/ and figures/")
