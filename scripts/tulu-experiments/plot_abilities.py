"""Paper-ready 'Abilities Overview' panel per model (cwbench v3).

One figure per model with four sub-panels (radars + strengths + weaknesses).
Light theme, custom pastel palette per model, 14pt fonts, no big title.
Saved as both .pdf and .png in cwbench-rl/eval/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

EVAL_DIR = Path("/Users/bhavesh/Developer/tourno/cwbench-rl/eval")
FIG_DIR = EVAL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS: dict[str, str] = {
    "base": "Base",
    "pointwise": "Pointwise",
    "pairwise": "Pairwise",
    "tourno": "TournO",
}

EVAL_FILENAMES: dict[str, str] = {
    "base": "base.json",
    "pointwise": "pointwise.json",
    "pairwise": "pairwise.json",
    "tourno": "tourno-step120.json",
}

# Palette matches plot_leaderboard_results.py — exact paper colors.
PALETTE = {
    "base":      "#E5E3DD",
    "pointwise": "#F8EED1",
    "pairwise":  "#EAEDCD",
    "tourno":    "#D1BEB1",
}

# Darker stroke colors for line/edge accents (improves contrast on white).
STROKE = {
    "base":      "#a8a59c",
    "pointwise": "#c8b568",
    "pairwise":  "#a3a86b",
    "tourno":    "#9a7e6c",
}

POSITIVE_CRITERIA = [
    "Adherence to Instructions",
    "Believable Character Actions",
    "Nuanced Characters",
    "Consistent Voice/Tone of Writing",
    "Imagery and Descriptive Quality",
    "Elegant Prose",
    "Emotionally Engaging",
    "Emotionally Complex",
    "Coherent",
    "Well-earned Lightness or Darkness",
    "Sentences Flow Naturally",
    "Overall Reader Engagement",
    "Overall Impression",
]

NEGATIVE_CRITERIA = [
    "Meandering",
    "Weak Dialogue",
    "Tell-Don't-Show",
    "Unsurprising or Uncreative",
    "Amateurish",
    "Purple Prose",
    "Overwrought",
    "Incongruent Ending Positivity",
    "Unearned Transformations",
]

ALL_CRITERIA = POSITIVE_CRITERIA + NEGATIVE_CRITERIA  # 22


# ---------- seaborn / mpl theme (matches leaderboard: light axes + outward ticks) ----------
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "axes.edgecolor":   "black",
    "axes.linewidth":   1.2,
    "axes.labelcolor":  "black",
    "text.color":       "black",
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.color":      "black",
    "ytick.color":      "black",
    "grid.color":       "#eeeeee",
    "grid.linewidth":   0.7,
})


# ---------- data ----------

def load_eval(key: str) -> dict:
    with open(EVAL_DIR / EVAL_FILENAMES[key], "r", encoding="utf-8") as f:
        return json.load(f)


def per_criterion_means(eval_data: dict) -> dict[str, float]:
    sums = {c: 0.0 for c in ALL_CRITERIA}
    counts = {c: 0 for c in ALL_CRITERIA}
    for row in eval_data["rows"]:
        for it in row["iterations"]:
            scores = it.get("judge_scores", {})
            for crit in ALL_CRITERIA:
                v = scores.get(crit)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv > 20.0:
                    continue
                if crit in NEGATIVE_CRITERIA:
                    fv = 20.0 - fv
                sums[crit] += fv
                counts[crit] += 1
    return {c: (sums[c] / counts[c]) if counts[c] else float("nan") for c in ALL_CRITERIA}


def short_label(crit: str) -> str:
    if len(crit) <= 18:
        return crit
    words = crit.split()
    mid = len(words) // 2
    while mid < len(words) and sum(len(w) + 1 for w in words[:mid]) < 14:
        mid += 1
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


# ---------- drawing helpers ----------

def draw_radar(ax, values, labels, *, fill_color, stroke_color, vmin, vmax,
               rings, ring_labels, panel_label):
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    closed_vals = list(values) + [values[0]]
    closed_angles = angles + [angles[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(vmin, vmax)
    ax.set_xticks(angles)
    ax.set_xticklabels([short_label(l) for l in labels], fontsize=11, color="black")
    ax.tick_params(axis="x", pad=8)

    ax.set_yticks(rings)
    ax.set_yticklabels(ring_labels, fontsize=11, color="black")
    ax.yaxis.grid(True, color="#cccccc", linewidth=0.7)
    ax.xaxis.grid(True, color="#cccccc", linewidth=0.7)
    ax.spines["polar"].set_color("black")
    ax.set_facecolor("white")

    ax.plot(closed_angles, closed_vals, color=stroke_color, linewidth=2.0)
    ax.fill(closed_angles, closed_vals, color=fill_color, alpha=0.55)

    # Small panel label below the radar (like a caption, not a true title)
    ax.set_title(panel_label, color="black", fontsize=14, pad=22, weight="semibold")


def draw_bars(ax, items, *, fill_color, stroke_color, panel_label):
    items = list(items)
    labels = [c for c, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    y = np.arange(len(items))

    ax.barh(y, vals, color=fill_color, edgecolor=stroke_color, linewidth=1.0, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14, color="black")
    ax.set_xlabel("Relative score", color="black", fontsize=14)
    ax.tick_params(axis="x", labelsize=13, colors="black")
    ax.tick_params(axis="y", length=0)
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
    ax.axvline(0, color="black", linewidth=1.0)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(panel_label, color="black", fontsize=14, pad=10, weight="semibold")

    for yi, v in zip(y, vals):
        ha = "left" if v >= 0 else "right"
        offset = 0.03 if v >= 0 else -0.03
        ax.text(v + offset, yi, f"{v:+.2f}", va="center", ha=ha,
                color="black", fontsize=13)


def make_panel(model_key, abs_scores, rel_scores, display_name) -> Path:
    fill = PALETTE[model_key]
    stroke = STROKE[model_key]

    fig = plt.figure(figsize=(15, 12), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.40,
                          left=0.13, right=0.95, top=0.93, bottom=0.06)

    # Top-left: absolute radar
    ax_abs = fig.add_subplot(gs[0, 0], projection="polar")
    abs_vals = [abs_scores[c] for c in ALL_CRITERIA]
    draw_radar(
        ax_abs, abs_vals, ALL_CRITERIA,
        fill_color=fill, stroke_color=stroke,
        vmin=0, vmax=20,
        rings=[5, 10, 15, 20], ring_labels=["5", "10", "15", "20"],
        panel_label="Absolute scores",
    )

    # Top-right: relative radar
    ax_rel = fig.add_subplot(gs[0, 1], projection="polar")
    rel_vals = [rel_scores[c] for c in ALL_CRITERIA]
    rmax = max(0.40, max(abs(v) for v in rel_vals) * 1.1)
    rings = [-rmax * 0.66, -rmax * 0.33, 0, rmax * 0.33, rmax * 0.66]
    ring_labels = [f"{r:+.2f}" for r in rings]
    draw_radar(
        ax_rel, rel_vals, ALL_CRITERIA,
        fill_color=fill, stroke_color=stroke,
        vmin=-rmax, vmax=rmax,
        rings=rings, ring_labels=ring_labels,
        panel_label="Relative scores",
    )

    # Bottom-left: strengths
    ax_str = fig.add_subplot(gs[1, 0])
    sorted_rel = sorted(rel_scores.items(), key=lambda kv: kv[1], reverse=True)
    strengths = sorted_rel[:5]
    weaknesses = sorted_rel[-5:][::-1]
    draw_bars(ax_str, strengths, fill_color=fill, stroke_color=stroke,
              panel_label="Top 5 strengths")

    # Bottom-right: weaknesses
    ax_weak = fig.add_subplot(gs[1, 1])
    draw_bars(ax_weak, weaknesses, fill_color=fill, stroke_color=stroke,
              panel_label="Top 5 weaknesses")

    out_png = FIG_DIR / f"cwb_abilities_{model_key}.png"
    out_pdf = FIG_DIR / f"cwb_abilities_{model_key}.pdf"
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    return out_png


# ---------- main ----------

def main() -> None:
    abs_per_model: dict[str, dict[str, float]] = {}
    for key in MODELS:
        eval_data = load_eval(key)
        abs_per_model[key] = per_criterion_means(eval_data)

    # Pool means across all models per criterion
    pool_mean = {}
    for crit in ALL_CRITERIA:
        vals = [abs_per_model[m][crit] for m in MODELS
                if not np.isnan(abs_per_model[m][crit])]
        pool_mean[crit] = float(np.mean(vals)) if vals else float("nan")

    rel_per_model = {
        key: {c: abs_per_model[key][c] - pool_mean[c] for c in ALL_CRITERIA}
        for key in MODELS
    }

    print("Pool means (cross-model averages):")
    for c in ALL_CRITERIA:
        print(f"  {c:42s} {pool_mean[c]:6.3f}")
    print()

    out_paths = []
    for key, display in MODELS.items():
        p = make_panel(key, abs_per_model[key], rel_per_model[key], display)
        out_paths.append(p)
        sorted_rel = sorted(rel_per_model[key].items(), key=lambda kv: kv[1], reverse=True)
        print(f"wrote {p}")
        print(f"  top strengths : {sorted_rel[:3]}")
        print(f"  top weaknesses: {sorted_rel[-3:][::-1]}")

    print(f"\n{len(out_paths)} figures (PNG + PDF) written.")


if __name__ == "__main__":
    main()
