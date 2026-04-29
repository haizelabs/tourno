"""Paper-ready single-plot figures for cwbench v3 leaderboard auxiliary metrics.

Conventions (per user):
  - One figure per metric (no subplots), saved as both .pdf and .png.
  - Seaborn 'whitegrid' theme, paper context.
  - 14pt+ fonts; clean axes; no titles.
  - Custom palette matching reference HealthBench-style figure:
      base = light grey, pointwise = pale yellow, pairwise = pale green, tourno = warm tan.

Reads `cwbench-rl/eval/leaderboard_metrics.json` (run compute_leaderboard_metrics.py first)
plus the raw eval JSONs (for piece-score bootstrap CIs and length distributions).

Outputs to `cwbench-rl/eval/figures/`:
  rubric_score_ci.{pdf,png}     — bar with 95% bootstrap CI
  slop.{pdf,png}                — slop index (lower=better)
  repetition.{pdf,png}          — n-gram repetition (lower=better)
  vocab_complexity.{pdf,png}    — Style / vocab complexity
  length.{pdf,png}              — mean ± std token length
  length_hist.{pdf,png}         — full length distribution overlay
  leaderboard_table.{pdf,png}   — tabular summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "cwbench-experiments"))
from bootstrap_eval import bootstrap_ci, piece_scores_from_eval  # noqa: E402

EVAL_DIR = REPO_ROOT / "cwbench-rl" / "eval"
FIG_DIR = EVAL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = EVAL_DIR / "leaderboard_metrics.json"

EVAL_FILES = [
    ("base", EVAL_DIR / "base.json"),
    ("pointwise", EVAL_DIR / "pointwise.json"),
    ("pairwise", EVAL_DIR / "pairwise.json"),
    ("tourno", EVAL_DIR / "tourno-step120.json"),
]

DISPLAY = {
    "base": "Base",
    "pointwise": "Pointwise",
    "pairwise": "Pairwise",
    "tourno": "TournO",
}

# Exact paper palette (per user spec).
COLORS = {
    "base":      "#E5E3DD",  # very light cream-grey
    "pointwise": "#F8EED1",  # pale yellow
    "pairwise":  "#EAEDCD",  # pale green
    "tourno":    "#D1BEB1",  # warm tan / mauve
}

# ---------- seaborn / mpl global config (HealthBench-style: black axes + outward ticks) ----------
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    # Both x and y spines visible & black. Top/right hidden (clean paper look).
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.edgecolor":     "black",
    "axes.linewidth":     1.2,
    "axes.labelcolor":    "black",
    "text.color":         "black",
    # Outward (downward on x, leftward on y) tick marks; small + crisp.
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.major.width":  1.0,
    "ytick.major.width":  1.0,
    "xtick.color":        "black",
    "ytick.color":        "black",
    # Subtle horizontal grid only.
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.color":         "#eeeeee",
    "grid.linewidth":     0.7,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
})


def save_both(fig, base_path: Path) -> Path:
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    return png


def _xlabels() -> list[str]:
    return [DISPLAY[k] for k, _ in EVAL_FILES]


def _palette() -> list[str]:
    return [COLORS[k] for k, _ in EVAL_FILES]


# ---------- 1. Rubric with 95% CI ----------

def fig_rubric_with_ci(rubric_rows: list[tuple[str, float, float]]) -> Path:
    keys = [k for k, _, _ in rubric_rows]
    means = [m for _, m, _ in rubric_rows]
    halfs = [h for _, _, h in rubric_rows]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = np.arange(len(keys))
    bars = ax.bar(
        xs, means,
        yerr=halfs,
        color=[COLORS[k] for k in keys],
        edgecolor="black", linewidth=0.8,
        capsize=6,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    for bar, m, h in zip(bars, means, halfs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + h + 0.6,
            f"{m:.2f}",
            ha="center", va="bottom", fontsize=14,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[k] for k in keys])
    ax.set_xlabel("Model")
    ax.set_ylabel("cwbench v3 rubric score")
    ymax = max(m + h for m, h in zip(means, halfs))
    ax.set_ylim(0, ymax + 6)
    fig.tight_layout()
    out = save_both(fig, FIG_DIR / "cwb_rubric_score_ci")
    plt.close(fig)
    return out


# ---------- 2/3/4. Single-metric bar charts ----------

def _bar_metric(metric_key: str, ylabel: str, fname: str,
                metrics: dict, lower_is_better: bool = False) -> Path:
    keys = [k for k, _ in EVAL_FILES if k in metrics]
    vals = [metrics[k][metric_key] for k in keys]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = np.arange(len(keys))
    bars = ax.bar(
        xs, vals,
        color=[COLORS[k] for k in keys],
        edgecolor="black", linewidth=0.8,
    )
    vmax = max(vals)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + vmax * 0.022,
            f"{v:.2f}",
            ha="center", va="bottom", fontsize=14,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[k] for k in keys])
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, vmax * 1.18)
    fig.tight_layout()
    fname_cwb = fname if fname.startswith("cwb_") else f"cwb_{fname}"
    out = save_both(fig, FIG_DIR / fname_cwb)
    plt.close(fig)
    return out


# ---------- 5. Length (mean ± std, tokens) ----------

def fig_length() -> Path:
    keys: list[str] = []
    means_tok: list[float] = []
    stds_tok: list[float] = []
    for key, path in EVAL_FILES:
        d = json.loads(Path(path).read_text())
        toks = [
            it.get("token_len", 0)
            for r in d.get("rows", [])
            for it in r.get("iterations", [])
            if it.get("token_len")
        ]
        keys.append(key)
        means_tok.append(float(np.mean(toks)) if toks else 0.0)
        stds_tok.append(float(np.std(toks)) if toks else 0.0)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = np.arange(len(keys))
    bars = ax.bar(
        xs, means_tok,
        yerr=stds_tok,
        color=[COLORS[k] for k in keys],
        edgecolor="black", linewidth=0.8,
        capsize=6,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    for bar, m, s in zip(bars, means_tok, stds_tok):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 60,
            f"{m:.0f}",
            ha="center", va="bottom", fontsize=14,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[k] for k in keys])
    ax.set_xlabel("Model")
    ax.set_ylabel("Tokens per completion")
    ax.set_ylim(0, max(m + s for m, s in zip(means_tok, stds_tok)) + 350)
    fig.tight_layout()
    out = save_both(fig, FIG_DIR / "cwb_length")
    plt.close(fig)
    return out


# ---------- 6. Length distribution overlay ----------

def fig_length_hist() -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.linspace(0, 4500, 40)
    for key, path in EVAL_FILES:
        d = json.loads(Path(path).read_text())
        toks = [
            it.get("token_len", 0)
            for r in d.get("rows", [])
            for it in r.get("iterations", [])
            if it.get("token_len")
        ]
        ax.hist(
            toks,
            bins=bins,
            histtype="step",
            linewidth=2.4,
            label=DISPLAY[key],
            color=COLORS[key],
        )
    ax.set_xlabel("Token length per completion")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out = save_both(fig, FIG_DIR / "cwb_length_hist")
    plt.close(fig)
    return out


# ---------- 7. Leaderboard table ----------

def fig_table(metrics: dict, rubric_rows: list[tuple[str, float, float]]) -> Path:
    rubric_lookup = {k: (m, h) for k, m, h in rubric_rows}
    headers = [
        "Model",
        "Rubric (0–100)",
        "Style",
        "Slop ↓",
        "Rep. ↓",
        "Tokens",
    ]
    rows = []
    for key, _ in EVAL_FILES:
        if key not in metrics:
            continue
        r = metrics[key]
        m, h = rubric_lookup.get(key, (None, None))
        rubric_str = f"{m:.2f} ± {h:.2f}" if m is not None else "—"
        rows.append([
            DISPLAY[key],
            rubric_str,
            f"{r['vocab_complexity']:.2f}",
            f"{r['slop_score']:.2f}",
            f"{r['repetition_score']:.2f}",
            f"{r['avg_token_len']:.0f}",
        ])

    fig, ax = plt.subplots(figsize=(9.4, 0.7 + 0.55 * (len(rows) + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 1.6)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#aaa")
        if i == 0:
            cell.set_facecolor("#e8eef6")
            cell.set_text_props(weight="bold")
        else:
            row_key = list(DISPLAY.values()).index(rows[i - 1][0])
            color_key = list(DISPLAY.keys())[row_key]
            cell.set_facecolor(COLORS[color_key] + "55")  # 33% alpha hex (approx)
    fig.tight_layout()
    out = save_both(fig, FIG_DIR / "cwb_leaderboard_table")
    plt.close(fig)
    return out


# ---------- main ----------


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text())

    # Compute bootstrap CI per model once
    rubric_rows: list[tuple[str, float, float]] = []
    for key, path in EVAL_FILES:
        _, scores, _ = piece_scores_from_eval(str(path))
        ci = bootstrap_ci(scores, n_boot=2000)
        rubric_rows.append((key, ci["point_0_100"], ci["ci_halfwidth_0_100"]))

    out_paths: list[Path] = []
    out_paths.append(fig_rubric_with_ci(rubric_rows))
    out_paths.append(_bar_metric(
        "slop_score", "Slop index", "slop", metrics, lower_is_better=True,
    ))
    out_paths.append(_bar_metric(
        "repetition_score", "Repetition index", "repetition", metrics, lower_is_better=True,
    ))
    out_paths.append(_bar_metric(
        "vocab_complexity", "Vocab complexity", "vocab_complexity", metrics,
    ))
    out_paths.append(fig_length())
    out_paths.append(fig_length_hist())
    out_paths.append(fig_table(metrics, rubric_rows))

    print("Saved paper-ready figures (PNG + PDF):")
    for p in out_paths:
        print(f"  {p.with_suffix('.png')}")
        print(f"  {p.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
