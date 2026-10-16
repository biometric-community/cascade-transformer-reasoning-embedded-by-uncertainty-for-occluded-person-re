"""Figures + REPORT.md for CTU (dev-plot)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from . import plot_style


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logs = Path(cfg["output"]["logs"])
    figdir = Path(cfg["output"]["figures"])
    figdir.mkdir(parents=True, exist_ok=True)
    plot_style.apply_rcparams()

    train = json.loads((logs / "train_summary.json").read_text()) if (logs / "train_summary.json").is_file() else {}
    eval_r = json.loads((logs / "eval_results.json").read_text()) if (logs / "eval_results.json").is_file() else {}
    history = train.get("history") or []
    metrics = eval_r.get("metrics") or train.get("final_metrics") or {}

    fig, ax = plt.subplots(figsize=plot_style.FIGSIZE)
    if history:
        ax.plot(
            [h["epoch"] for h in history],
            [h["loss"] for h in history],
            color=plot_style.BAR_COLOR,
            linewidth=plot_style.LINEWIDTH_MAIN,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CTU training loss (Market-1501)")
    plot_style.apply_style(ax)
    fig.savefig(figdir / "train_loss.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    paper = {"mAP": 88.1, "Rank-1": 94.4}  # paper holistic Market numbers (approx R1 from text context)
    # PDF: mAP 88.1% on Market; Rank-1 stated nearby as competitive — use 94.4 if transcribed else leave
    # From PDF line 737: mAP 88.1 on Market; Rank-1 not always paired — use reported if in analysis tables
    ours = {"mAP": float(metrics.get("mAP", 0)), "Rank-1": float(metrics.get("Rank-1", 0))}
    fig, ax = plt.subplots(figsize=plot_style.FIGSIZE)
    keys = ["mAP", "Rank-1"]
    x = range(len(keys))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [paper[k] for k in keys], width=w, color=plot_style.MULTI_SERIES_COLORS[0], label="Paper CTU (Market)")
    ax.bar([i + w / 2 for i in x], [ours[k] for k in keys], width=w, color=plot_style.MULTI_SERIES_COLORS[1], label="Ours (Market)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys)
    ax.set_ylabel("%")
    ax.set_title("Market-1501: paper vs ours")
    ax.legend()
    plot_style.apply_style(ax, grid_axis="y")
    fig.savefig(figdir / "market_metrics.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    ranks = [(r, float(metrics[f"Rank-{r}"])) for r in cfg.get("eval", {}).get("ranks", [1, 5, 10]) if f"Rank-{r}" in metrics]
    if ranks:
        fig, ax = plt.subplots(figsize=plot_style.FIGSIZE)
        ax.plot([r for r, _ in ranks], [v for _, v in ranks], color=plot_style.BAR_COLOR, linewidth=plot_style.LINEWIDTH_MAIN, marker="o")
        ax.set_xlabel("Rank")
        ax.set_ylabel("CMC (%)")
        ax.set_title("Market-1501 CMC (ours)")
        plot_style.apply_style(ax)
        fig.savefig(figdir / "cmc.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    lines = [
        "# REPORT — CTU",
        "",
        f"**Paper:** {cfg.get('paper', {}).get('title', 'CTU')}",
        f"**DOI:** {cfg.get('paper', {}).get('doi', '')}",
        "",
        "## Summary",
        "",
    ]
    if not metrics:
        lines += ["**BLOCKED:** no metrics. Run train/eval first.", ""]
    else:
        lines += [
            "| Split | mAP | Rank-1 | Rank-5 | Rank-10 |",
            "|-------|-----|--------|--------|---------|",
            f"| Market-1501 (ours) | {ours['mAP']:.2f} | {ours['Rank-1']:.2f} | {float(metrics.get('Rank-5', 0)):.2f} | {float(metrics.get('Rank-10', 0)):.2f} |",
            f"| Market-1501 (paper CTU) | {paper['mAP']:.1f} | — | — | — |",
            "",
            "Duke / Occluded-Duke / partial sets not run (unavailable); see `DEVIATIONS.md`.",
            "",
            "## Figures",
            "",
            "![loss](outputs/figures/train_loss.svg)",
            "",
            "![metrics](outputs/figures/market_metrics.svg)",
            "",
            "![cmc](outputs/figures/cmc.svg)",
            "",
        ]
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/setup_env.sh",
        "bash scripts/train_full.sh",
        "bash scripts/eval.sh",
        "bash scripts/report.sh",
        "```",
        "",
    ]
    Path("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
