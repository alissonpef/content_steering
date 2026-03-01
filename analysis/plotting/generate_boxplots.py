import os
import re
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import (
    apply_global_style,
    configure_logger,
    save_figure,
    get_strategy_style,
    extract_strategy_from_filename,
    STRATEGY_LEGEND_ORDER,
    CB_RED,
    CB_BLACK,
)

logger = logging.getLogger("generate_boxplots")
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "logs", "aggregated_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "boxplots")


def _extract_strategy(fname_no_ext: str) -> str:
    return extract_strategy_from_filename(fname_no_ext)


def _apply_box_style(bp, color: str, is_hero: bool = False):
    edge_lw = 2.2 if is_hero else 1.2
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.55 if not is_hero else 0.70)
        box.set_edgecolor(color)
        box.set_linewidth(edge_lw)
    for med in bp["medians"]:
        med.set_color(CB_BLACK)
        med.set_linewidth(2.0)
    for whisk in bp["whiskers"]:
        whisk.set_color(color)
        whisk.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_color(color)
        cap.set_linewidth(1.0)
    for flier in bp["fliers"]:
        flier.set(
            marker="D",
            markerfacecolor=CB_RED,
            markeredgecolor=CB_RED,
            markersize=3.5,
            alpha=0.6,
        )


def generate_individual_boxplot(df, strat_key, metric, output_dir):
    if df.empty or metric not in df.columns or df[metric].dropna().empty:
        return
    style = get_strategy_style(strat_key)
    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    bp = ax.boxplot(
        df[metric].dropna(),
        vert=True,
        patch_artist=True,
        labels=[style["label"]],
        widths=0.45,
    )
    _apply_box_style(bp, style["color"], is_hero=(strat_key == "linucb"))
    ax.set_title(f"Latency Distribution: {style['label']}", pad=8)
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_figure(
        fig, os.path.join(output_dir, f"boxplot_individual_{strat_key}_{metric}")
    )


def generate_comparison_boxplot(all_data: dict, metric, output_dir):
    ordered_keys = [k for k in STRATEGY_LEGEND_ORDER if k in all_data]
    ordered_keys += sorted(k for k in all_data if k not in ordered_keys)
    plot_data, labels, colors, heroes = [], [], [], []
    for sk in ordered_keys:
        df = all_data[sk]
        if df.empty or metric not in df.columns or df[metric].dropna().empty:
            continue
        plot_data.append(df[metric].dropna().tolist())
        style = get_strategy_style(sk)
        labels.append(style["label"])
        colors.append(style["color"])
        heroes.append(sk == "linucb")
    if not plot_data:
        logger.warning("No data for comparison boxplot.")
        return
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 1.4), 4.5))
    bp = ax.boxplot(plot_data, vert=True, patch_artist=True, labels=labels, widths=0.5)
    for i, (box, color, hero) in enumerate(zip(bp["boxes"], colors, heroes)):
        box.set_facecolor(color)
        box.set_alpha(0.55 if not hero else 0.70)
        box.set_edgecolor(color)
        box.set_linewidth(2.2 if hero else 1.2)
    for med in bp["medians"]:
        med.set_color(CB_BLACK)
        med.set_linewidth(2.0)
    for whisk in bp["whiskers"]:
        whisk.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_linewidth(1.0)
    for flier in bp["fliers"]:
        flier.set(
            marker="D",
            markerfacecolor=CB_RED,
            markeredgecolor=CB_RED,
            markersize=3.5,
            alpha=0.6,
        )
    nice = metric.replace("_", " ").replace("ms", "").strip().title()
    ax.set_title(f"Latency Distribution by Strategy", pad=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Strategy")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, f"boxplot_comparison_all_{metric}"))


def main():
    parser = argparse.ArgumentParser(
        description="Publication-ready boxplots from aggregated logs."
    )
    parser.add_argument("--agg_dir", default=PROCESSED_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--metric",
        default="experienced_latency_ms",
        choices=[
            "experienced_latency_ms",
            "experienced_latency_ms_CLIENT",
            "experienced_latency_ms_ORACLE",
            "dynamic_best_server_latency",
        ],
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    apply_global_style()
    configure_logger(logger, args.verbose)
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isdir(args.agg_dir):
        logger.error(f"Directory not found: {args.agg_dir}")
        return
    all_data = {}
    for fn in sorted(os.listdir(args.agg_dir)):
        if not (fn.startswith("log_") and fn.endswith("_average.csv")):
            continue
        path = os.path.join(args.agg_dir, fn)
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            sk = _extract_strategy(os.path.splitext(fn)[0])
            if sk == "Unknown":
                continue
            generate_individual_boxplot(df, sk, args.metric, args.output_dir)
            all_data[sk] = df
        except Exception as exc:
            logger.error(f"Error processing {fn}: {exc}")
    if all_data:
        generate_comparison_boxplot(all_data, args.metric, args.output_dir)
    logger.info("Boxplot generation complete.")


if __name__ == "__main__":
    main()
