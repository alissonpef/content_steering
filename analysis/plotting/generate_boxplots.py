import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import logging
import re

import matplotlib.pyplot as plt
import pandas as pd
from plot_utils import (
    CB_BLACK,
    CB_RED,
    KNOWN_STRATEGY_KEYS,
    STRATEGY_LEGEND_ORDER,
    apply_global_style,
    configure_logger,
    extract_strategy_from_filename,
    get_strategy_style,
    save_figure,
)

logger = logging.getLogger("generate_boxplots")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "logs", "aggregated")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "results")


def _extract_strategy(fname_no_ext: str) -> str:
    return extract_strategy_from_filename(fname_no_ext)


def _extract_scenario_from_filename(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    m = re.search("_scenario\\d+_([a-zA-Z0-9_]+)_average$", name)
    if m:
        return m.group(1).lower()
    m2 = re.search("_(baseline|mobility|spam|spam_extreme)_average$", name)
    if m2:
        return m2.group(1).lower()
    return "all"


def _apply_box_style(bp, color: str, is_hero: bool = False):
    edge_lw = 2.2 if is_hero else 1.2
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.55 if not is_hero else 0.7)
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


def generate_comparison_boxplot(all_data: dict, metric: str, output_dir: str, scenario_key: str):
    ordered_keys = [k for k in STRATEGY_LEGEND_ORDER if k in all_data]
    ordered_keys += sorted(k for k in all_data if k not in ordered_keys)
    plot_data, labels, colors, heroes = ([], [], [], [])
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
    bp = ax.boxplot(plot_data, vert=True, patch_artist=True, tick_labels=labels, widths=0.5)
    for _i, (box, color, hero) in enumerate(zip(bp["boxes"], colors, heroes, strict=False)):
        box.set_facecolor(color)
        box.set_alpha(0.55 if not hero else 0.7)
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
    ax.set_title(f"Strategy Latency Distribution ({scenario_key.title()})", pad=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Strategy")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(True, axis="y")
    fig.tight_layout()
    base = metric.replace("experienced_latency_ms", "latency")
    scenario_output_dir = os.path.join(output_dir, "comparative_analysis", scenario_key)
    save_figure(fig, os.path.join(scenario_output_dir, f"all_strategies_{base}_boxplot"))


def main():
    parser = argparse.ArgumentParser(description="Publication-ready boxplots from aggregated logs.")
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
    if not os.path.isdir(args.agg_dir):
        logger.error(f"Directory not found: {args.agg_dir}")
        return
    entries_by_scenario: dict[str, dict[str, pd.DataFrame]] = {}
    for fn in sorted(os.listdir(args.agg_dir)):
        if not (fn.startswith("log_") and fn.endswith("_average.csv")):
            continue
        path = os.path.join(args.agg_dir, fn)
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            sk = _extract_strategy(os.path.splitext(fn)[0])
            if sk not in KNOWN_STRATEGY_KEYS:
                continue
            if args.metric not in df.columns or df[args.metric].dropna().empty:
                continue
            scenario_key = _extract_scenario_from_filename(fn)
            entries_by_scenario.setdefault(scenario_key, {})[sk] = df
        except Exception as exc:
            logger.error(f"Error processing {fn}: {exc}")
    if not entries_by_scenario:
        logger.warning("No scenario data available for comparison boxplots.")
        return
    for scenario_key, all_data in sorted(entries_by_scenario.items()):
        generate_comparison_boxplot(all_data, args.metric, args.output_dir, scenario_key)
        logger.info(
            f"Scenario boxplot saved to {os.path.join(args.output_dir, 'comparative_analysis', scenario_key)}"
        )
    logger.info("Boxplot generation complete.")


if __name__ == "__main__":
    main()
