import os
import argparse
import logging
import re
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import (
    apply_global_style,
    configure_logger,
    save_figure,
    format_axes,
    sort_legend_by_strategy,
    get_strategy_style,
    extract_strategy_from_filename,
    STRATEGY_LEGEND_ORDER,
    KNOWN_STRATEGY_KEYS,
)

logger = logging.getLogger("compare_strategies")
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "logs", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
FILL_ALPHA = 0.15
WINDOW_SIZE = 5


def _extract_scenario_from_filename(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    m = re.search(r"_scenario\d+_([a-zA-Z0-9_]+)_average$", name)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"_(baseline|mobility|spam)_average$", name)
    if m2:
        return m2.group(1).lower()
    return "all"


def plot_average_latency_comparison(
    agg_dir: str,
    output_dir: str = OUTPUT_DIR,
    metric: str = "experienced_latency_ms",
    max_time: float = None,
):
    if not os.path.isdir(agg_dir):
        logger.error(f"Aggregated logs directory not found: {agg_dir}")
        return
    entries_by_scenario: dict[str, list[tuple[str, str, pd.DataFrame]]] = {}
    for fn in sorted(os.listdir(agg_dir)):
        if not (fn.startswith("log_") and "_average" in fn and fn.endswith(".csv")):
            continue
        path = os.path.join(agg_dir, fn)
        try:
            df = pd.read_csv(path)
            if "sim_time_client" not in df.columns or metric not in df.columns:
                continue
            fname = os.path.splitext(fn)[0]
            strat = extract_strategy_from_filename(fname)
            if "rl_strategy" in df.columns and not df["rl_strategy"].dropna().empty:
                strat = df["rl_strategy"].dropna().iloc[0]
            if strat not in KNOWN_STRATEGY_KEYS:
                continue
            scenario_key = _extract_scenario_from_filename(fn)
            entries_by_scenario.setdefault(scenario_key, []).append((strat, fn, df))
        except Exception:
            continue
    if not entries_by_scenario:
        logger.warning("No aggregated files found for comparison.")
        return

    def _order(item):
        sk = item[0]
        try:
            return STRATEGY_LEGEND_ORDER.index(sk)
        except ValueError:
            return len(STRATEGY_LEGEND_ORDER)

    for scenario_key, entries in sorted(entries_by_scenario.items()):
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        plotted = set()
        scenario_xmax = 0.0
        for strat_key, fn, df in sorted(entries, key=_order, reverse=True):
            if strat_key in plotted:
                continue
            subdf = df.sort_values("sim_time_client").copy()
            if max_time is not None:
                subdf = subdf[subdf["sim_time_client"] <= max_time].copy()
            sub = subdf.dropna(subset=["sim_time_client", metric])
            if sub.empty:
                continue
            scenario_xmax = max(scenario_xmax, float(sub["sim_time_client"].max()))
            style = get_strategy_style(strat_key)
            y = sub[metric].rolling(WINDOW_SIZE, center=True, min_periods=1).mean()
            ax.plot(
                sub["sim_time_client"],
                y,
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                label=style["label"],
            )
            std_col = metric + "_std_agg"
            if std_col in subdf.columns:
                std = (
                    subdf.loc[sub.index, std_col]
                    .rolling(WINDOW_SIZE, center=True, min_periods=1)
                    .mean()
                )
                ax.fill_between(
                    sub["sim_time_client"],
                    y - std,
                    y + std,
                    color=style["color"],
                    alpha=FILL_ALPHA,
                    linewidth=0,
                )
            plotted.add(strat_key)

        if not plotted:
            logger.warning(f"No data plotted for comparison in scenario: {scenario_key}")
            plt.close(fig)
            continue

        nice_metric = metric.replace("_", " ").replace("ms", "").strip().title()
        if metric == "dynamic_best_server_latency":
            nice_metric = "Oracle Optimal Latency"

        title = f"Strategy Comparison ({scenario_key.title()}) — Avg. {nice_metric}"
        format_axes(
            ax,
            title,
            "Simulation Time (s)",
            "Latency (ms)",
            legend_loc="upper right",
            xlim_max=(max_time if max_time is not None else scenario_xmax),
        )
        sort_legend_by_strategy(ax)
        fig.tight_layout()

        scenario_output_dir = os.path.join(output_dir, "comparison_by_scenario", scenario_key)
        base = metric.replace("experienced_latency_ms", "latency")
        save_figure(
            fig,
            os.path.join(scenario_output_dir, f"all_strategies_{base}_comparison"),
        )
        logger.info(f"Comparison plot saved to {scenario_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Publication-ready comparison of strategy latencies."
    )
    parser.add_argument(
        "--agg_dir", default=PROCESSED_DIR, help="Directory with aggregated CSV files."
    )
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--metric",
        default="experienced_latency_ms",
        choices=[
            "experienced_latency_ms",
            "experienced_latency_ms_CLIENT",
            "dynamic_best_server_latency",
        ],
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=None,
        help="Optional max simulation time (seconds). If omitted, uses data max time.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    apply_global_style()
    configure_logger(logger, args.verbose)
    plot_average_latency_comparison(
        args.agg_dir, args.output_dir, metric=args.metric, max_time=args.max_time
    )


if __name__ == "__main__":
    main()
