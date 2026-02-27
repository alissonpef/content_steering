import os
import argparse
import logging
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
)

logger = logging.getLogger("compare_strategies")
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "logs", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
FILL_ALPHA = 0.15
WINDOW_SIZE = 5


def plot_average_latency_comparison(
    agg_dir: str,
    output_dir: str = OUTPUT_DIR,
    metric: str = "experienced_latency_ms",
    max_time: float = None,
):
    if not os.path.isdir(agg_dir):
        logger.error(f"Aggregated logs directory not found: {agg_dir}")
        return
    entries: list[tuple[str, str, pd.DataFrame]] = []
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
            entries.append((strat, fn, df))
        except Exception:
            continue
    if not entries:
        logger.warning("No aggregated files found for comparison.")
        return
    if max_time is None:
        xlim = max(df["sim_time_client"].max() for _, _, df in entries)
    else:
        xlim = max_time
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    plotted = set()

    def _order(item):
        sk = item[0]
        try:
            return STRATEGY_LEGEND_ORDER.index(sk)
        except ValueError:
            return len(STRATEGY_LEGEND_ORDER)

    for strat_key, fn, df in sorted(entries, key=_order, reverse=True):
        if strat_key in plotted:
            continue
        df = df[df["sim_time_client"] <= xlim].sort_values("sim_time_client").copy()
        sub = df.dropna(subset=["sim_time_client", metric])
        if sub.empty:
            continue
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
        if std_col in df.columns:
            std = (
                df.loc[sub.index, std_col]
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
        logger.warning("No data plotted for comparison.")
        plt.close(fig)
        return
    nice_metric = metric.replace("_", " ").replace("ms", "").strip().title()
    if metric == "dynamic_best_server_latency":
        nice_metric = "Oracle Optimal Latency"
    format_axes(
        ax,
        f"Strategy Comparison — Avg. {nice_metric}",
        "Simulation Time (s)",
        "Latency (ms)",
        legend_loc="upper right",
        xlim_max=xlim,
    )
    sort_legend_by_strategy(ax)
    fig.tight_layout()
    base = metric.replace("experienced_latency_ms", "latency")
    save_figure(fig, os.path.join(output_dir, f"all_strategies_{base}_comparison"))
    logger.info(f"Comparison plot saved to {output_dir}")


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
    parser.add_argument("--max_time", type=float, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    apply_global_style()
    configure_logger(logger, args.verbose)
    plot_average_latency_comparison(
        args.agg_dir, args.output_dir, metric=args.metric, max_time=args.max_time
    )


if __name__ == "__main__":
    main()
