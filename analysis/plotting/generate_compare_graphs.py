import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import logging
import re
import pandas as pd
import numpy as np
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
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "logs", "aggregated")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "results")
WINDOW_SIZE = 25


def _extract_scenario_from_filename(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    m = re.search("_scenario\\d+_([a-zA-Z0-9_]+)_average$", name)
    if m:
        return m.group(1).lower()
    m2 = re.search("_(baseline|mobility|spam)_average$", name)
    if m2:
        return m2.group(1).lower()
    return "all"


def plot_average_latency_comparison(
    agg_dir: str,
    output_dir: str = OUTPUT_DIR,
    metric: str = "experienced_latency_ms",
    max_time: float | None = None,
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
            if (
                "rl_strategy" in df.columns
                and len(pd.Series(df["rl_strategy"]).dropna()) > 0
            ):
                strat = str(pd.Series(df["rl_strategy"]).dropna().iloc[0])
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
            max_val = sub["sim_time_client"].max()
            if max_val is not None and (not pd.isna(max_val)):
                scenario_xmax = max(scenario_xmax, float(max_val))
            style = get_strategy_style(strat_key)
            y = sub[metric].rolling(WINDOW_SIZE, center=True, min_periods=1).mean()
            ax.plot(
                sub["sim_time_client"].to_numpy(),
                np.asarray(y),
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                zorder=50 if strat_key == "best" else style["zorder"],
                label=style["label"],
            )
            plotted.add(strat_key)
        if not plotted:
            logger.warning(
                f"No data plotted for comparison in scenario: {scenario_key}"
            )
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
            legend_loc="best",
            xlim_max=max_time if max_time is not None else scenario_xmax,
        )
        sort_legend_by_strategy(
            ax,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=4,
        )
        fig.tight_layout()
        scenario_output_dir = os.path.join(
            output_dir, "comparative_analysis", scenario_key
        )
        base = metric.replace("experienced_latency_ms", "latency")
        save_figure(
            fig, os.path.join(scenario_output_dir, f"all_strategies_{base}_comparison")
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
    plot_cumulative_regret(args.agg_dir, args.output_dir, max_time=args.max_time)
    plot_selection_distribution(args.agg_dir, args.output_dir)


def plot_cumulative_regret(
    agg_dir: str, output_dir: str, max_time: float | None = None
):
    entries_by_scenario: dict[str, list[tuple[str, str, pd.DataFrame]]] = {}
    for fn in sorted(os.listdir(agg_dir)):
        if not (fn.startswith("log_") and "_average" in fn and fn.endswith(".csv")):
            continue
        path = os.path.join(agg_dir, fn)
        try:
            df = pd.read_csv(path)
            if (
                "sim_time_client" not in df.columns
                or "experienced_latency_ms" not in df.columns
                or "dynamic_best_server_latency" not in df.columns
            ):
                continue
            strat = extract_strategy_from_filename(os.path.splitext(fn)[0])
            if (
                "rl_strategy" in df.columns
                and len(pd.Series(df["rl_strategy"]).dropna()) > 0
            ):
                strat = str(pd.Series(df["rl_strategy"]).dropna().iloc[0])
            if strat not in KNOWN_STRATEGY_KEYS or strat == "best":
                continue
            scenario_key = _extract_scenario_from_filename(fn)
            entries_by_scenario.setdefault(scenario_key, []).append((strat, fn, df))
        except Exception:
            continue

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
            sub = subdf.dropna(
                subset=[
                    "sim_time_client",
                    "experienced_latency_ms",
                    "dynamic_best_server_latency",
                ]
            )
            if sub.empty:
                continue
            sub["regret"] = (
                sub["experienced_latency_ms"] - sub["dynamic_best_server_latency"]
            )
            sub["regret"] = sub["regret"].clip(lower=0)
            sub["cumulative_regret"] = sub["regret"].cumsum()
            max_val_reg = sub["sim_time_client"].max()
            if max_val_reg is not None and (not pd.isna(max_val_reg)):
                scenario_xmax = max(scenario_xmax, float(max_val_reg))
            style = get_strategy_style(strat_key)
            ax.plot(
                sub["sim_time_client"].to_numpy(),
                sub["cumulative_regret"].to_numpy(),
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                label=style["label"],
            )
            plotted.add(strat_key)
        if not plotted:
            plt.close(fig)
            continue
        format_axes(
            ax,
            f"Cumulative Regret ({scenario_key.title()})",
            "Simulation Time (s)",
            "Cumulative Regret (ms)",
            legend_loc="best",
            xlim_max=max_time if max_time is not None else scenario_xmax,
        )
        sort_legend_by_strategy(
            ax,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
        )
        fig.tight_layout()
        scenario_output_dir = os.path.join(
            output_dir, "comparative_analysis", scenario_key
        )
        save_figure(
            fig, os.path.join(scenario_output_dir, "cumulative_regret_comparison")
        )


def plot_selection_distribution(agg_dir: str, output_dir: str):
    import seaborn as sns
    from plot_utils import get_server_label

    entries_by_scenario: dict[str, list[tuple[str, str, pd.DataFrame, list[str]]]] = {}
    for fn in sorted(os.listdir(agg_dir)):
        if not (fn.startswith("log_") and "_average" in fn and fn.endswith(".csv")):
            continue
        path = os.path.join(agg_dir, fn)
        try:
            df = pd.read_csv(path)
            cnt_cols = sorted(
                (
                    c
                    for c in df.columns
                    if c.startswith("count_") and (not c.endswith("_std_agg"))
                )
            )
            if not cnt_cols:
                continue
            strat = extract_strategy_from_filename(os.path.splitext(fn)[0])
            if (
                "rl_strategy" in df.columns
                and len(pd.Series(df["rl_strategy"]).dropna()) > 0
            ):
                strat = str(pd.Series(df["rl_strategy"]).dropna().iloc[0])
            if strat not in KNOWN_STRATEGY_KEYS or strat == "best":
                continue
            scenario_key = _extract_scenario_from_filename(fn)
            entries_by_scenario.setdefault(scenario_key, []).append(
                (strat, fn, df, cnt_cols)
            )
        except Exception:
            continue

    def _order(item):
        sk = item[0]
        try:
            return STRATEGY_LEGEND_ORDER.index(sk)
        except ValueError:
            return len(STRATEGY_LEGEND_ORDER)

    for scenario_key, entries in sorted(entries_by_scenario.items()):
        server_keys = set()
        for strat, fn, df, cnt_cols in entries:
            for c in cnt_cols:
                server_keys.add(c.replace("count_", "").replace("_", "-"))
        server_keys = sorted(list(server_keys))
        matrix = []
        labels = []
        for strat_key, fn, df, cnt_cols in sorted(entries, key=_order):
            if strat_key in labels:
                continue
            row_data = []
            last_row = (
                df.dropna(subset=cnt_cols).iloc[-1]
                if not df.dropna(subset=cnt_cols).empty
                else None
            )
            if last_row is None:
                continue
            total_pulls = sum((last_row[c] for c in cnt_cols))
            if total_pulls <= 0:
                continue
            for sk in server_keys:
                c_name = "count_" + sk.replace("-", "_")
                val = last_row.get(c_name, 0)
                row_data.append(val / total_pulls)
            matrix.append(row_data)
            labels.append(get_strategy_style(strat_key)["label"])
        if not matrix:
            continue
        fig, ax = plt.subplots(figsize=(6, 4 + 0.3 * len(matrix)))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1%",
            cmap="YlGnBu",
            xticklabels=[get_server_label(sk) for sk in server_keys],
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(f"Final Selection Distribution ({scenario_key.title()})", pad=15)
        ax.set_xlabel("Servers")
        ax.set_ylabel("Strategies")
        fig.tight_layout()
        scenario_output_dir = os.path.join(
            output_dir, "comparative_analysis", scenario_key
        )
        save_figure(fig, os.path.join(scenario_output_dir, "selection_heatmap"))


if __name__ == "__main__":
    main()
