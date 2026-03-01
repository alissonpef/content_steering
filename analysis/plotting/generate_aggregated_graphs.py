import os
import re
import json
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import (
    apply_global_style,
    configure_logger,
    save_figure,
    format_axes,
    get_server_color,
    get_server_label,
    get_strategy_display_name,
    extract_strategy_from_filename,
    KNOWN_STRATEGY_KEYS,
    CB_BLACK,
    CB_ORANGE,
    KNOWN_SERVER_KEYS_UNDERSCORE,
)

logger = logging.getLogger("plot_aggregated_logs")
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "logs", "aggregated_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "consolidated_charts")
FILL_ALPHA = 0.18
SHOW_STD_BANDS = False


def parse_json_column(series: pd.Series, prefix: str = "") -> pd.DataFrame:
    rows, indices = [], []
    for idx, raw in series.dropna().items():
        try:
            d = json.loads(raw) if isinstance(raw, str) else {}
            if isinstance(d, dict):
                rows.append({f"{prefix}{k.replace('-', '_')}": v for k, v in d.items()})
                indices.append(idx)
        except (json.JSONDecodeError, TypeError):
            pass
    return pd.DataFrame(rows, index=indices) if rows else pd.DataFrame()


def _shade(ax, x, mean, std, color, alpha=FILL_ALPHA):
    if not SHOW_STD_BANDS:
        return
    if std is None or std.isna().all():
        return
    lo = mean - std
    hi = mean + std
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


def _resolve_oracle_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "dynamic_best_server_latency",
        "experienced_latency_ms_ORACLE",
        "experienced_latency_ms_oracle",
    ]
    for col in candidates:
        if col in df.columns and not df[col].dropna().empty:
            return col
    return None


def generate_plots_for_aggregated(csv_path: str, max_time: float = None):
    if not os.path.exists(csv_path):
        logger.error(f"Aggregated CSV not found: {csv_path}")
        return
    fname = os.path.splitext(os.path.basename(csv_path))[0]
    img_dir = os.path.join(OUTPUT_DIR, fname)
    os.makedirs(img_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning(f"Empty CSV: {csv_path}")
        return
    df.columns = [str(c).strip() for c in df.columns]
    strat_key = "N/A"
    if "rl_strategy" in df.columns and not df["rl_strategy"].dropna().empty:
        strat_key = df["rl_strategy"].dropna().iloc[0]
    else:
        strat_key = extract_strategy_from_filename(fname)
    if strat_key not in KNOWN_STRATEGY_KEYS:
        logger.info(f"Skipping unsupported/legacy strategy file: {fname}")
        return
    strat_display = get_strategy_display_name(strat_key)
    if max_time is not None:
        df = df[df["sim_time_client"] <= max_time].copy()
        if df.empty:
            logger.warning(f"No data ≤ {max_time}s in {fname}")
            return
    xlim = float(df["sim_time_client"].max()) if "sim_time_client" in df.columns else None
    has_std = "experienced_latency_ms_std_agg" in df.columns
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    h, l = [], []
    if "experienced_latency_ms" in df.columns:
        sub = df.dropna(subset=["sim_time_client", "experienced_latency_ms"])
        (line,) = ax.plot(
            sub["sim_time_client"],
            sub["experienced_latency_ms"],
            color=CB_BLACK,
            lw=1.6,
        )
        h.append(line)
        l.append("Avg. Chosen Latency")
        if has_std and "experienced_latency_ms_std_agg" in df.columns:
            std = df.loc[sub.index, "experienced_latency_ms_std_agg"]
            _shade(
                ax, sub["sim_time_client"], sub["experienced_latency_ms"], std, CB_BLACK
            )
    oracle_col = _resolve_oracle_column(df)
    if oracle_col is not None:
        sub = df.dropna(subset=["sim_time_client", oracle_col])
        (line,) = ax.plot(
            sub["sim_time_client"],
            sub[oracle_col],
            color=CB_ORANGE,
            lw=2.0,
            ls=(0, (2, 2)),
            zorder=20,
        )
        h.append(line)
        l.append("Avg. Oracle Optimal")
        std_candidates = [
            f"{oracle_col}_std_agg",
            "dynamic_best_server_latency_std_agg",
            "experienced_latency_ms_ORACLE_std_agg",
        ]
        std_col = next((c for c in std_candidates if c in df.columns), None)
        if has_std and std_col is not None:
            std = df.loc[sub.index, std_col]
            _shade(
                ax,
                sub["sim_time_client"],
                sub[oracle_col],
                std,
                CB_ORANGE,
            )
    if h:
        format_axes(
            ax,
            f"Avg. Chosen vs. Optimal Latency — {strat_display}",
            "Simulation Time (s)",
            "Latency (ms)",
            custom_handles=h,
            custom_labels=l,
            legend_loc="upper right",
            xlim_max=xlim,
        )
        fig.tight_layout()
        save_figure(fig, os.path.join(img_dir, "1_avg_latency_chosen_vs_optimal"))
    else:
        plt.close(fig)
    value_cols = sorted(
        c for c in df.columns if c.startswith("value_") and not c.endswith("_std_agg")
    )
    if value_cols:
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        ylabel = (
            "Avg. Estimated Reward"
            if "ucb" in strat_key.lower()
            else "Avg. Estimated Value"
        )
        if strat_key == "epsilon_greedy":
            ylabel = "Avg. Estimated Reward"
        for col in value_cols:
            sk = col.replace("value_", "").replace("_", "-")
            sub = df.dropna(subset=["sim_time_client", col])
            if not sub.empty:
                ax.plot(
                    sub["sim_time_client"],
                    sub[col],
                    color=get_server_color(sk),
                    lw=1.2,
                    alpha=0.8,
                    label=get_server_label(sk),
                )
                std_col = col + "_std_agg"
                if std_col in df.columns:
                    _shade(
                        ax,
                        sub["sim_time_client"],
                        sub[col],
                        df.loc[sub.index, std_col],
                        get_server_color(sk),
                    )
        format_axes(
            ax,
            f"Avg. RL Estimated Values — {strat_display}",
            "Simulation Time (s)",
            ylabel,
            legend_loc="upper right",
            xlim_max=xlim,
        )
        fig.tight_layout()
        save_figure(fig, os.path.join(img_dir, "2_avg_rl_estimated_values"))
    actual_cnt = sorted(
        c
        for c in df.columns
        if c.startswith("actual_count_") and not c.endswith("_std_agg")
    )
    cnt_cols = sorted(
        c for c in df.columns if c.startswith("count_") and not c.endswith("_std_agg")
    )
    cols3, prefix3, ylabel3 = [], "", "Avg. Selections (Pulls)"
    if cnt_cols:
        cols3, prefix3 = cnt_cols, "count_"
    if cols3:
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        for col in cols3:
            sk = col.replace(prefix3, "").replace("_", "-")
            sub = df.dropna(subset=["sim_time_client", col])
            if not sub.empty:
                ax.plot(
                    sub["sim_time_client"],
                    sub[col],
                    color=get_server_color(sk),
                    lw=1.2,
                    alpha=0.8,
                    label=get_server_label(sk),
                )
                std_col = col + "_std_agg"
                if std_col in df.columns:
                    _shade(
                        ax,
                        sub["sim_time_client"],
                        sub[col],
                        df.loc[sub.index, std_col],
                        get_server_color(sk),
                    )
        format_axes(
            ax,
            f"Avg. Selection Counts — {strat_display}",
            "Simulation Time (s)",
            ylabel3,
            legend_loc="upper left",
            xlim_max=xlim,
        )
        fig.tight_layout()
        save_figure(fig, os.path.join(img_dir, "3_avg_rl_selection_counts"))
    if (
        "all_servers_oracle_latency_json" in df.columns
        and not df["all_servers_oracle_latency_json"].dropna().empty
    ):
        parsed = parse_json_column(df["all_servers_oracle_latency_json"])
        if not parsed.empty:
            merged = pd.concat(
                [df.loc[parsed.index, "sim_time_client"], parsed], axis=1
            )
            merged = merged.drop_duplicates("sim_time_client", keep="last")
            fig, ax = plt.subplots(figsize=(7.0, 3.5))
            lat_cols = sorted(
                c for c in merged.columns if c in KNOWN_SERVER_KEYS_UNDERSCORE
            )
            for col in lat_cols:
                sk = col.replace("_", "-")
                sub = merged.dropna(subset=["sim_time_client", col])
                if not sub.empty:
                    ax.plot(
                        sub["sim_time_client"],
                        sub[col],
                        color=get_server_color(sk),
                        lw=1.0,
                        alpha=0.6,
                        label=get_server_label(sk),
                    )
            format_axes(
                ax,
                f"Avg. Oracle Latency Landscape — {strat_display}",
                "Simulation Time (s)",
                "Simulated Latency (ms)",
                legend_loc="upper right",
                xlim_max=xlim,
            )
            fig.tight_layout()
            save_figure(fig, os.path.join(img_dir, "4_avg_oracle_latency_landscape"))
    logger.info(f"Aggregated graphs for '{fname}' saved to {img_dir}")


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Generate publication-ready graphs from AGGREGATED CSV logs."
    )
    parser.add_argument(
        "csv_filename", nargs="?", default=None, help="Aggregated CSV path/filename."
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=None,
        help="Optional max simulation time (seconds). If omitted, uses data max time.",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Base output directory for generated figures.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    apply_global_style()
    configure_logger(logger, args.verbose)
    if args.csv_filename:
        path = args.csv_filename
        if not os.path.isabs(path) and not os.path.exists(path):
            candidate = os.path.join(PROCESSED_DIR, os.path.basename(path))
            if os.path.exists(candidate):
                path = candidate
        if os.path.exists(path):
            generate_plots_for_aggregated(path, max_time=args.max_time)
        else:
            logger.error(f"File not found: {args.csv_filename}")
    else:
        logger.info("No CSV specified — processing all aggregated logs in processed/")
        if os.path.isdir(PROCESSED_DIR):
            for fn in sorted(os.listdir(PROCESSED_DIR)):
                if fn.startswith("log_") and fn.endswith("_average.csv"):
                    generate_plots_for_aggregated(
                        os.path.join(PROCESSED_DIR, fn), max_time=args.max_time
                    )


if __name__ == "__main__":
    main()
