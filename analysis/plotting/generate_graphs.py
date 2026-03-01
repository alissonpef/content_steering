import os
import re
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    CB_RED,
    CB_CYAN,
    KNOWN_SERVER_KEYS_UNDERSCORE,
)

logger = logging.getLogger("generate_graphs")
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
RAW_LOGS_DIR = os.path.join(PROJECT_ROOT, "logs", "raw_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "individual_runs")
ACTUAL_CACHE_NAMES_HYPHEN = [
    "video-streaming-cache-1",
    "video-streaming-cache-2",
    "video-streaming-cache-3",
]
MAX_PLOT_TIME_SECONDS = 300


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
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=indices)


def find_dynamic_best(row):
    raw = row.get("all_servers_oracle_latency_json")
    if pd.isna(raw):
        return None, np.nan
    try:
        lats = json.loads(raw)
        valid = {
            k.replace("_", "-"): v
            for k, v in lats.items()
            if k.replace("_", "-") in ACTUAL_CACHE_NAMES_HYPHEN
            and isinstance(v, (int, float))
        }
        if not valid:
            return None, np.nan
        best = min(valid, key=valid.get)
        return best, valid[best]
    except Exception:
        return None, np.nan


def generate_plots(csv_path: str, max_time: float = None):
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return
    fname = os.path.splitext(os.path.basename(csv_path))[0]
    img_dir = os.path.join(OUTPUT_DIR, fname)
    os.makedirs(img_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning(f"Empty CSV: {csv_path}")
        return
    df.sort_values("sim_time_client", inplace=True)
    df.reset_index(drop=True, inplace=True)
    if max_time is not None:
        df = df[df["sim_time_client"] <= max_time].copy()
        if df.empty:
            logger.warning(f"No data ≤ {max_time}s in {csv_path}")
            return
    xlim = float(df["sim_time_client"].max()) if "sim_time_client" in df.columns else None
    strat_key = "N/A"
    if "rl_strategy" in df.columns and not df["rl_strategy"].dropna().empty:
        strat_key = df["rl_strategy"].dropna().iloc[0]
    else:
        strat_key = extract_strategy_from_filename(fname)
    if strat_key not in KNOWN_STRATEGY_KEYS:
        logger.info(f"Skipping unsupported/legacy strategy file: {fname}")
        return
    strat_display = get_strategy_display_name(strat_key)
    if "all_servers_oracle_latency_json" in df.columns:
        best_info = df.apply(find_dynamic_best, axis=1, result_type="expand")
        df[["dynamic_best_server_name", "dynamic_best_server_latency"]] = best_info
    else:
        df["dynamic_best_server_name"] = None
        df["dynamic_best_server_latency"] = np.nan
    window = 10
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    handles, labels = [], []
    if "experienced_latency_ms" in df.columns:
        tmp = df.dropna(subset=["sim_time_client", "experienced_latency_ms"])
        if len(tmp) >= window:
            ma = (
                tmp["experienced_latency_ms"]
                .rolling(window, center=True, min_periods=1)
                .mean()
            )
            (h,) = ax.plot(tmp["sim_time_client"], ma, color=CB_BLACK, lw=1.6)
            handles.append(h)
            labels.append(f"MA({window}s) — Chosen")
    if "dynamic_best_server_latency" in df.columns:
        tmp = df.dropna(subset=["sim_time_client", "dynamic_best_server_latency"])
        if len(tmp) >= window:
            ma = (
                tmp["dynamic_best_server_latency"]
                .rolling(window, center=True, min_periods=1)
                .mean()
            )
            (h,) = ax.plot(tmp["sim_time_client"], ma, color=CB_RED, lw=1.4, ls="--")
            handles.append(h)
            labels.append(f"MA({window}s) — Oracle Optimal")
    if handles:
        format_axes(
            ax,
            f"Chosen vs. Optimal Latency — {strat_display}",
            "Simulation Time (s)",
            "Latency (ms)",
            custom_handles=handles,
            custom_labels=labels,
            legend_loc="upper right",
            xlim_max=xlim,
        )
        fig.tight_layout()
        save_figure(fig, os.path.join(img_dir, "1_latency_chosen_vs_optimal"))
    else:
        plt.close(fig)
    if "steering_decision_main_server" in df.columns:
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        h2, l2 = [], []
        tmp = df.dropna(
            subset=["steering_decision_main_server", "sim_time_client"]
        ).copy()
        tmp = tmp.drop_duplicates(subset=["sim_time_client"], keep="first")
        all_entities = sorted(
            set(ACTUAL_CACHE_NAMES_HYPHEN)
            | set(tmp["steering_decision_main_server"].unique())
        )
        ent_map = {e: i for i, e in enumerate(all_entities)}
        tmp["decision_int"] = tmp["steering_decision_main_server"].map(ent_map)
        tmp2 = tmp.dropna(subset=["decision_int"])
        if not tmp2.empty:
            (h,) = ax.plot(
                tmp2["sim_time_client"],
                tmp2["decision_int"],
                drawstyle="steps-post",
                color=CB_CYAN,
                lw=1.4,
                marker="o",
                ms=2,
            )
            h2.append(h)
            l2.append("Algorithm Choice")
        if "dynamic_best_server_name" in tmp.columns:
            tmp["best_int"] = tmp["dynamic_best_server_name"].map(ent_map)
            tmp3 = tmp.dropna(subset=["best_int"])
            if not tmp3.empty:
                ax.scatter(
                    tmp3["sim_time_client"],
                    tmp3["best_int"],
                    marker="x",
                    s=20,
                    color=CB_RED,
                    zorder=5,
                    alpha=0.7,
                )
                h2.append(Line2D([], [], ls="None", marker="x", color=CB_RED, ms=5))
                l2.append("Oracle Optimal")
        if ent_map:
            ax.set_yticks(list(ent_map.values()))
            ax.set_yticklabels([get_server_label(e) for e in all_entities])
            ax.set_ylim(min(ent_map.values()) - 0.5, max(ent_map.values()) + 0.5)
        format_axes(
            ax,
            f"Steering Decisions — {strat_display}",
            "Simulation Time (s)",
            "Server",
            custom_handles=h2,
            custom_labels=l2,
            legend_loc="upper right",
            xlim_max=xlim,
        )
        fig.tight_layout()
        save_figure(fig, os.path.join(img_dir, "2_steering_decisions"))
    else:
        logger.info("Plot 2 skipped: no steering decision column.")
    if "rl_values_json" in df.columns and not df["rl_values_json"].dropna().empty:
        parsed = parse_json_column(df["rl_values_json"], prefix="value_")
        if not parsed.empty:
            merged = pd.concat(
                [df.loc[parsed.index, "sim_time_client"], parsed], axis=1
            )
            merged = merged.drop_duplicates("sim_time_client", keep="last")
            fig, ax = plt.subplots(figsize=(7.0, 3.5))
            value_cols = sorted(
                c
                for c in merged.columns
                if c.startswith("value_")
                and c.replace("value_", "") in KNOWN_SERVER_KEYS_UNDERSCORE
            )
            for col in value_cols:
                sk = col.replace("value_", "").replace("_", "-")
                sub = merged.dropna(subset=["sim_time_client", col])
                if not sub.empty:
                    ax.plot(
                        sub["sim_time_client"],
                        sub[col],
                        color=get_server_color(sk),
                        lw=1.2,
                        alpha=0.8,
                        label=get_server_label(sk),
                    )
            ylabel = (
                "Estimated Reward" if "ucb" in strat_key.lower() else "Estimated Value"
            )
            if strat_key == "epsilon_greedy":
                ylabel = "Estimated Reward"
            format_axes(
                ax,
                f"RL Estimated Values — {strat_display}",
                "Simulation Time (s)",
                ylabel,
                legend_loc="upper right",
                xlim_max=xlim,
            )
            fig.tight_layout()
            save_figure(fig, os.path.join(img_dir, "3_rl_estimated_values"))
        else:
            logger.info("Plot 3 skipped: no parseable RL values.")
    counts_col = None
    ylabel4 = "Selections (Pulls)"
    if "rl_counts_json" in df.columns:
        counts_col = "rl_counts_json"
    if counts_col and not df[counts_col].dropna().empty:
        parsed = parse_json_column(df[counts_col], prefix="data_")
        if not parsed.empty:
            merged = pd.concat(
                [df.loc[parsed.index, "sim_time_client"], parsed], axis=1
            )
            merged = merged.drop_duplicates("sim_time_client", keep="last")
            fig, ax = plt.subplots(figsize=(7.0, 3.5))
            cnt_cols = sorted(
                c
                for c in merged.columns
                if c.startswith("data_")
                and c.replace("data_", "") in KNOWN_SERVER_KEYS_UNDERSCORE
            )
            for col in cnt_cols:
                sk = col.replace("data_", "").replace("_", "-")
                sub = merged.dropna(subset=["sim_time_client", col])
                if not sub.empty:
                    ax.plot(
                        sub["sim_time_client"],
                        sub[col],
                        color=get_server_color(sk),
                        lw=1.2,
                        alpha=0.8,
                        label=get_server_label(sk),
                    )
            format_axes(
                ax,
                f"Selection Counts — {strat_display}",
                "Simulation Time (s)",
                ylabel4,
                legend_loc="upper left",
                xlim_max=xlim,
            )
            fig.tight_layout()
            save_figure(fig, os.path.join(img_dir, "4_rl_selection_counts"))
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
                f"Oracle Latency Landscape — {strat_display}",
                "Simulation Time (s)",
                "Simulated Latency (ms)",
                legend_loc="upper right",
                xlim_max=xlim,
            )
            fig.tight_layout()
            save_figure(fig, os.path.join(img_dir, "5_oracle_latency_landscape"))
    logger.info(f"Graphs for '{fname}' saved to {img_dir}")


def _resolve_csv(name: str):
    candidates = [name]
    for d in [os.getcwd(), RAW_LOGS_DIR]:
        candidates.append(os.path.join(d, os.path.basename(name)))
        if not name.endswith(".csv"):
            candidates.append(os.path.join(d, os.path.basename(name) + ".csv"))
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Generate per-run publication-ready graphs."
    )
    parser.add_argument(
        "csv_argument",
        nargs="?",
        default=None,
        help="CSV path or filename (searched in standard dirs).",
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
    if args.csv_argument:
        path = _resolve_csv(args.csv_argument)
        if path:
            generate_plots(path, max_time=args.max_time)
        else:
            logger.error(f"File not found: {args.csv_argument}")
    else:
        logger.info("No CSV specified — processing all raw logs.")
        if os.path.isdir(RAW_LOGS_DIR):
            for fn in sorted(os.listdir(RAW_LOGS_DIR)):
                if (
                    fn.startswith("log_")
                    and fn.endswith(".csv")
                    and "_average" not in fn
                ):
                    generate_plots(os.path.join(RAW_LOGS_DIR, fn), max_time=args.max_time)


if __name__ == "__main__":
    main()
