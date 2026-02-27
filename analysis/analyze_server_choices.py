import os
import re
import json
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))
from plot_utils import (
    apply_global_style,
    configure_logger,
    save_figure,
    get_strategy_display_name,
    extract_strategy_from_filename,
    CB_BLACK,
    CB_BLUE,
    CB_GREY,
    STRATEGY_LEGEND_ORDER,
)

logger = logging.getLogger("analyze_server_choices")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "logs", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "analysis")
ACTUAL_CACHE_NAMES_HYPHEN = [
    "video-streaming-cache-1",
    "video-streaming-cache-2",
    "video-streaming-cache-3",
]
HEADER_BG = "#0072B2"
HEADER_FG = "#FFFFFF"
ROW_EVEN_BG = "#F0F4F8"
ROW_ODD_BG = "#FFFFFF"
CELL_EDGE = "#CCCCCC"


def _extract_strategy(fname_no_ext: str) -> str:
    return extract_strategy_from_filename(fname_no_ext)


def analyze_server_choices(logs_dir, output_csv=None, output_img=None):
    if not os.path.isdir(logs_dir):
        logger.error(f"Directory not found: {logs_dir}")
        return
    results = {}
    for fn in sorted(os.listdir(logs_dir)):
        if not (fn.startswith("log_") and fn.endswith("_average.csv")):
            continue
        path = os.path.join(logs_dir, fn)
        try:
            df = pd.read_csv(path)
            required = [
                "steering_decision_main_server",
                "all_servers_oracle_latency_json",
            ]
            if df.empty or not all(c in df.columns for c in required):
                continue
            sk = _extract_strategy(os.path.splitext(fn)[0])
            if sk == "Unknown":
                continue
            total, correct = 0, 0
            for _, row in df.iterrows():
                dec = row["steering_decision_main_server"]
                raw = row["all_servers_oracle_latency_json"]
                if pd.isna(dec) or pd.isna(raw) or "N/A" in str(dec):
                    continue
                total += 1
                try:
                    lats = json.loads(raw)
                    valid = {
                        k.replace("_", "-"): v
                        for k, v in lats.items()
                        if k.replace("_", "-") in ACTUAL_CACHE_NAMES_HYPHEN
                        and isinstance(v, (int, float))
                    }
                    if not valid:
                        continue
                    if dec == min(valid, key=valid.get):
                        correct += 1
                except (json.JSONDecodeError, TypeError):
                    continue
            if total > 0:
                results.setdefault(sk, {"total": 0, "correct": 0})
                results[sk]["total"] += total
                results[sk]["correct"] += correct
        except Exception as exc:
            logger.error(f"Error in {fn}: {exc}")
    if not results:
        logger.warning("No results — nothing to output.")
        return
    rows = []
    ordered = [k for k in STRATEGY_LEGEND_ORDER if k in results]
    ordered += sorted(k for k in results if k not in ordered)
    for sk in ordered:
        d = results[sk]
        acc = d["correct"] / d["total"] * 100 if d["total"] else 0
        rows.append(
            {
                "Strategy": get_strategy_display_name(sk),
                "Correct (#)": d["correct"],
                "Total (#)": d["total"],
                "Accuracy (%)": f"{acc:.1f}%",
            }
        )
    df_res = pd.DataFrame(rows)
    logger.info("\n" + df_res.to_string(index=False))
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_res.to_csv(output_csv, index=False)
        logger.info(f"CSV saved: {output_csv}")
    if output_img:
        _save_table_image(
            df_res, output_img, title="Dynamic Best Server Choice Accuracy"
        )


def _save_table_image(df, path_without_ext, title):
    ncols, nrows = len(df.columns), len(df)
    fig_w = max(6.5, ncols * 2.0)
    fig_h = max(2.5, nrows * 0.45 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(CELL_EDGE)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(weight="bold", color=HEADER_FG)
            cell.set_facecolor(HEADER_BG)
        else:
            cell.set_facecolor(ROW_EVEN_BG if r % 2 == 0 else ROW_ODD_BG)
    ax.set_title(title, fontsize=13, pad=12, weight="bold")
    fig.tight_layout()
    base = os.path.splitext(path_without_ext)[0]
    save_figure(fig, base)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse dynamic-best-server choice accuracy."
    )
    parser.add_argument(
        "--logs_dir",
        default=PROCESSED_DIR,
        help="Directory containing *_average.csv files to analyse.",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join(PROCESSED_DIR, "dynamic_best_choice_accuracy.csv"),
    )
    parser.add_argument(
        "--output_img",
        default=os.path.join(OUTPUT_DIR, "dynamic_best_choice_accuracy_table.png"),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    apply_global_style()
    configure_logger(logger, args.verbose)
    analyze_server_choices(args.logs_dir, args.output_csv, args.output_img)


if __name__ == "__main__":
    main()
