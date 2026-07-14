import argparse
import json
import logging
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))
from plot_utils import (
    STRATEGY_LEGEND_ORDER,
    apply_global_style,
    configure_logger,
    extract_strategy_from_filename,
    get_strategy_display_name,
    save_figure,
)

logger = logging.getLogger("analyze_server_choices")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "logs", "aggregated")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "results", "analysis")
ACTUAL_CACHE_NAMES_HYPHEN = [
    "delivery-node-1",
    "delivery-node-2",
    "delivery-node-3",
]
HEADER_BG = "#4A4A4A"
HEADER_FG = "#FFFFFF"
ROW_EVEN_BG = "#F0F4F8"
ROW_ODD_BG = "#FFFFFF"
CELL_EDGE = "#CCCCCC"


def _extract_scenario(fname: str) -> str:
    m = re.search(r"_scenario\d+_([a-zA-Z0-9_]+)_average\.csv$", fname)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"_(baseline|mobility|spam)_average\.csv$", fname)
    if m2:
        return m2.group(1).lower()
    return "all"


def _build_accuracy_dataframe(results: dict) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def _extract_strategy(fname_no_ext: str) -> str:
    return extract_strategy_from_filename(fname_no_ext)


def analyze_server_choices(
    logs_dir,
    output_csv=None,
    output_img=None,
    comparison_output_dir=None,
):
    if not os.path.isdir(logs_dir):
        logger.error(f"Directory not found: {logs_dir}")
        return
    results = {}
    scenario_results = {}
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
                    lats = json.loads(str(raw))
                    valid = {
                        k.replace("_", "-"): v
                        for k, v in lats.items()
                        if k.replace("_", "-") in ACTUAL_CACHE_NAMES_HYPHEN
                        and isinstance(v, (int, float))
                    }
                    if not valid:
                        continue
                    if dec == min(valid, key=lambda k: float(valid[k])):
                        correct += 1
                except json.JSONDecodeError, TypeError:
                    continue
            if total > 0:
                results.setdefault(sk, {"total": 0, "correct": 0})
                results[sk]["total"] += total
                results[sk]["correct"] += correct

                scenario_key = _extract_scenario(fn)
                scenario_results.setdefault(scenario_key, {})
                scenario_results[scenario_key].setdefault(sk, {"total": 0, "correct": 0})
                scenario_results[scenario_key][sk]["total"] += total
                scenario_results[scenario_key][sk]["correct"] += correct
        except Exception as exc:
            logger.error(f"Error in {fn}: {exc}")
    if not results:
        logger.warning("No results — nothing to output.")
        return
    df_res = _build_accuracy_dataframe(results)
    logger.info("\n" + df_res.to_string(index=False))
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_res.to_csv(output_csv, index=False)
        logger.info(f"CSV saved: {output_csv}")
    if output_img:
        _save_table_image(df_res, output_img, title="Dynamic Best Server Choice Accuracy")

    if output_csv:
        by_scenario_dir = os.path.join(os.path.dirname(output_csv), "by_scenario")
        os.makedirs(by_scenario_dir, exist_ok=True)
        for scenario_key, scenario_data in sorted(scenario_results.items()):
            sdf = _build_accuracy_dataframe(scenario_data)
            s_csv = os.path.join(
                by_scenario_dir,
                f"dynamic_best_choice_accuracy_{scenario_key}.csv",
            )
            sdf.to_csv(s_csv, index=False)
            logger.info(f"Scenario CSV saved: {s_csv}")
            if output_img:
                by_scenario_img_dir = os.path.join(os.path.dirname(output_img), "by_scenario")
                os.makedirs(by_scenario_img_dir, exist_ok=True)
                s_img = os.path.join(
                    by_scenario_img_dir,
                    f"dynamic_best_choice_accuracy_{scenario_key}_table.png",
                )
                _save_table_image(
                    sdf,
                    s_img,
                    title=f"Dynamic Best Server Choice Accuracy ({scenario_key.title()})",
                )

    if comparison_output_dir:
        for scenario_key, scenario_data in sorted(scenario_results.items()):
            sdf = _build_accuracy_dataframe(scenario_data)
            scenario_dir = os.path.join(comparison_output_dir, scenario_key)
            os.makedirs(scenario_dir, exist_ok=True)
            scenario_csv = os.path.join(
                scenario_dir,
                f"algorithms_accuracy_comparison_{scenario_key}.csv",
            )
            sdf.to_csv(scenario_csv, index=False)
            logger.info(f"Scenario comparison CSV saved: {scenario_csv}")
            scenario_img = os.path.join(
                scenario_dir,
                f"algorithms_accuracy_comparison_{scenario_key}_table.png",
            )
            _save_table_image(
                sdf,
                scenario_img,
                title=f"Algorithm Comparison ({scenario_key.title()})",
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
    for (r, _c), cell in tbl.get_celld().items():
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
    parser = argparse.ArgumentParser(description="Analyse dynamic-best-server choice accuracy.")
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
    parser.add_argument(
        "--comparison_output_dir",
        default=None,
        help="If provided, saves per-scenario algorithm comparison tables under <dir>/<scenario>/.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    apply_global_style()
    configure_logger(logger, args.verbose)
    analyze_server_choices(
        args.logs_dir,
        args.output_csv,
        args.output_img,
        args.comparison_output_dir,
    )


if __name__ == "__main__":
    main()
