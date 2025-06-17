import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import argparse
import logging
import numpy as np

logger = logging.getLogger("generate_boxplots")

BASE_GRAPHICS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AVERAGE_LOGS_DIR = os.path.join(BASE_GRAPHICS_DIR, "Logs", "Average")
DEFAULT_IMG_OUTPUT_DIR = os.path.join(BASE_GRAPHICS_DIR, "Img", "boxplots")

STRATEGY_STYLES = {
    "ucb1": {"label": "UCB1", "color": "tab:blue"},
    "epsilon_greedy": {"label": "Epsilon Greedy", "color": "tab:green"},
    "random": {"label": "Random", "color": "tab:red"},
    "oracle_best_choice": {"label": "Optimal Strategy", "color": "tab:purple"},
    "no_steering": {"label": "No Steering", "color": "tab:brown"},
    "d_ucb": {"label": "D-UCB", "color": "tab:cyan"},
    "default": {"label": "Unknown", "color": "tab:grey"}
}
KNOWN_STRATEGY_KEYS = list(STRATEGY_STYLES.keys())
if "default" in KNOWN_STRATEGY_KEYS:
    KNOWN_STRATEGY_KEYS.remove("default")

def extract_strategy_name_from_filename(filename_no_ext: str) -> str:
    for known_key in KNOWN_STRATEGY_KEYS:
        if f"log_{known_key}_average" in filename_no_ext:
            if filename_no_ext == f"log_{known_key}_average" or \
               filename_no_ext.startswith(f"log_{known_key}_average_"):
                is_prefix_of_another = False
                if known_key == "ucb1" and "d_ucb" in filename_no_ext and "log_d_ucb" in filename_no_ext:
                    is_prefix_of_another = True
                if not is_prefix_of_another:
                    return known_key

    match_generic = re.match(r"log_([a-zA-Z0-9_]+?)_average", filename_no_ext)
    if match_generic:
        potential_strategy = match_generic.group(1)
        for known_key in KNOWN_STRATEGY_KEYS:
            if potential_strategy == known_key:
                return known_key
        return potential_strategy
    logger.warning(f"Could not extract strategy name from '{filename_no_ext}'. Returning 'Unknown'.")
    return "Unknown"


def generate_individual_boxplot(df: pd.DataFrame, strategy_name_key: str, metric_column: str, output_dir: str):
    if df.empty or metric_column not in df.columns or df[metric_column].dropna().empty:
        logger.warning(f"No data or metric '{metric_column}' for strategy '{strategy_name_key}' to generate individual boxplot.")
        return

    strategy_style = STRATEGY_STYLES.get(strategy_name_key, STRATEGY_STYLES["default"])
    label = strategy_style.get("label", strategy_name_key.title())
    color = strategy_style.get("color", "lightgray")

    plt.figure(figsize=(6, 8))
    bp = plt.boxplot(df[metric_column].dropna(), vert=True, patch_artist=True, labels=[label])
    
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')


    plot_title = f"Latency Distribution: {label}"
    plt.title(plot_title, fontsize=14)
    plt.ylabel(f"{metric_column.replace('_', ' ').title()} (ms)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7, axis='y')
    plt.tight_layout()

    filename = f"boxplot_individual_{strategy_name_key}_{metric_column}.png"
    plot_path = os.path.join(output_dir, filename)
    try:
        plt.savefig(plot_path, dpi=200)
        logger.info(f"Individual boxplot saved: {plot_path}")
    except Exception as e:
        logger.error(f"Error saving individual boxplot {filename}: {e}")
    finally:
        plt.close()


def generate_comparison_boxplot(all_strategy_data: dict, metric_column: str, output_dir: str):
    plot_data = []
    plot_labels = []
    box_colors_list = []

    sorted_strategy_keys = sorted(all_strategy_data.keys(), key=lambda k: STRATEGY_STYLES.get(k, {}).get("label", k.title()))

    for strategy_key in sorted_strategy_keys:
        df = all_strategy_data[strategy_key]
        if not df.empty and metric_column in df.columns and not df[metric_column].dropna().empty:
            plot_data.append(df[metric_column].dropna().tolist())
            strategy_style = STRATEGY_STYLES.get(strategy_key, STRATEGY_STYLES["default"])
            plot_labels.append(strategy_style.get("label", strategy_key.title()))
            box_colors_list.append(strategy_style.get("color", "lightgray"))
        else:
            logger.warning(f"No data or metric '{metric_column}' for strategy '{strategy_key}' in comparison boxplot.")

    if not plot_data:
        logger.warning(f"No data to generate comparison boxplot for metric '{metric_column}'.")
        return

    plt.figure(figsize=(max(10, len(plot_labels) * 1.8), 8))
    bp = plt.boxplot(plot_data, vert=True, patch_artist=True, labels=plot_labels)

    for patch, color in zip(bp['boxes'], box_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')


    plt.title(f"Comparison of {metric_column.replace('_', ' ').title()} Distribution by Strategy", fontsize=16)
    plt.ylabel(f"{metric_column.replace('_', ' ').title()} (ms)", fontsize=14)
    plt.xlabel("Strategy", fontsize=14)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7, axis='y')
    plt.tight_layout()

    filename = f"boxplot_comparison_all_strategies_{metric_column}.png"
    plot_path = os.path.join(output_dir, filename)
    try:
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Comparison boxplot saved: {plot_path}")
    except Exception as e:
        logger.error(f"Error saving comparison boxplot {filename}: {e}")
    finally:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate individual and comparison boxplots from aggregated simulation logs.")
    parser.add_argument("--agg_dir", type=str, default=DEFAULT_AVERAGE_LOGS_DIR,
                        help=f"Directory with aggregated CSV files. Default: {DEFAULT_AVERAGE_LOGS_DIR}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_IMG_OUTPUT_DIR,
                        help=f"Directory to save the boxplot images. Default: {DEFAULT_IMG_OUTPUT_DIR}")
    parser.add_argument("--metric", type=str, default="experienced_latency_ms",
                        choices=["experienced_latency_ms", "experienced_latency_ms_CLIENT", "experienced_latency_ms_ORACLE", "dynamic_best_server_latency"],
                        help="Metric to plot for boxplots. Default: experienced_latency_ms")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    _handler_boxplot = logging.StreamHandler()
    log_level_to_set = logging.DEBUG if args.verbose else logging.INFO
    _formatter_boxplot = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    _handler_boxplot.setFormatter(_formatter_boxplot)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(_handler_boxplot)
    logger.setLevel(log_level_to_set)

    logger.info(f"Logging level set to {logging.getLevelName(logger.getEffectiveLevel())}.")
    os.makedirs(args.output_dir, exist_ok=True)

    all_strategy_data_for_comparison = {}

    if not os.path.isdir(args.agg_dir):
        logger.error(f"Aggregated logs directory not found: {args.agg_dir}")
        return

    for filename in sorted(os.listdir(args.agg_dir)):
        if filename.startswith("log_") and filename.endswith("_average.csv"):
            file_path = os.path.join(args.agg_dir, filename)
            logger.info(f"Processing file for boxplots: {filename}")
            try:
                df_agg = pd.read_csv(file_path)
                if df_agg.empty:
                    logger.warning(f"File {filename} is empty. Skipping.")
                    continue

                filename_no_ext = os.path.splitext(filename)[0]
                strategy_key = extract_strategy_name_from_filename(filename_no_ext)

                if strategy_key == "Unknown":
                    logger.warning(f"Could not determine strategy for {filename}. Skipping individual boxplot.")
                else:
                    generate_individual_boxplot(df_agg, strategy_key, args.metric, args.output_dir)
                    all_strategy_data_for_comparison[strategy_key] = df_agg.copy()

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}", exc_info=True)

    if all_strategy_data_for_comparison:
        generate_comparison_boxplot(all_strategy_data_for_comparison, args.metric, args.output_dir)
    else:
        logger.warning("No data collected from any strategy files to generate a comparison boxplot.")

    logger.info("Boxplot generation process finished.")

if __name__ == "__main__":
    main()