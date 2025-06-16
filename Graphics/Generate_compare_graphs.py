import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import re
import argparse
import logging
import numpy as np
import math

logger = logging.getLogger("compare_strategies")

BASE_GRAPHICS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AVERAGE_LOGS_DIR = os.path.join(BASE_GRAPHICS_DIR, "Logs", "Average")
DEFAULT_IMG_DIR = os.path.join(BASE_GRAPHICS_DIR, "Img")
COMPARISON_X_AXIS_LIMIT = 150

STRATEGY_STYLES = {
    "ucb1": {"color": "tab:blue", "label": "UCB1"},
    "epsilon_greedy": {"color": "tab:green", "label": "Epsilon Greedy"},
    "random": {"color": "tab:red", "label": "Random"},
    "oracle_best_choice": {"color": "tab:purple", "label": "Optimal Strategy"},
    "no_steering": {"color": "tab:brown", "label": "No Steering"},
    "d_ucb": {"color": "tab:cyan", "label": "D-UCB"},
    "default": {"color": "tab:grey", "label": "Unknown"}
}
KNOWN_STRATEGY_KEYS = list(STRATEGY_STYLES.keys())
if "default" in KNOWN_STRATEGY_KEYS:
    KNOWN_STRATEGY_KEYS.remove("default")

def extract_strategy_name(filename_no_ext: str, df_column_data: pd.Series = None) -> str:
    if df_column_data is not None and not df_column_data.empty:
        first_valid_strategy = df_column_data.dropna().iloc[0] if not df_column_data.dropna().empty else None
        if first_valid_strategy and isinstance(first_valid_strategy, str):
            normalized_strategy = first_valid_strategy.lower().replace(" ", "_").replace("-", "_")
            for known_key in KNOWN_STRATEGY_KEYS:
                if normalized_strategy == known_key:
                    return known_key
            if normalized_strategy != "n/a_(aggregated)" and normalized_strategy != "n/a":
                 logger.debug(f"Strategy '{first_valid_strategy}' (normalized to '{normalized_strategy}') is not an exact known key, returning as is.")
                 return normalized_strategy
    for known_key in KNOWN_STRATEGY_KEYS:
        if f"log_{known_key}" in filename_no_ext:
            if filename_no_ext.startswith(f"log_{known_key}_") or filename_no_ext == f"log_{known_key}":
                is_prefix_of_another = False
                if known_key == "ucb1" and "d_ucb" in filename_no_ext:
                    is_prefix_of_another = True
                if not is_prefix_of_another:
                    return known_key
    match_generic = re.match(r"log_([a-zA-Z0-9_]+?)_average", filename_no_ext)
    if match_generic:
        potential_strategy = match_generic.group(1)
        for known_key in KNOWN_STRATEGY_KEYS:
            if potential_strategy == known_key:
                return known_key
        logger.debug(f"Strategy extracted from filename as '{potential_strategy}' (generic).")
        return potential_strategy
    if filename_no_ext.startswith("log_") and "_average" in filename_no_ext:
        temp_name = filename_no_ext.split("log_", 1)[1]
        temp_name = temp_name.split("_average", 1)[0]
        temp_name = re.sub(r'_gamma[\d_p]+$', '', temp_name)
        temp_name = re.sub(r'_\d+$', '', temp_name)
        for known_key in KNOWN_STRATEGY_KEYS:
            if temp_name == known_key:
                return known_key
        logger.debug(f"Strategy extracted from filename as '{temp_name}' (final fallback).")
        return temp_name
    logger.warning(f"Could not extract strategy name from '{filename_no_ext}'. Returning 'Unknown'.")
    return "Unknown"

def format_comparison_plot(ax, title, xlabel, ylabel, legend_loc='best',
                           custom_legend_handles=None, custom_legend_labels=None,
                           xlim_max=None):
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    if xlim_max is not None and ax.has_data():
        ax.set_xticks(np.arange(0, xlim_max + 1, 15))
        ax.set_xlim(left=0, right=xlim_max)
    elif ax.has_data():
        current_xlim_left, current_xlim_right = ax.get_xlim()
        start_tick = 0
        if current_xlim_left > 7.5 :
             start_tick = math.floor(current_xlim_left / 15) * 15
        ax.set_xticks(np.arange(start_tick, current_xlim_right + 1, 15))

    if custom_legend_handles and custom_legend_labels:
        unique_entries = {}
        final_handles, final_labels = [], []
        for handle, label in zip(custom_legend_handles, custom_legend_labels):
            if label not in unique_entries:
                unique_entries[label] = handle
                final_handles.append(handle)
                final_labels.append(label)
        ax.legend(final_handles, final_labels, loc=legend_loc, fontsize=11)
    elif ax.has_data():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique_entries = {}
            final_handles, final_labels = [], []
            for handle, label in zip(handles, labels):
                if label not in unique_entries:
                    unique_entries[label] = handle
                    final_handles.append(handle)
                    final_labels.append(label)
            ax.legend(final_handles, final_labels, loc=legend_loc, fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    has_plotted_data_non_negative = False
    if ax.has_data():
        for line in ax.get_lines():
            ydata = line.get_ydata()
            if isinstance(ydata, (pd.Series, np.ndarray)) and ydata.size > 0:
                numeric_ydata = pd.to_numeric(ydata, errors='coerce')
                if np.any(numeric_ydata[~np.isnan(numeric_ydata)] >= 0):
                    has_plotted_data_non_negative = True
                    break
    if has_plotted_data_non_negative:
        ax.set_ylim(bottom=0)
    plt.tight_layout(pad=1.5)

def plot_average_latency_comparison(average_logs_dir: str, output_dir: str = DEFAULT_IMG_DIR, metric_to_plot: str = 'experienced_latency_ms'):
    fig, ax = plt.subplots(figsize=(15, 7))
    strategies_plotted_count = 0
    window_size = 5
    legend_handles, legend_labels = [], []
    logger.info(f"Searching for aggregated CSV files in: {average_logs_dir} to plot '{metric_to_plot}'")
    if not os.path.isdir(average_logs_dir):
        logger.error(f"Aggregated logs directory not found: {average_logs_dir}")
        plt.close(fig)
        return

    xlim_for_comparison_plot = COMPARISON_X_AXIS_LIMIT

    y_axis_label = "Average Latency (ms)"
    if metric_to_plot == 'experienced_latency_ms':
        plot_title_metric_part = "Average Chosen Server Latency"
    elif metric_to_plot == 'experienced_latency_ms_CLIENT':
        plot_title_metric_part = "Average Client Measured Latency"
    elif metric_to_plot == 'dynamic_best_server_latency':
        plot_title_metric_part = "Average Optimal Server Latency"
    else:
        plot_title_metric_part = f"Average {metric_to_plot.replace('_', ' ').title()}"
    main_plot_title = f"Comparison of {plot_title_metric_part}\nAcross Steering Strategies"

    for filename in sorted(os.listdir(average_logs_dir)):
        if not (filename.startswith("log_") and "_average" in filename and filename.endswith(".csv")):
            continue
        agg_file_path = os.path.join(average_logs_dir, filename)
        logger.debug(f"Processing aggregated file for comparison: {filename}")
        try:
            df_agg_full = pd.read_csv(agg_file_path)
            df_agg = df_agg_full[df_agg_full['sim_time_client'] <= xlim_for_comparison_plot].copy()

            if df_agg.empty or 'sim_time_client' not in df_agg.columns or metric_to_plot not in df_agg.columns:
                logger.warning(f"File {filename} is empty or missing '{metric_to_plot}' or 'sim_time_client' columns (or no data up to {xlim_for_comparison_plot}s). Skipping.")
                continue

            df_agg = df_agg.sort_values(by='sim_time_client').copy()
            filename_no_ext = os.path.splitext(filename)[0]
            strategy_col_data = df_agg['rl_strategy'] if 'rl_strategy' in df_agg.columns else None
            base_strategy_name = extract_strategy_name(filename_no_ext, strategy_col_data)
            logger.debug(f"File: {filename}, Extracted Strategy: {base_strategy_name}")

            style = STRATEGY_STYLES.get(base_strategy_name, STRATEGY_STYLES["default"])
            current_legend_label = style['label']
            if style['label'] == "Unknown" and base_strategy_name != "Unknown":
                current_legend_label = base_strategy_name.replace('_', ' ').title()
                if base_strategy_name == "d_ucb": current_legend_label = "D-UCB"

            df_plot_data = df_agg.dropna(subset=['sim_time_client', metric_to_plot]).copy()
            if not df_plot_data.empty:
                if len(df_plot_data[metric_to_plot]) >= window_size:
                    y_values_to_plot = df_plot_data[metric_to_plot].rolling(
                        window=window_size, center=True, min_periods=1).mean()
                else:
                     y_values_to_plot = df_plot_data[metric_to_plot]
                line, = ax.plot(df_plot_data['sim_time_client'], y_values_to_plot,
                                 linestyle='-', linewidth=2, alpha=0.9, color=style["color"])
                if not any(lbl == current_legend_label for lbl in legend_labels):
                    legend_handles.append(line)
                    legend_labels.append(current_legend_label)
                strategies_plotted_count += 1
            else:
                logger.warning(f"No valid '{metric_to_plot}' data in {filename} (up to {xlim_for_comparison_plot}s) to plot.")
        except Exception as e:
            logger.error(f"Error processing file {filename} for comparison: {e}", exc_info=True)

    if strategies_plotted_count == 0:
        logger.warning(f"No strategies were plotted for metric '{metric_to_plot}'.")
        plt.close(fig)
        return

    format_comparison_plot(ax,
                           main_plot_title,
                           "Average Simulation Time (s)",
                           y_axis_label,
                           legend_loc='upper right',
                           custom_legend_handles=legend_handles,
                           custom_legend_labels=legend_labels,
                           xlim_max=xlim_for_comparison_plot)

    output_filename_base = metric_to_plot.replace("experienced_latency_ms", "latency")
    if "CLIENT" in output_filename_base:
        output_filename_base = output_filename_base.replace("_CLIENT", "_client_measured")
    if "dynamic_best_server_latency" in output_filename_base:
        output_filename_base = "optimal_latency"
    output_filename = f"all_strategies_{output_filename_base}_comparison.png"
    plot_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    logger.info(f"Comparison graph ({metric_to_plot}) saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compares average latency metrics from aggregated logs of different strategies.")
    parser.add_argument("--agg_dir", type=str, default=DEFAULT_AVERAGE_LOGS_DIR,
                        help=f"Directory with aggregated CSV files. Default: {DEFAULT_AVERAGE_LOGS_DIR}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_IMG_DIR,
                        help=f"Directory to save the graph. Default: {DEFAULT_IMG_DIR}")
    parser.add_argument("--metric", type=str, default="experienced_latency_ms",
                        choices=["experienced_latency_ms", "experienced_latency_ms_CLIENT", "dynamic_best_server_latency"],
                        help="Latency metric to plot for comparison.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    _handler_compare = logging.StreamHandler()
    log_level_to_set = logging.DEBUG if args.verbose else logging.INFO
    _formatter_compare = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    _handler_compare.setFormatter(_formatter_compare)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(_handler_compare)
    logger.setLevel(log_level_to_set)

    logger.info(f"Logging level set to {logging.getLevelName(logger.getEffectiveLevel())}.")
    logger.info(f"Starting strategy comparison (metric: {args.metric}) from logs in: {args.agg_dir}")
    plot_average_latency_comparison(args.agg_dir, args.output_dir, metric_to_plot=args.metric)