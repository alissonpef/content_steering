import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import argparse
import logging
import numpy as np

logger = logging.getLogger("compare_strategies")

BASE_GRAPHICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_PROCESSED_LOGS_DIR = os.path.join(BASE_GRAPHICS_DIR, "data", "processed")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_GRAPHICS_DIR, "output")
COMPARISON_X_AXIS_LIMIT = 150
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

STRATEGY_STYLES = {
    "ucb1": {"color": "tab:blue", "label": "UCB1"},
    "epsilon_greedy": {"color": "tab:green", "label": "Epsilon Greedy"},
    "random": {"color": "tab:red", "label": "Random"},
    "oracle_best_choice": {"color": "tab:purple", "label": "Optimal Strategy"},
    "no_steering": {"color": "tab:brown", "label": "No Steering"},
    "d_ucb": {"color": "tab:cyan", "label": "D-UCB"},
    "linucb": {"color": "tab:pink", "label": "LinUCB"},
    "default": {"color": "tab:grey", "label": "Unknown"}
}

def extract_strategy_name(filename_no_ext: str, df_column_data: pd.Series = None) -> str:
    if df_column_data is not None and not df_column_data.dropna().empty:
        first_valid_strategy = df_column_data.dropna().iloc[0]
        if isinstance(first_valid_strategy, str):
            return first_valid_strategy

    match = re.match(r"log_([a-zA-Z0-9_]+?)_average", filename_no_ext)
    if match:
        return match.group(1)
        
    logger.warning(f"Could not extract strategy name from '{filename_no_ext}'.")
    return "Unknown"

def format_comparison_plot(ax, title, xlabel, ylabel, legend_loc='best', xlim_max=None):
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    if xlim_max is not None and ax.has_data():
        ax.set_xticks(np.arange(0, xlim_max + 1, 15))
        ax.set_xlim(left=0, right=xlim_max)

    if ax.has_data():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            sorted_legend = sorted(zip(labels, handles), key=lambda t: t[0])
            labels_sorted, handles_sorted = zip(*sorted_legend)
            ax.legend(handles_sorted, labels_sorted, loc=legend_loc, fontsize=11)
    
    ax.grid(True, linestyle=':', alpha=0.7)
    if ax.has_data():
        ax.set_ylim(bottom=0)
    plt.tight_layout(pad=1.5)

def plot_average_latency_comparison(average_logs_dir: str, output_dir: str = DEFAULT_OUTPUT_DIR, metric_to_plot: str = 'experienced_latency_ms'):
    fig, ax = plt.subplots(figsize=(15, 7))
    strategies_plotted = set()
    window_size = 5
    
    logger.info(f"Searching for aggregated CSV files in: {average_logs_dir} to plot '{metric_to_plot}'")
    if not os.path.isdir(average_logs_dir):
        logger.error(f"Aggregated logs directory not found: {average_logs_dir}")
        plt.close(fig)
        return

    xlim_for_comparison_plot = COMPARISON_X_AXIS_LIMIT

    y_axis_label = "Average Latency (ms)"
    plot_title_metric_part = f"Average {metric_to_plot.replace('_', ' ').replace('ms', '').strip().title()}"
    if metric_to_plot == 'dynamic_best_server_latency':
        plot_title_metric_part = "Average Optimal Server Latency"
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
                logger.warning(f"File {filename} is empty or missing required columns. Skipping.")
                continue

            df_agg = df_agg.sort_values(by='sim_time_client').copy()
            filename_no_ext = os.path.splitext(filename)[0]
            strategy_col_data = df_agg['rl_strategy'] if 'rl_strategy' in df_agg.columns else None
            base_strategy_name = extract_strategy_name(filename_no_ext, strategy_col_data)
            
            if base_strategy_name in strategies_plotted:
                logger.debug(f"Strategy '{base_strategy_name}' from {filename} already plotted. Skipping to avoid duplicates.")
                continue

            style = STRATEGY_STYLES.get(base_strategy_name, STRATEGY_STYLES["default"])
            current_legend_label = style['label']

            df_plot_data = df_agg.dropna(subset=['sim_time_client', metric_to_plot]).copy()
            if not df_plot_data.empty:
                y_values_to_plot = df_plot_data[metric_to_plot].rolling(
                    window=window_size, center=True, min_periods=1).mean()
                
                ax.plot(df_plot_data['sim_time_client'], y_values_to_plot,
                        linestyle='-', linewidth=2, alpha=0.9, color=style["color"], label=current_legend_label)
                strategies_plotted.add(base_strategy_name)
            else:
                logger.warning(f"No valid data in {filename} for metric '{metric_to_plot}' to plot.")
        except Exception as e:
            logger.error(f"Error processing file {filename} for comparison: {e}", exc_info=True)

    if not strategies_plotted:
        logger.warning(f"No strategies were plotted for metric '{metric_to_plot}'. No graph will be generated.")
        plt.close(fig)
        return

    format_comparison_plot(ax,
                           main_plot_title,
                           "Average Simulation Time (s)",
                           y_axis_label,
                           legend_loc='upper right',
                           xlim_max=xlim_for_comparison_plot)

    output_filename_base = metric_to_plot.replace("experienced_latency_ms", "latency")
    output_filename = f"all_strategies_{output_filename_base}_comparison.png"
    plot_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    logger.info(f"Comparison graph ({metric_to_plot}) saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compares average latency metrics from aggregated logs of different strategies.")
    parser.add_argument("--agg_dir", type=str, default=DEFAULT_PROCESSED_LOGS_DIR,
                        help=f"Directory with aggregated CSV files. Default: {DEFAULT_PROCESSED_LOGS_DIR}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the graph. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--metric", type=str, default="experienced_latency_ms",
                        choices=["experienced_latency_ms", "experienced_latency_ms_CLIENT", "dynamic_best_server_latency"],
                        help="Latency metric to plot for comparison.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    _handler_compare = logging.StreamHandler()
    log_level_to_set = logging.DEBUG if args.verbose else logging.INFO
    _formatter_compare = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    _handler_compare.setFormatter(_formatter_compare)
    if not logger.handlers:
        logger.addHandler(_handler_compare)
    logger.setLevel(log_level_to_set)

    logger.info(f"Logging level set to {logging.getLevelName(logger.getEffectiveLevel())}.")
    logger.info(f"Starting strategy comparison (metric: {args.metric}) from logs in: {args.agg_dir}")
    plot_average_latency_comparison(args.agg_dir, args.output_dir, metric_to_plot=args.metric)