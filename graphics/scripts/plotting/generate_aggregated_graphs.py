import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import argparse
import re
import json
import logging
import numpy as np
import math

logger = logging.getLogger("plot_aggregated_logs")

BASE_GRAPHICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_PROCESSED_LOGS_DIR = os.path.join(BASE_GRAPHICS_DIR, "data", "processed")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_GRAPHICS_DIR, "output")
AGGREGATED_X_AXIS_LIMIT = 150
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

SERVER_DISPLAY_NAMES = {
    "video-streaming-cache-1": "Cache Server 1 (BR)",
    "video-streaming-cache-2": "Cache Server 2 (CL)",
    "video-streaming-cache-3": "Cache Server 3 (CO)",
}
SERVER_COLORS = {
    "video-streaming-cache-1": "tab:green",
    "video-streaming-cache-2": "tab:orange",
    "video-streaming-cache-3": "tab:blue",
}
KNOWN_CACHE_SERVER_KEYS_UNDERSCORE = [
    "video_streaming_cache_1", "video_streaming_cache_2", "video_streaming_cache_3"
]

def parse_json_series_to_dataframe(series: pd.Series, prefix: str = "") -> pd.DataFrame:
    parsed_rows = []
    all_normalized_keys_in_series = set()
    temp_parsed_dicts = []
    valid_indices = series.dropna().index
    for json_str in series.dropna():
        try:
            data_dict = json.loads(json_str)
            if isinstance(data_dict, dict):
                normalized_dict = {str(k).replace('-', '_'): v for k, v in data_dict.items()}
                all_normalized_keys_in_series.update(normalized_dict.keys())
                temp_parsed_dicts.append(normalized_dict)
            else: temp_parsed_dicts.append({})
        except (json.JSONDecodeError, TypeError):
            logger.debug(f"Failed to parse JSON (agg): '{str(json_str)[:70]}...'")
            temp_parsed_dicts.append({})
    final_column_keys_to_check = KNOWN_CACHE_SERVER_KEYS_UNDERSCORE
    if all_normalized_keys_in_series:
        final_column_keys_to_check = list(set(final_column_keys_to_check) | all_normalized_keys_in_series)
    prefixed_final_column_keys = {f"{prefix}{key}" for key in final_column_keys_to_check}
    for norm_dict in temp_parsed_dicts:
        row_data = {prefixed_key: norm_dict.get(prefixed_key.replace(prefix, "", 1)) for prefixed_key in prefixed_final_column_keys}
        parsed_rows.append(row_data)
    if not parsed_rows:
        return pd.DataFrame(columns=list(prefixed_final_column_keys))
    df_result = pd.DataFrame(parsed_rows, index=valid_indices, columns=list(prefixed_final_column_keys))
    return df_result

def format_plot_aggregated(ax, title, xlabel, ylabel, legend_loc='best',
                           y_log_scale=False, custom_legend_handles=None,
                           custom_legend_labels=None, xlim_max=None):
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if xlim_max is not None and ax.has_data():
        ax.set_xticks(np.arange(0, xlim_max + 1, 15))
        ax.set_xlim(left=0, right=xlim_max)
    elif ax.has_data():
        current_xlim_left, current_xlim_right = ax.get_xlim()
        start_tick = 0
        if current_xlim_left > 7.5 :
             start_tick = math.floor(current_xlim_left / 15) * 15
        ax.set_xticks(np.arange(start_tick, current_xlim_right + 1, 15))

    if y_log_scale:
        ax.set_yscale('log')
        if ax.has_data(): ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    else:
        if ax.has_data():
            ax.set_ylim(bottom=0)

    if custom_legend_handles and custom_legend_labels:
        ax.legend(custom_legend_handles, custom_legend_labels, loc=legend_loc, fontsize=10)
    elif ax.has_data() and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, which='major')
    if y_log_scale and ax.has_data(): ax.grid(True, linestyle=':', alpha=0.3, which='minor')
    plt.tight_layout()

def generate_plots_for_aggregated(csv_file_path: str):
    if not os.path.exists(csv_file_path):
        logger.error(f"Aggregated CSV file not found: {csv_file_path}")
        return
    csv_filename_no_ext = os.path.splitext(os.path.basename(csv_file_path))[0]
    current_img_dir = os.path.join(DEFAULT_OUTPUT_DIR, csv_filename_no_ext)
    os.makedirs(current_img_dir, exist_ok=True)
    logger.info(f"Reading aggregated data from: {csv_filename_no_ext}.csv")
    try:
        df_agg = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        logger.warning(f"Aggregated CSV file {csv_filename_no_ext}.csv is empty.")
        return
    if df_agg.empty:
        logger.warning(f"Aggregated CSV file {csv_filename_no_ext}.csv is empty.")
        return

    strategy_name_from_df = "N/A (Aggregated)"
    if 'rl_strategy' in df_agg.columns and not df_agg['rl_strategy'].dropna().empty:
        strategy_name_from_df = df_agg['rl_strategy'].dropna().iloc[0]
    else:
        match = re.match(r"log_([a-zA-Z0-9_]+?)_average", csv_filename_no_ext)
        if match: strategy_name_from_df = match.group(1)

    strategy_display_name = strategy_name_from_df.replace("_", " ").title()
    if strategy_name_from_df == "d_ucb": strategy_display_name = "D-UCB"
    if strategy_name_from_df == "linucb": strategy_display_name = "LinUCB"


    xlim_for_plots = AGGREGATED_X_AXIS_LIMIT
    df_agg = df_agg[df_agg['sim_time_client'] <= xlim_for_plots].copy()
    if df_agg.empty:
        logger.warning(f"No data in aggregated file {csv_filename_no_ext}.csv up to {xlim_for_plots}s.")
        return

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    if 'experienced_latency_ms' in df_agg.columns and 'dynamic_best_server_latency' in df_agg.columns:
        legend1_handles, legend1_labels = [], []
        df_plot_chosen_oracle = df_agg.dropna(subset=['sim_time_client', 'experienced_latency_ms'])
        line_chosen, = ax1.plot(df_plot_chosen_oracle['sim_time_client'], df_plot_chosen_oracle['experienced_latency_ms'],
                                marker='.', linestyle='-', markersize=4, alpha=0.8, color='darkblue')
        legend1_handles.append(line_chosen)
        legend1_labels.append('Avg. Chosen Server Latency')
        
        df_plot_optimal_oracle = df_agg.dropna(subset=['sim_time_client', 'dynamic_best_server_latency'])
        line_optimal, = ax1.plot(df_plot_optimal_oracle['sim_time_client'], df_plot_optimal_oracle['dynamic_best_server_latency'],
                                 marker='.', linestyle='--', markersize=4, alpha=0.7, color='tab:red')
        legend1_handles.append(line_optimal)
        legend1_labels.append('Avg. Optimal Server Latency')
        
        format_plot_aggregated(ax1, f"Average Chosen Server Latency vs Optimal Latency\nStrategy: {strategy_display_name}",
                               "Average Simulation Time (s)", "Average Latency (ms)", legend_loc='upper right',
                               custom_legend_handles=legend1_handles, custom_legend_labels=legend1_labels,
                               xlim_max=xlim_for_plots)
        plt.savefig(os.path.join(current_img_dir, "1_avg_latency_chosen_vs_optimal.png"))
    else:
        logger.info(f"Plot 1 (Latency vs Optimal) skipped: Missing required columns.")
    plt.close(fig1)

    value_cols_agg = [col for col in df_agg.columns if col.startswith('value_')]
    if value_cols_agg:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        legend2_handles, legend2_labels = [], []
        y_label_g2 = "Average Estimated RL Value"
        if "ucb" in strategy_name_from_df.lower():
            y_label_g2 = f"Average Estimated Reward ({strategy_display_name})"
        elif strategy_name_from_df.lower() == "epsilon_greedy":
            y_label_g2 = "Average Estimated Latency (Punishment)"
        
        for col_name in value_cols_agg:
            server_key_u = col_name.replace('value_', '')
            server_key_h = server_key_u.replace('_', '-')
            color = SERVER_COLORS.get(server_key_h, 'grey')
            label_text = SERVER_DISPLAY_NAMES.get(server_key_h, server_key_u)
            df_subset = df_agg.dropna(subset=['sim_time_client', col_name])
            if not df_subset.empty:
                line, = ax2.plot(df_subset['sim_time_client'], df_subset[col_name],
                                 marker='.', linestyle='-', markersize=3, alpha=0.8, color=color)
                if not any(l == label_text for l in legend2_labels):
                    legend2_handles.append(line)
                    legend2_labels.append(label_text)
        
        format_plot_aggregated(ax2, f"Average RL Algorithm's Estimated Server Values\nStrategy: {strategy_display_name}",
                               "Average Simulation Time (s)", y_label_g2, legend_loc='upper right',
                               custom_legend_handles=legend2_handles, custom_legend_labels=legend2_labels,
                               xlim_max=xlim_for_plots)
        plt.savefig(os.path.join(current_img_dir, "2_avg_rl_estimated_values.png"))
        plt.close(fig2)
    else:
        logger.info(f"Plot 2 (RL Values) skipped: No 'value_*' columns found for {strategy_display_name}.")

    actual_count_cols = [col for col in df_agg.columns if col.startswith('actual_count_')]
    count_cols = [col for col in df_agg.columns if col.startswith('count_')]
    cols_to_plot_counts, prefix_in_use, y_label_g3 = [], None, "Avg. Selections (Pulls)"
    
    if strategy_name_from_df.lower() == "d_ucb" and actual_count_cols:
        cols_to_plot_counts, prefix_in_use = actual_count_cols, 'actual_count_'
        y_label_g3 = "Avg. Actual Selections (D-UCB)"
    elif count_cols:
        cols_to_plot_counts, prefix_in_use = count_cols, 'count_'
        if strategy_name_from_df.lower() == "d_ucb":
            y_label_g3 = "Avg. Discounted Selections (D-UCB Fallback)"

    if cols_to_plot_counts:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        legend3_handles, legend3_labels = [], []
        for col_name in cols_to_plot_counts:
            server_key_u = col_name.replace(prefix_in_use, '')
            server_key_h = server_key_u.replace('_', '-')
            color = SERVER_COLORS.get(server_key_h, 'grey')
            label_text = SERVER_DISPLAY_NAMES.get(server_key_h, server_key_u)
            df_subset = df_agg.dropna(subset=['sim_time_client', col_name])
            if not df_subset.empty:
                line, = ax3.plot(df_subset['sim_time_client'], df_subset[col_name],
                                 marker='.', linestyle='-', markersize=3, alpha=0.8, color=color)
                if not any(l == label_text for l in legend3_labels):
                    legend3_handles.append(line)
                    legend3_labels.append(label_text)
        
        format_plot_aggregated(ax3, f"Average RL Algorithm's Server Selection Counts\nStrategy: {strategy_display_name}",
                               "Average Simulation Time (s)", y_label_g3, legend_loc='upper left',
                               custom_legend_handles=legend3_handles, custom_legend_labels=legend3_labels,
                               xlim_max=xlim_for_plots)
        plt.savefig(os.path.join(current_img_dir, "3_avg_rl_selection_counts.png"))
        plt.close(fig3)
    else:
        logger.info(f"Plot 3 (RL Counts) skipped: No RL count columns found for {strategy_display_name}.")
        
    if 'all_servers_oracle_latency_json' in df_agg.columns and not df_agg['all_servers_oracle_latency_json'].dropna().empty:
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        legend4_handles, legend4_labels = [], []
        df_all_lat_agg_parsed = parse_json_series_to_dataframe(df_agg['all_servers_oracle_latency_json'].dropna(), prefix="")
        if not df_all_lat_agg_parsed.empty:
            df_all_lat_agg_with_time = pd.concat([
                df_agg.loc[df_all_lat_agg_parsed.index, 'sim_time_client'], df_all_lat_agg_parsed
            ], axis=1).reset_index(drop=True)
            
            all_oracle_cols = sorted([c for c in df_all_lat_agg_with_time.columns if c in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE])
            for server_col_u in all_oracle_cols:
                server_col_h = server_col_u.replace('_', '-')
                color = SERVER_COLORS.get(server_col_h, 'grey')
                label_text = SERVER_DISPLAY_NAMES.get(server_col_h, server_col_u)
                df_subset = df_all_lat_agg_with_time.dropna(subset=['sim_time_client', server_col_u])
                if not df_subset.empty:
                    line, = ax4.plot(df_subset['sim_time_client'], df_subset[server_col_u],
                                     marker='.', linestyle='-', markersize=2, alpha=0.7, color=color)
                    if not any(l == label_text for l in legend4_labels):
                        legend4_handles.append(line)
                        legend4_labels.append(label_text)
            
            format_plot_aggregated(ax4, f"Average Simulated Latency Landscape for All Servers\nStrategy: {strategy_display_name}",
                                   "Average Simulation Time (s)", "Average Simulated Latency (ms)", legend_loc='upper right',
                                   custom_legend_handles=legend4_handles, custom_legend_labels=legend4_labels,
                                   xlim_max=xlim_for_plots)
            plt.savefig(os.path.join(current_img_dir, "4_avg_all_servers_oracle_latency.png"))
        plt.close(fig4)
    else:
        logger.info(f"Plot 4 (Latency Landscape) skipped: 'all_servers_oracle_latency_json' not found or empty.")

    logger.info(f"Aggregated graph generation for '{csv_filename_no_ext}' complete. Saved in: {current_img_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs from AGGREGATED simulation CSV logs.")
    parser.add_argument("csv_filename", type=str, help="Name of the aggregated CSV file (e.g., log_ucb1_average.csv).")
    parser.add_argument("--output_dir",type=str,default=DEFAULT_OUTPUT_DIR,help=f"Base directory to save graphs. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--verbose","-v",action="store_true",help="Enable DEBUG logging.")
    args = parser.parse_args()

    _handler_main_agg = logging.StreamHandler()
    log_level_to_set = logging.DEBUG if args.verbose else logging.INFO
    _formatter_main_agg = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    _handler_main_agg.setFormatter(_formatter_main_agg)
    if not logger.handlers:
        logger.addHandler(_handler_main_agg)
    logger.setLevel(log_level_to_set)

    logger.info(f"Logging level set to {logging.getLevelName(logger.getEffectiveLevel())}.")

    csv_path = args.csv_filename
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        potential_path = os.path.join(DEFAULT_PROCESSED_LOGS_DIR, csv_path)
        if os.path.exists(potential_path):
            csv_path = potential_path
    
    if os.path.exists(csv_path):
        logger.info(f"Processing aggregated file: {os.path.basename(csv_path)}")
        generate_plots_for_aggregated(csv_path)
    else:
        logger.error(f"File not found: {csv_path}")