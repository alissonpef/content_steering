import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import json
import argparse
import numpy as np
import logging
import re 


logger = logging.getLogger("generate_graphs")

BASE_GRAPHICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_RAW_LOGS_DIR = os.path.join(BASE_GRAPHICS_DIR, "data", "raw")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_GRAPHICS_DIR, "output")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

SERVER_DISPLAY_NAMES = {
    "video-streaming-cache-1": "Cache Server 1 (BR)",
    "video-streaming-cache-2": "Cache Server 2 (CL)",
    "video-streaming-cache-3": "Cache Server 3 (CO)",
    "N/A_NO_NODES_FROM_SELECTION": "No Selection",
    "N/A_NO_NODES_FROM_RL": "No RL Nodes",
    "N/A": "N/A",
    "DynamicBest": "Optimal Server"
}
SERVER_COLORS = {
    "video-streaming-cache-1": "tab:green",
    "video-streaming-cache-2": "tab:orange",
    "video-streaming-cache-3": "tab:blue",
    "DynamicBest": "tab:red",
    "N/A_NO_NODES_FROM_SELECTION": "tab:grey",
    "N/A_NO_NODES_FROM_RL": "silver",
    "N/A": "lightgrey"
}
KNOWN_CACHE_SERVER_KEYS_UNDERSCORE = [
    "video_streaming_cache_1", "video_streaming_cache_2", "video_streaming_cache_3"
]
ACTUAL_CACHE_SERVER_NAMES_HYPHEN = [key for key in SERVER_DISPLAY_NAMES.keys() if "cache" in key.lower()]

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
            logger.debug(f"Failed to parse JSON: '{str(json_str)[:70]}...'")
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

def find_dynamic_best_server_and_latency(row):
    if pd.isna(row['all_servers_oracle_latency_json']):
        return None, np.nan
    try:
        server_latencies = json.loads(row['all_servers_oracle_latency_json'])
        valid_server_latencies = {
            s_name: lat
            for s_name, lat in server_latencies.items()
            if s_name.replace('_','-') in ACTUAL_CACHE_SERVER_NAMES_HYPHEN and isinstance(lat, (int, float))
        }
        if not valid_server_latencies: return None, np.nan
        best_server_name_key = min(valid_server_latencies, key=valid_server_latencies.get)
        best_server_latency = valid_server_latencies[best_server_name_key]
        return best_server_name_key.replace('_','-'), best_server_latency
    except (json.JSONDecodeError, TypeError): return None, np.nan
    except Exception: return None, np.nan

def format_plot(ax, title, xlabel, ylabel, legend_loc='best', y_log_scale=False, custom_legend_handles=None, custom_legend_labels=None):
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if y_log_scale:
        ax.set_yscale('log')
        if ax.has_data(): ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    else:
        has_plotted_data = False
        if ax.has_data():
            for line in ax.get_lines():
                ydata = line.get_ydata()
                if isinstance(ydata, (pd.Series, np.ndarray)) and ydata.size > 0:
                    numeric_ydata = pd.to_numeric(ydata, errors='coerce')
                    if np.any(numeric_ydata[~np.isnan(numeric_ydata)] >= 0):
                        has_plotted_data = True
                        break
        if has_plotted_data:
            ax.set_ylim(bottom=0)

    if custom_legend_handles and custom_legend_labels:
        ax.legend(custom_legend_handles, custom_legend_labels, loc=legend_loc, fontsize=10)
    else:
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(handles, labels, loc=legend_loc, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, which='major')
    if y_log_scale and ax.has_data(): ax.grid(True, linestyle=':', alpha=0.3, which='minor')
    plt.tight_layout(pad=1.2)

def generate_plots(csv_file_path: str):
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file not found: {csv_file_path}")
        return
    csv_filename_with_ext = os.path.basename(csv_file_path)
    csv_filename_no_ext = os.path.splitext(csv_filename_with_ext)[0]
    current_img_dir = os.path.join(DEFAULT_OUTPUT_DIR, csv_filename_no_ext)
    os.makedirs(current_img_dir, exist_ok=True)
    logger.info(f"Reading data from: {csv_filename_with_ext}")
    try:
        df = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file {csv_filename_with_ext} is empty. No graphs will be generated.")
        return
    if df.empty:
        logger.warning(f"CSV file {csv_filename_with_ext} is empty. No graphs will be generated.")
        return
    df.sort_values(by="sim_time_client", inplace=True)
    df.reset_index(drop=True, inplace=True)

    strategy_name_from_df = "N/A"
    if 'rl_strategy' in df.columns and not df['rl_strategy'].dropna().empty:
        strategy_name_from_df = df['rl_strategy'].dropna().iloc[0]
    else: 
        match = re.match(r"log_([a-zA-Z0-9_]+)", csv_filename_no_ext)
        if match:
            strategy_name_from_df = match.group(1)

    strategy_display_name = strategy_name_from_df.replace('_', ' ').title()
    if strategy_name_from_df == "d_ucb": strategy_display_name = "D-UCB"
    if strategy_name_from_df == "linucb": strategy_display_name = "LinUCB"

    strategy_display_name = strategy_name_from_df.replace('_', ' ').title()
    if strategy_name_from_df == "d_ucb": strategy_display_name = "D-UCB"
    if strategy_name_from_df == "linucb": strategy_display_name = "LinUCB"

    if 'all_servers_oracle_latency_json' in df.columns:
        dynamic_best_info = df.apply(find_dynamic_best_server_and_latency, axis=1, result_type='expand')
        df[['dynamic_best_server_name', 'dynamic_best_server_latency']] = dynamic_best_info
    else:
        df['dynamic_best_server_name'] = None
        df['dynamic_best_server_latency'] = np.nan

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    plot_made_g1 = False
    legend1_handles, legend1_labels = [], []
    window_size = 10
    if 'experienced_latency_ms' in df.columns and 'sim_time_client' in df.columns:
        df_chosen_latency = df.dropna(subset=['sim_time_client', 'experienced_latency_ms'])
        if not df_chosen_latency.empty and len(df_chosen_latency) >= window_size:
            ma_chosen = df_chosen_latency['experienced_latency_ms'].rolling(window=window_size, center=True, min_periods=1).mean()
            line_chosen, = ax1.plot(df_chosen_latency['sim_time_client'], ma_chosen,
                                    linestyle='-', color='navy', linewidth=1.5, alpha=0.9)
            legend1_handles.append(line_chosen)
            legend1_labels.append(f'MA ({window_size}s) - Chosen Server')
            plot_made_g1 = True
    if 'dynamic_best_server_latency' in df.columns and 'sim_time_client' in df.columns:
        df_dynamic_best_latency = df.dropna(subset=['sim_time_client', 'dynamic_best_server_latency'])
        if not df_dynamic_best_latency.empty and len(df_dynamic_best_latency) >= window_size:
            ma_dynamic_best = df_dynamic_best_latency['dynamic_best_server_latency'].rolling(window=window_size, center=True, min_periods=1).mean()
            line_optimal, = ax1.plot(df_dynamic_best_latency['sim_time_client'], ma_dynamic_best,
                                     linestyle='--', color=SERVER_COLORS.get("DynamicBest", "tab:red"), linewidth=1.5, alpha=0.9)
            legend1_handles.append(line_optimal)
            legend1_labels.append(f'MA ({window_size}s) - Optimal Server')
            plot_made_g1 = True
    if plot_made_g1:
        format_plot(ax1, f"Chosen Server Latency vs Optimal Latency\nStrategy: {strategy_display_name}",
                    "Simulation Time (s)", "Simulated Latency (ms)", legend_loc='upper right',
                    custom_legend_handles=legend1_handles, custom_legend_labels=legend1_labels)
        plt.savefig(os.path.join(current_img_dir, "1_latency_chosen_vs_optimal.png"))
    else: logger.info(f"Insufficient data for Plot 1 for {csv_filename_with_ext}.")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    plot_made_g2 = False
    legend2_handles, legend2_labels = [], []
    if 'steering_decision_main_server' in df.columns and 'sim_time_client' in df.columns:
        df_steering = df.dropna(subset=['steering_decision_main_server', 'sim_time_client'])
        if not df_steering.empty:
            df_s_unique = df_steering.drop_duplicates(subset=['sim_time_client'], keep='first').copy()
            all_y_entities = ACTUAL_CACHE_SERVER_NAMES_HYPHEN + \
                             [val for val in df_s_unique['steering_decision_main_server'].unique() if "N/A" in str(val) or pd.isna(val)] + \
                             (['DynamicBest'] if 'dynamic_best_server_name' in df.columns and df['dynamic_best_server_name'].notna().any() else [])
            unique_y_entities = sorted(list(set(entity for entity in all_y_entities if pd.notna(entity))))
            entity_to_int_map = {entity: i for i, entity in enumerate(unique_y_entities)}
            df_s_unique.loc[:, 'decision_int'] = df_s_unique['steering_decision_main_server'].map(entity_to_int_map)
            df_plot_decision = df_s_unique.dropna(subset=['decision_int'])
            if not df_plot_decision.empty:
                line_algo, = ax2.plot(df_plot_decision['sim_time_client'], df_plot_decision['decision_int'],
                                      drawstyle='steps-post', marker='o', markersize=3, alpha=0.8, color='tab:cyan', label="Algorithm's Server Choice")
                legend2_handles.append(line_algo)
                legend2_labels.append("Algorithm's Server Choice")
                plot_made_g2 = True
            if 'dynamic_best_server_name' in df_s_unique.columns:
                df_s_unique.loc[:, 'dynamic_best_int'] = df_s_unique['dynamic_best_server_name'].map(entity_to_int_map).fillna(-1)
                df_plot_dynamic_best = df_s_unique[df_s_unique['dynamic_best_int'] != -1].dropna(subset=['sim_time_client'])
                if not df_plot_dynamic_best.empty:
                    ax2.scatter(df_plot_dynamic_best['sim_time_client'], df_plot_dynamic_best['dynamic_best_int'],
                                marker='x', s=50, color=SERVER_COLORS.get("DynamicBest", "tab:red"),
                                label="Optimal Server Choice", alpha=0.9, zorder=5)
                    line_optimal_proxy, = plt.plot([], [], linestyle='None', marker='x', markersize=7,
                                                  color=SERVER_COLORS.get("DynamicBest", "tab:red"), label="Optimal Server Choice")
                    if not any(label == "Optimal Server Choice" for label in legend2_labels):
                        legend2_handles.append(line_optimal_proxy)
                        legend2_labels.append("Optimal Server Choice")
                    plot_made_g2 = True
            if plot_made_g2 and entity_to_int_map and unique_y_entities:
                ax2.set_yticks(list(entity_to_int_map.values()))
                ytick_labels = [SERVER_DISPLAY_NAMES.get(entity, str(entity).replace("_", " ").title()) for entity in unique_y_entities]
                ax2.set_yticklabels(ytick_labels)
                ax2.set_ylim(min(entity_to_int_map.values()) - 0.5, max(entity_to_int_map.values()) + 0.5)
    if plot_made_g2:
        format_plot(ax2, f"Steering Decisions and Optimal Server\nStrategy: {strategy_display_name}",
                    "Simulation Time (s)", "Server Entity", legend_loc='upper right',
                    custom_legend_handles=legend2_handles, custom_legend_labels=legend2_labels)
        plt.setp(ax2.get_yticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(current_img_dir, "2_steering_decision_vs_optimal.png"))
    else: logger.info(f"Insufficient data for Plot 2 for {csv_filename_with_ext}.")
    plt.close(fig2)

    if 'rl_values_json' in df.columns and not df['rl_values_json'].dropna().empty:
        fig_rl_values, ax_rl_values = plt.subplots(figsize=(12, 6))
        plot_made_rl_values = False
        legend_rl_values_handles, legend_rl_values_labels = [], []
        y_label_rl_values = "Estimated RL Value"
        df_values_parsed = parse_json_series_to_dataframe(df['rl_values_json'].dropna(), prefix="value_")
        if not df_values_parsed.empty:
            df_values_with_time = pd.concat([df.loc[df_values_parsed.index, 'sim_time_client'], df_values_parsed], axis=1).reset_index(drop=True)
            df_values_unique_time = df_values_with_time.drop_duplicates(subset=['sim_time_client'], keep='last').copy()
            if "ucb" in strategy_name_from_df.lower(): 
                y_label_rl_values = f"Estimated Reward ({strategy_display_name})"
            elif strategy_name_from_df.lower() == "epsilon_greedy":
                y_label_rl_values = "Estimated Average Latency (Punishment)"
            value_cols_plot = sorted([col for col in df_values_unique_time.columns if col.startswith('value_') and any(k_u in col for k_u in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE)])
            for col_name in value_cols_plot:
                s_key_u = col_name.replace('value_', '')
                s_key_h = s_key_u.replace('_', '-')
                color = SERVER_COLORS.get(s_key_h, 'grey')
                label_text = SERVER_DISPLAY_NAMES.get(s_key_h, s_key_u)
                df_subset = df_values_unique_time.dropna(subset=['sim_time_client', col_name]).copy()
                if not df_subset.empty:
                    line, = ax_rl_values.plot(df_subset['sim_time_client'], df_subset[col_name], marker='.', linestyle='-', ms=3, alpha=0.7, color=color)
                    if not any(l == label_text for l in legend_rl_values_labels):
                        legend_rl_values_handles.append(line)
                        legend_rl_values_labels.append(label_text)
                    plot_made_rl_values = True
            if plot_made_rl_values:
                format_plot(ax_rl_values, f"RL Algorithm's Estimated Server Values\nStrategy: {strategy_display_name}",
                            "Simulation Time (s)", y_label_rl_values, legend_loc='upper right',
                            custom_legend_handles=legend_rl_values_handles, custom_legend_labels=legend_rl_values_labels)
                plt.savefig(os.path.join(current_img_dir, "3_rl_estimated_values.png"))
        plt.close(fig_rl_values)
    else:
        logger.info(f"Plot 3 (RL Values) skipped: 'rl_values_json' not found or empty for {strategy_display_name}.")

    counts_json_col_to_use = None
    y_label_rl_counts = "Number of Selections (Pulls)"
    if strategy_name_from_df.lower() == "d_ucb" and 'rl_actual_counts_json' in df.columns and not df['rl_actual_counts_json'].dropna().empty:
        counts_json_col_to_use = 'rl_actual_counts_json'
        y_label_rl_counts = "Actual Number of Selections (D-UCB)"
    elif 'rl_counts_json' in df.columns and not df['rl_counts_json'].dropna().empty:
        counts_json_col_to_use = 'rl_counts_json'
        if strategy_name_from_df.lower() == "d_ucb":
             y_label_rl_counts = "Discounted Selections (D-UCB Fallback)"

    if counts_json_col_to_use:
        fig_rl_counts, ax_rl_counts = plt.subplots(figsize=(12, 6))
        plot_made_rl_counts = False
        legend_rl_counts_handles, legend_rl_counts_labels = [], []
        df_counts_parsed = parse_json_series_to_dataframe(df[counts_json_col_to_use].dropna(), prefix="data_")
        if not df_counts_parsed.empty:
            df_counts_with_time = pd.concat([df.loc[df_counts_parsed.index, 'sim_time_client'], df_counts_parsed], axis=1).reset_index(drop=True)
            df_counts_unique_time = df_counts_with_time.drop_duplicates(subset=['sim_time_client'], keep='last').copy()
            cols_to_plot_counts = sorted([c for c in df_counts_unique_time.columns if c.startswith('data_') and c.replace('data_', '') in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE])
            for col_name_with_prefix in cols_to_plot_counts:
                server_key_u = col_name_with_prefix.replace('data_', '')
                server_key_h = server_key_u.replace('_', '-')
                color = SERVER_COLORS.get(server_key_h, 'grey')
                label_text = SERVER_DISPLAY_NAMES.get(server_key_h, server_key_u)
                df_server_counts = df_counts_unique_time[['sim_time_client', col_name_with_prefix]].copy().dropna(subset=[col_name_with_prefix])
                if not df_server_counts.empty:
                    line, = ax_rl_counts.plot(df_server_counts['sim_time_client'], df_server_counts[col_name_with_prefix], marker='.', linestyle='-', ms=3, alpha=0.7, color=color)
                    if not any(l == label_text for l in legend_rl_counts_labels):
                        legend_rl_counts_handles.append(line)
                        legend_rl_counts_labels.append(label_text)
                    plot_made_rl_counts = True
            if plot_made_rl_counts:
                format_plot(ax_rl_counts, f"RL Algorithm's Server Selection Counts\nStrategy: {strategy_display_name}",
                            "Simulation Time (s)", y_label_rl_counts, legend_loc='upper left',
                            custom_legend_handles=legend_rl_counts_handles, custom_legend_labels=legend_rl_counts_labels)
                plt.savefig(os.path.join(current_img_dir, "4_rl_selection_counts.png"))
        plt.close(fig_rl_counts)
    else:
        logger.info(f"Plot 4 (RL Counts) skipped: No suitable RL counts column found for {strategy_display_name}.")

    if 'all_servers_oracle_latency_json' in df.columns and not df['all_servers_oracle_latency_json'].dropna().empty:
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        plot_made_g5 = False
        legend5_handles, legend5_labels = [], []
        df_all_lat_parsed = parse_json_series_to_dataframe(df['all_servers_oracle_latency_json'].dropna(), prefix="")
        if not df_all_lat_parsed.empty:
            df_all_lat_with_time = pd.concat([df.loc[df_all_lat_parsed.index, 'sim_time_client'], df_all_lat_parsed], axis=1).reset_index(drop=True)
            df_all_lat_unique_time = df_all_lat_with_time.drop_duplicates(subset=['sim_time_client'], keep='last').copy()
            if not df_all_lat_unique_time.empty and 'sim_time_client' in df_all_lat_unique_time.columns:
                all_oracle_cols = sorted([c for c in df_all_lat_unique_time.columns if c != 'sim_time_client' and c in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE])
                for server_col_u in all_oracle_cols:
                    s_key_h = server_col_u.replace('_', '-')
                    color = SERVER_COLORS.get(s_key_h, 'grey')
                    label_text = SERVER_DISPLAY_NAMES.get(s_key_h, server_col_u)
                    df_subset = df_all_lat_unique_time.dropna(subset=['sim_time_client', server_col_u])
                    if not df_subset.empty:
                        line, = ax5.plot(df_subset['sim_time_client'], df_subset[server_col_u], marker='.', linestyle='-', ms=2, alpha=0.6, color=color)
                        if not any(l == label_text for l in legend5_labels):
                            legend5_handles.append(line)
                            legend5_labels.append(label_text)
                        plot_made_g5 = True
                if plot_made_g5:
                    format_plot(ax5, f"Simulated Latency Landscape for All Servers\nStrategy: {strategy_display_name}",
                                "Simulation Time (s)", "Simulated Latency (ms)", legend_loc='upper right',
                                custom_legend_handles=legend5_handles, custom_legend_labels=legend5_labels)
                    plt.savefig(os.path.join(current_img_dir, "5_all_servers_oracle_latency.png"))
        plt.close(fig5)
    else:
        logger.info(f"Plot 5 (Latency Landscape) skipped: 'all_servers_oracle_latency_json' not found or empty.")

    logger.info(f"Graph generation for '{csv_filename_no_ext}' complete. Saved in: {current_img_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs from Content Steering simulation CSV logs.")
    parser.add_argument("csv_argument", type=str, nargs='?', default=None,
                        help="Filename/path to CSV log. Searched in standard dirs if not absolute.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    _handler_main = logging.StreamHandler()
    log_level_to_set = logging.DEBUG if args.verbose else logging.INFO
    _formatter_main = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    _handler_main.setFormatter(_formatter_main)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(_handler_main)
    logger.setLevel(log_level_to_set)

    logger.info(f"Logging level set to {logging.getLevelName(logger.getEffectiveLevel())}.")

    if args.csv_argument:
        csv_to_process, resolved_path = args.csv_argument, None
        paths_to_check = []
        if os.path.isabs(csv_to_process): paths_to_check.append(csv_to_process)
        if not csv_to_process.lower().endswith(".csv"):
             if os.path.isabs(csv_to_process): paths_to_check.append(csv_to_process + ".csv")
        for d_dir in [os.getcwd(), DEFAULT_RAW_LOGS_DIR, os.path.join(DEFAULT_RAW_LOGS_DIR, "Average")]:
            paths_to_check.append(os.path.join(d_dir, os.path.basename(csv_to_process)))
            if not csv_to_process.lower().endswith(".csv"):
                paths_to_check.append(os.path.join(d_dir, os.path.basename(csv_to_process) + ".csv"))
        unique_paths_to_check = sorted(list(set(paths_to_check)), key=lambda p: (not os.path.isabs(p) or not p.startswith(os.getcwd()), p))
        for potential_path in unique_paths_to_check:
            if os.path.exists(potential_path) and os.path.isfile(potential_path):
                resolved_path = os.path.abspath(potential_path)
                break
        if resolved_path:
            logger.info(f"Processing file: {os.path.basename(resolved_path)} (Resolved from '{args.csv_argument}')")
            generate_plots(resolved_path)
        else:
            logger.error(f"File '{args.csv_argument}' not found in search directories: CWD, {DEFAULT_RAW_LOGS_DIR}, {os.path.join(DEFAULT_RAW_LOGS_DIR, 'Average')}.")
    else:
        logger.info(f"No CSV file specified. Processing all CSV files in default directories (non-aggregated).")
        processed_any = False
        for dirname in [DEFAULT_RAW_LOGS_DIR]:
            if os.path.isdir(dirname):
                logger.info(f"Searching for CSV files in: {dirname}")
                for filename in sorted(os.listdir(dirname)):
                    if filename.startswith("log_") and filename.endswith(".csv") and "_average" not in filename:
                        full_path = os.path.join(dirname, filename)
                        logger.info(f"---> Processing {filename} from {os.path.basename(dirname)}/")
                        generate_plots(full_path)
                        processed_any = True
            else: logger.warning(f"Directory not found: {dirname}")
        if not processed_any: logger.warning("No CSV files found to process.")