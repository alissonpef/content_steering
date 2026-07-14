import argparse
import json
import logging
import os
import re

import numpy as np
import pandas as pd

logger = logging.getLogger("aggregate_logs")
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RAW_LOGS_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "logs", "raw")
DEFAULT_PROCESSED_LOGS_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "logs", "aggregated")
MAX_AGGREGATION_TIME_SECONDS = 300
os.makedirs(DEFAULT_PROCESSED_LOGS_DIR, exist_ok=True)
EXPECTED_MAIN_NUMERIC_COLS = [
    "sim_time_client",
    "experienced_latency_ms",
    "experienced_latency_ms_CLIENT",
    "experienced_latency_ms_ORACLE",
    "dynamic_best_server_latency",
    "gamma_value",
]
EXPECTED_CATEGORICAL_COLS_FROM_FIRST_RUN = [
    "client_lat",
    "client_lon",
    "steering_decision_main_server",
    "rl_strategy",
]
KNOWN_CACHE_SERVER_KEYS_UNDERSCORE = [
    "delivery_node_1",
    "delivery_node_2",
    "delivery_node_3",
]


def find_dynamic_best_server_and_latency_for_agg(row_series):
    if pd.isna(row_series["all_servers_oracle_latency_json"]):
        return pd.Series(
            [None, np.nan],
            index=["dynamic_best_server_name_temp", "dynamic_best_server_latency"],
        )
    try:
        raw_latencies = json.loads(row_series["all_servers_oracle_latency_json"])
        normalized_latencies = {k.replace("-", "_"): v for k, v in raw_latencies.items()}
        valid_server_latencies = {
            s_key: lat
            for s_key, lat in normalized_latencies.items()
            if s_key in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE and isinstance(lat, (int, float))
        }
        if not valid_server_latencies:
            return pd.Series(
                [None, np.nan],
                index=["dynamic_best_server_name_temp", "dynamic_best_server_latency"],
            )
        best_server_key_underscore = min(
            list(valid_server_latencies.keys()),
            key=lambda k: float(valid_server_latencies[k]),
        )
        best_server_latency = valid_server_latencies[best_server_key_underscore]
        best_server_name_hyphen = best_server_key_underscore.replace("_", "-")
        return pd.Series(
            [best_server_name_hyphen, best_server_latency],
            index=["dynamic_best_server_name_temp", "dynamic_best_server_latency"],
        )
    except json.JSONDecodeError, TypeError, AttributeError:
        logger.debug(
            f"JSON error in find_dynamic_best_server_and_latency_for_agg: {str(row_series['all_servers_oracle_latency_json'])[:70]}"
        )
        return pd.Series(
            [None, np.nan],
            index=["dynamic_best_server_name_temp", "dynamic_best_server_latency"],
        )
    except Exception as e:
        logger.error(
            f"Unexpected exception in find_dynamic_best_server_and_latency_for_agg: {e}",
            exc_info=True,
        )
        return pd.Series(
            [None, np.nan],
            index=["dynamic_best_server_name_temp", "dynamic_best_server_latency"],
        )


def parse_json_series_to_dataframe(series: pd.Series, prefix: str = "") -> pd.DataFrame:
    parsed_rows = []
    all_normalized_keys_in_series = set()
    temp_parsed_dicts = []
    valid_indices = series.dropna().index
    for json_str in series.dropna():
        try:
            data_dict = json.loads(json_str)
            if isinstance(data_dict, dict):
                normalized_dict = {str(k).replace("-", "_"): v for k, v in data_dict.items()}
                all_normalized_keys_in_series.update(normalized_dict.keys())
                temp_parsed_dicts.append(normalized_dict)
            else:
                temp_parsed_dicts.append({})
        except json.JSONDecodeError, TypeError:
            logger.debug(
                f"Failed to parse JSON in parse_json_series_to_dataframe: '{str(json_str)[:70]}...'"
            )
            temp_parsed_dicts.append({})
    final_column_keys = all_normalized_keys_in_series.union(set(KNOWN_CACHE_SERVER_KEYS_UNDERSCORE))
    prefixed_final_column_keys = {f"{prefix}{key}" for key in final_column_keys}
    for norm_dict in temp_parsed_dicts:
        row_data = {
            prefixed_key: norm_dict.get(prefixed_key.replace(prefix, "", 1))
            for prefixed_key in prefixed_final_column_keys
        }
        parsed_rows.append(row_data)
    if not parsed_rows:
        return pd.DataFrame(columns=list(prefixed_final_column_keys))
    df_result = pd.DataFrame(
        parsed_rows, index=valid_indices, columns=list(prefixed_final_column_keys)
    )
    return df_result


def aggregate_strategy_logs(
    strategy_name: str,
    suffix_pattern: str = "",
    input_dir: str = DEFAULT_RAW_LOGS_DIR,
    output_dir: str = DEFAULT_PROCESSED_LOGS_DIR,
):
    log_files = []
    base_pattern_str = f"log_{strategy_name}"
    if suffix_pattern:
        file_pattern = re.compile(rf"^{re.escape(base_pattern_str + suffix_pattern)}(_\d+)?\.csv$")
    else:
        file_pattern = re.compile(rf"^{re.escape(base_pattern_str)}(.*?)(_\d+)?\.csv$")
    logger.info(
        f"Aggregating logs for strategy: '{strategy_name}', suffix pattern: '{suffix_pattern}', input: '{input_dir}'"
    )
    for filename in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, filename)):
            match = file_pattern.match(filename)
            if match:
                if not suffix_pattern:
                    user_suffix_part_if_any = match.group(1)
                    if (
                        user_suffix_part_if_any
                        and not user_suffix_part_if_any.replace("_", "").isdigit()
                        and user_suffix_part_if_any != ""
                    ):
                        logger.debug(
                            f"Ignoring {filename} due to unspecified suffix '{user_suffix_part_if_any}'. Use --suffix_pattern='{user_suffix_part_if_any}'."
                        )
                        continue
                log_files.append(os.path.join(input_dir, filename))
    if not log_files:
        logger.warning(f"No logs found for '{strategy_name}{suffix_pattern}'.")
        return
    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        f"{len(log_files)} file(s) for aggregation: {', '.join([os.path.basename(f) for f in log_files])}"
    )
    (
        all_main_dfs,
        all_rl_values_dfs,
        all_rl_counts_dfs,
        all_rl_actual_counts_dfs,
        all_server_latencies_dfs,
    ) = [], [], [], [], []
    actual_min_common_duration = float("inf")
    first_run_categorical_data = {}
    for i, f_path in enumerate(log_files):
        try:
            df_run = pd.read_csv(f_path, na_filter=True)
            if df_run.empty or "sim_time_client" not in df_run.columns:
                logger.warning(
                    f"File {os.path.basename(f_path)} empty or missing 'sim_time_client'. Skipping."
                )
                continue
            df_run.dropna(subset=["sim_time_client"], inplace=True)
            if df_run.empty:
                continue
            df_run = df_run[df_run["sim_time_client"] <= MAX_AGGREGATION_TIME_SECONDS].copy()
            if df_run.empty:
                continue
            max_time_this_run_after_cap = df_run["sim_time_client"].max()
            if isinstance(max_time_this_run_after_cap, (int, float)):
                actual_min_common_duration = min(
                    actual_min_common_duration, float(max_time_this_run_after_cap)
                )
            if "all_servers_oracle_latency_json" in df_run.columns:
                best_info = df_run.apply(find_dynamic_best_server_and_latency_for_agg, axis=1)
                df_run["dynamic_best_server_latency"] = best_info["dynamic_best_server_latency"]
            df_run["sim_time_group"] = df_run["sim_time_client"].round().astype(int)
            cols_to_avg = [col for col in EXPECTED_MAIN_NUMERIC_COLS if col in df_run.columns]
            all_main_dfs.append(df_run[["sim_time_group"] + cols_to_avg].copy())
            if i == 0:
                cols_cat = [
                    col for col in EXPECTED_CATEGORICAL_COLS_FROM_FIRST_RUN if col in df_run.columns
                ]
                if cols_cat:
                    temp_cat_df = df_run.groupby("sim_time_group")[cols_cat].first().reset_index()
                    for _, row in temp_cat_df.iterrows():
                        cat_dict = {}
                        for c in cols_cat:
                            val = row[c]
                            if isinstance(val, pd.Series):
                                val = val.iloc[0]
                            if not pd.isna(val) and str(val).lower() != "nan":
                                cat_dict[c] = val
                        first_run_categorical_data[row["sim_time_group"]] = cat_dict
            json_processing_map = [
                ("rl_values_json", all_rl_values_dfs, "value_"),
                ("rl_counts_json", all_rl_counts_dfs, "count_"),
                ("rl_actual_counts_json", all_rl_actual_counts_dfs, "actual_count_"),
                ("all_servers_oracle_latency_json", all_server_latencies_dfs, ""),
            ]
            for json_col, target_list, prefix in json_processing_map:
                if json_col in df_run.columns:
                    col_data = df_run[json_col]
                    if isinstance(col_data, pd.Series) and not col_data.dropna().empty:
                        parsed_df = parse_json_series_to_dataframe(
                            pd.Series(col_data), prefix=prefix
                        )
                        if not parsed_df.empty:
                            parsed_df["sim_time_group"] = df_run.loc[
                                parsed_df.index, "sim_time_group"
                            ]
                            parsed_df.dropna(subset=["sim_time_group"], inplace=True)
                            parsed_df["sim_time_group"] = parsed_df["sim_time_group"].astype(int)
                            target_list.append(parsed_df)
        except Exception as e:
            logger.error(
                f"Error processing {os.path.basename(f_path)}: {e}. Skipping.",
                exc_info=True,
            )
    if not all_main_dfs:
        logger.error("No valid data found for aggregation.")
        return
    effective_duration = min(actual_min_common_duration, MAX_AGGREGATION_TIME_SECONDS)
    logger.info(f"Aggregating data up to effective simulation time of {effective_duration:.2f}s.")
    combined_main_df = pd.concat(all_main_dfs)
    combined_main_df = combined_main_df[combined_main_df["sim_time_group"] <= effective_duration]
    main_cols_to_agg = [
        col for col in combined_main_df.columns if col not in ["sim_time_group", "sim_time_client"]
    ]
    aggregated_df = (
        combined_main_df.groupby("sim_time_group")[main_cols_to_agg].mean().reset_index()
    )
    std_df = combined_main_df.groupby("sim_time_group")[main_cols_to_agg].std().reset_index()
    std_df.rename(columns={c: f"{c}_std_agg" for c in main_cols_to_agg}, inplace=True)
    aggregated_df = pd.merge(aggregated_df, std_df, on="sim_time_group", how="left")
    aggregated_df.rename(columns={"sim_time_group": "sim_time_client"}, inplace=True)
    if first_run_categorical_data:
        cat_df = pd.DataFrame.from_dict(first_run_categorical_data, orient="index").reset_index()
        cat_df.rename(columns={"index": "sim_time_client"}, inplace=True)
        cat_df = cat_df[cat_df["sim_time_client"] <= effective_duration]
        aggregated_df = pd.merge(aggregated_df, cat_df, on="sim_time_client", how="left")
    json_data_to_merge_final = [
        (all_server_latencies_dfs, "oracle_latency_json"),
        (all_rl_values_dfs, "rl_values"),
        (all_rl_counts_dfs, "rl_counts"),
        (all_rl_actual_counts_dfs, "rl_actual_counts"),
    ]
    for data_list, data_type in json_data_to_merge_final:
        if data_list:
            combined_json_df = pd.concat(data_list).dropna(how="all", axis=1)
            if "sim_time_group" in combined_json_df.columns:
                combined_json_df = combined_json_df[
                    combined_json_df["sim_time_group"] <= effective_duration
                ]
                if combined_json_df.empty:
                    continue
                numeric_cols = [c for c in combined_json_df.columns if c != "sim_time_group"]
                avg_json_df = (
                    combined_json_df.groupby("sim_time_group")[numeric_cols].mean().reset_index()
                )
                std_json_df = (
                    combined_json_df.groupby("sim_time_group")[numeric_cols].std().reset_index()
                )
                std_json_df.rename(columns={c: f"{c}_std_agg" for c in numeric_cols}, inplace=True)
                avg_json_df = pd.merge(avg_json_df, std_json_df, on="sim_time_group", how="left")
                avg_json_df.rename(columns={"sim_time_group": "sim_time_client"}, inplace=True)
                if data_type == "oracle_latency_json":
                    server_latency_cols = [
                        c for c in avg_json_df.columns if c in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE
                    ]
                    if server_latency_cols:
                        avg_json_df["all_servers_oracle_latency_json"] = avg_json_df.apply(
                            lambda row, cols=server_latency_cols: json.dumps(
                                {c: row[c] for c in cols if pd.notna(row[c])}
                            ),
                            axis=1,
                        )
                        cols_to_keep = [
                            "sim_time_client",
                            "all_servers_oracle_latency_json",
                        ]
                        aggregated_df = pd.merge(
                            aggregated_df,
                            avg_json_df[cols_to_keep],
                            on="sim_time_client",
                            how="left",
                        )
                else:
                    aggregated_df = pd.merge(
                        aggregated_df, avg_json_df, on="sim_time_client", how="left"
                    )
    if aggregated_df.empty:
        logger.error("Final aggregated DataFrame is empty.")
        return
    preferred_order = [
        "sim_time_client",
        "client_lat",
        "client_lon",
        "experienced_latency_ms",
        "experienced_latency_ms_CLIENT",
        "experienced_latency_ms_ORACLE",
        "dynamic_best_server_latency",
        "all_servers_oracle_latency_json",
        "steering_decision_main_server",
        "rl_strategy",
        "gamma_value",
    ]
    existing_cols = set(aggregated_df.columns)
    final_cols = [col for col in preferred_order if col in existing_cols]
    remaining_cols = sorted(list(existing_cols - set(final_cols)))
    final_cols.extend(remaining_cols)
    aggregated_df_final = aggregated_df[final_cols].sort_values(by="sim_time_client")
    output_base_name = f"log_{strategy_name}{suffix_pattern}_average"
    output_file = os.path.join(output_dir, f"{output_base_name}.csv")
    aggregated_df_final.to_csv(output_file, index=False, float_format="%.3f")
    logger.info(f"Aggregated CSV file saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregates multiple simulation logs for a strategy."
    )
    parser.add_argument(
        "strategy_name",
        type=str,
        help="Base name of the strategy (e.g., ucb1, epsilon_greedy, no_steering).",
    )
    parser.add_argument(
        "--suffix_pattern",
        type=str,
        default="",
        help="Optional suffix pattern in filenames (e.g., _runA).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_RAW_LOGS_DIR,
        help=f"Directory of logs. Default: {DEFAULT_RAW_LOGS_DIR}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_PROCESSED_LOGS_DIR,
        help=f"Directory to save aggregated log. Default: {DEFAULT_PROCESSED_LOGS_DIR}",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()
    _handler_agg = logging.StreamHandler()
    log_level_to_set = logging.DEBUG if args.verbose else logging.INFO
    _formatter_agg = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    _handler_agg.setFormatter(_formatter_agg)
    if not logger.handlers:
        logger.addHandler(_handler_agg)
    logger.setLevel(log_level_to_set)
    logger.info(f"Logging level set to {logging.getLevelName(logger.getEffectiveLevel())}.")
    aggregate_strategy_logs(
        args.strategy_name, args.suffix_pattern, args.input_dir, args.output_dir
    )
