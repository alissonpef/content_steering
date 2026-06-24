import os
import csv
import threading
from .config import LOG_DIR, app_logger

_csv_write_lock = threading.Lock()

CSV_HEADERS = [
    "timestamp_server",
    "sim_time_client",
    "client_lat",
    "client_lon",
    "server_used_for_latency",
    "experienced_latency_ms_CLIENT",
    "experienced_latency_ms_ORACLE",
    "experienced_latency_ms",
    "dynamic_best_server_latency",
    "all_servers_oracle_latency_json",
    "steering_decision_main_server",
    "rl_strategy",
    "rl_counts_json",
    "rl_actual_counts_json",
    "rl_values_json",
    "stall_time_ms",
]


def setup_csv_logging(filename: str):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode="w", newline="", buffering=1) as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)
            file.flush()
        app_logger.info(f"CSV log configured: {filename}")
    except Exception as e:
        app_logger.critical(
            f"Error setting up CSV log for {filename}: {e}", exc_info=True
        )


def log_data_to_csv(data_dict: dict, filename: str):
    row = [data_dict.get(h) for h in CSV_HEADERS]
    try:
        with _csv_write_lock:
            with open(filename, mode="a", newline="", buffering=1) as file:
                csv.writer(file).writerow(row)
                file.flush()
    except Exception as e:
        app_logger.error(f"Error writing to CSV {filename}: {e}", exc_info=True)


def get_unique_log_filename(
    base_name: str, user_suffix: str, directory: str = LOG_DIR
) -> str:
    full_base_with_suffix = f"{base_name}{user_suffix}"
    cnt = 1
    while True:
        numbered_filename = f"{full_base_with_suffix}_{cnt}.csv"
        numbered_path = os.path.join(directory, numbered_filename)
        if not os.path.exists(numbered_path):
            return numbered_path
        cnt += 1
