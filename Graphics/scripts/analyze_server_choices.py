import pandas as pd
import os
import re
import argparse
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.font_manager

logger = logging.getLogger("analyze_server_choices")

BASE_GRAPHICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_PROCESSED_LOGS_DIR = os.path.join(BASE_GRAPHICS_DIR, "data", "processed")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_GRAPHICS_DIR, "output", "analysis")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

STRATEGY_DISPLAY_NAMES = {
    "ucb1": "UCB1",
    "epsilon_greedy": "Epsilon Greedy",
    "random": "Random",
    "oracle_best_choice": "Optimal Strategy",
    "no_steering": "No Steering",
    "d_ucb": "D-UCB",
    "linucb": "LinUCB"
}
ACTUAL_CACHE_SERVER_NAMES_HYPHEN = [
    "video-streaming-cache-1", "video-streaming-cache-2", "video-streaming-cache-3"
]

def extract_strategy_name_from_filename(filename_no_ext: str) -> str:
    match = re.match(r"log_([a-zA-Z0-9_]+?)_average", filename_no_ext)
    if match:
        return match.group(1)
    logger.warning(f"Could not extract strategy name from '{filename_no_ext}'.")
    return "Unknown"

def dataframe_to_image(df: pd.DataFrame, output_image_path: str, title: str ="Server Choice Analysis"):
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['font.family'] = 'sans-serif'
    except Exception: pass

    num_cols, num_rows = len(df.columns), len(df)
    fig_width = max(8, num_cols * 2.2)
    fig_height = max(3, num_rows * 0.4 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight'); ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2C3E50')
        else:
            cell.set_facecolor('#ECF0F1' if row % 2 == 1 else 'white')
    
    plt.title(title, fontsize=16, y=0.95)
    
    try:
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        plt.savefig(output_image_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        logger.info(f"Table image saved: {output_image_path}")
    except Exception as e: 
        logger.error(f"Error saving table image: {e}")
    finally: 
        plt.close(fig)

def analyze_server_choices(logs_dir: str, output_csv_path: str = None, output_img_path: str = None):
    results = {}
    logger.info(f"Analyzing server choices in: {logs_dir}")
    if not os.path.isdir(logs_dir):
        logger.error(f"Logs directory not found: {logs_dir}")
        return

    for filename in sorted(os.listdir(logs_dir)):
        if not (filename.startswith("log_") and filename.endswith("_average.csv")):
            continue

        file_path = os.path.join(logs_dir, filename)
        logger.debug(f"Processing aggregated file: {filename}")
        try:
            df_log = pd.read_csv(file_path)
            required_cols = ['steering_decision_main_server', 'all_servers_oracle_latency_json']
            if df_log.empty or not all(col in df_log.columns for col in required_cols):
                logger.warning(f"Skipping {filename}: empty or missing essential columns.")
                continue

            filename_no_ext = os.path.splitext(filename)[0]
            strategy_key = extract_strategy_name_from_filename(filename_no_ext)
            if strategy_key == "Unknown":
                logger.warning(f"Could not determine strategy for {filename}. Skipping.")
                continue

            total_decisions = 0
            dynamic_best_choices = 0
            for index, row in df_log.iterrows():
                decision = row['steering_decision_main_server']
                json_str = row['all_servers_oracle_latency_json']
                if pd.isna(decision) or pd.isna(json_str) or "N/A" in str(decision):
                    continue
                
                total_decisions += 1
                try:
                    latencies = json.loads(json_str)
                    valid_latencies = {k.replace('_', '-'): v for k, v in latencies.items() 
                                       if k.replace('_', '-') in ACTUAL_CACHE_SERVER_NAMES_HYPHEN and isinstance(v, (int, float))}
                    if not valid_latencies:
                        continue
                    
                    dynamic_best_server = min(valid_latencies, key=valid_latencies.get)
                    if decision == dynamic_best_server:
                        dynamic_best_choices += 1
                except (json.JSONDecodeError, TypeError):
                    continue

            if total_decisions > 0:
                if strategy_key not in results:
                    results[strategy_key] = {'total_decisions': 0, 'dynamic_best_server_choices': 0}
                results[strategy_key]['total_decisions'] += total_decisions
                results[strategy_key]['dynamic_best_server_choices'] += dynamic_best_choices
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}", exc_info=False)

    if not results:
        logger.warning("No results processed. No table generated.")
        return

    table_data = []
    for strategy_key, data in results.items():
        total = data['total_decisions']
        best_choices = data['dynamic_best_server_choices']
        accuracy = (best_choices / total * 100) if total > 0 else 0
        display_name = STRATEGY_DISPLAY_NAMES.get(strategy_key, strategy_key.replace('_', ' ').title())
        
        table_data.append({
            'Strategy': display_name,
            'Correct Choices (#)': best_choices,
            'Total Decisions (#)': total,
            'Accuracy (%)': f"{accuracy:.1f}%"
        })

    df_results = pd.DataFrame(table_data)
    if not df_results.empty:
        df_results.sort_values(by='Strategy', inplace=True)
        logger.info("\nDynamic Best Server Choice Analysis:\n" + df_results.to_string(index=False))
        
        if output_csv_path:
            try:
                os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
                df_results.to_csv(output_csv_path, index=False)
                logger.info(f"Analysis table CSV saved: {output_csv_path}")
            except Exception as e: 
                logger.error(f"Error saving analysis CSV: {e}")
        
        if output_img_path:
            dataframe_to_image(df_results, output_img_path, title="Dynamic Best Server Choice Accuracy by Strategy")
    else:
        logger.warning("Resulting DataFrame is empty, no table to show or save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzes logs for dynamic best server choices.")
    parser.add_argument("--output_csv", type=str,
                        default=os.path.join(DEFAULT_PROCESSED_LOGS_DIR, "dynamic_best_choice_accuracy.csv"),
                        help="Path to save the CSV table.")
    parser.add_argument("--output_img", type=str,
                        default=os.path.join(DEFAULT_OUTPUT_DIR, "dynamic_best_choice_accuracy_table.png"),
                        help="Path to save the image of the table.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    _handler = logging.StreamHandler()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    _formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    if not logger.handlers:
        logger.addHandler(_handler)
    logger.setLevel(log_level)

    logger.info(f"Logging level set to {logging.getLevelName(logger.getEffectiveLevel())}.")
    
    analyze_server_choices(DEFAULT_PROCESSED_LOGS_DIR, args.output_csv, args.output_img)