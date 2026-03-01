from __future__ import annotations
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional
import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STEERING_SRC_DIR = os.path.join(PROJECT_ROOT, "steering-service", "src")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")
PLOTTING_DIR = os.path.join(ANALYSIS_DIR, "plotting")
LOG_ROOT_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_RAW_DATA_DIR = os.path.join(LOG_ROOT_DIR, "raw_data")
LOG_AGGREGATED_DIR = os.path.join(LOG_ROOT_DIR, "aggregated_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RESULTS_INDIVIDUAL_DIR = os.path.join(RESULTS_DIR, "individual_runs")
RESULTS_CONSOLIDATED_DIR = os.path.join(RESULTS_DIR, "consolidated_charts")
RESULTS_COMPARATIVE_DIR = os.path.join(RESULTS_DIR, "comparative_analysis")
BASE_URL = "https://localhost:30500"
ALGORITHMS: list[str] = [
    "ucb1",
    "linucb",
    "epsilon_greedy",
    "oracle_best_choice",
    "random",
]

SCENARIO_DURATION = 300
DEFAULT_RUNS = 10
INIT_LAT, INIT_LON = -23.0, -47.0
CACHE_COORDS = {
    "video-streaming-cache-1": {"lat": -23.0, "lon": -47.0},
    "video-streaming-cache-2": {"lat": -33.0, "lon": -71.0},
    "video-streaming-cache-3": {"lat": 5.0, "lon": -74.0},
}
MOBILITY_START = 120
MOBILITY_END = 180
MOBILITY_TARGET = "video-streaming-cache-2"
SPAM_SERVER = "video-streaming-cache-1"
SPAM_START = 120
SPAM_DURATION = 60
SPAM_FACTOR = 10.0
EXTREME_SPAM_START = 120
EXTREME_SPAM_DURATION = 60
EXTREME_SPAM_FACTOR = 20.0
STARTUP_WAIT = 5
SERVICE_POLL_TIMEOUT = 30
TICK_INTERVAL = 1.0
INTER_RUN_PAUSE = 2
INTER_STRATEGY_PAUSE = 3

logger = logging.getLogger("ScenarioRunner")


@dataclass
class Scenario:
    name: str
    suffix: str
    raw_subdir: str
    duration: int = SCENARIO_DURATION
    mobility_target: Optional[str] = None
    mobility_start: int = 0
    mobility_end: int = 0
    spam_server: Optional[str] = None
    spam_start: int = 0
    spam_duration: int = 0
    spam_factor: float = 1.0


SCENARIOS: list[Scenario] = [
    Scenario(
        name="Baseline (Static & Stable)",
        suffix="_scenario1_baseline",
        raw_subdir="baseline",
    ),
    Scenario(
        name="Mobility (Spatial Non-Stationarity)",
        suffix="_scenario2_mobility",
        raw_subdir="mobility",
        mobility_target=MOBILITY_TARGET,
        mobility_start=MOBILITY_START,
        mobility_end=MOBILITY_END,
    ),
    Scenario(
        name="Latency Shock (Temporal Non-Stationarity)",
        suffix="_scenario3_spam",
        raw_subdir="spam",
        spam_server=SPAM_SERVER,
        spam_start=SPAM_START,
        spam_duration=SPAM_DURATION,
        spam_factor=SPAM_FACTOR,
    ),
    Scenario(
        name="Extreme Latency Shock (+1000x)",
        suffix="_scenario4_spam_extreme",
        raw_subdir="spam_extreme",
        spam_server=SPAM_SERVER,
        spam_start=EXTREME_SPAM_START,
        spam_duration=EXTREME_SPAM_DURATION,
        spam_factor=EXTREME_SPAM_FACTOR,
    ),
]


def _scenario_raw_dir(scenario: Scenario) -> str:
    return os.path.join(LOG_RAW_DATA_DIR, scenario.raw_subdir)


def _api(method: str, path: str, **kwargs) -> Optional[requests.Response]:
    url = f"{BASE_URL}{path}"
    kwargs.setdefault("verify", False)
    kwargs.setdefault("timeout", 10)
    try:
        return requests.request(method, url, **kwargs)
    except requests.exceptions.RequestException as exc:
        logger.debug(f"HTTP request failed for {method} {path}: {exc}")
        return None


def _wait_for_service(
    max_wait: int = SERVICE_POLL_TIMEOUT, expected_strategy: Optional[str] = None
) -> bool:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/sim_state", verify=False, timeout=3)
            if r.status_code == 200:
                if expected_strategy is None:
                    return True
                try:
                    active_strategy = r.json().get("strategy")
                except Exception:
                    active_strategy = None
                if active_strategy == expected_strategy:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    frac = current / total if total else 0
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {current}/{total}s ({frac * 100:.0f}%)"


def _run_scenario_tick_loop(
    scenario: Scenario, strategy: str, run_idx: int, num_runs: int
):
    r = _api(
        "POST",
        "/reset_simulation",
        json={
            "log_subdir": scenario.raw_subdir,
            "log_filename": f"log_{strategy}_{run_idx}.csv",
        },
    )
    if r is None or r.status_code != 200:
        logger.error(f"  /reset_simulation failed — skipping run {run_idx}")
        return False
    new_log = r.json().get("new_log", "")
    logger.info(f"  Reset OK → {new_log}")
    time.sleep(1)
    target = CACHE_COORDS.get(scenario.mobility_target or "")
    spam_sent = False
    lat, lon = INIT_LAT, INIT_LON
    duration = scenario.duration
    for tick in range(1, duration + 1):
        t0 = time.time()
        if target and scenario.mobility_start < scenario.mobility_end:
            if tick < scenario.mobility_start:
                lat, lon = INIT_LAT, INIT_LON
            elif tick <= scenario.mobility_end:
                frac = (tick - scenario.mobility_start) / (
                    scenario.mobility_end - scenario.mobility_start
                )
                lat = _lerp(INIT_LAT, target["lat"], frac)
                lon = _lerp(INIT_LON, target["lon"], frac)
            else:
                lat, lon = target["lat"], target["lon"]
        if scenario.spam_server and tick == scenario.spam_start and not spam_sent:
            spam_sent = True
            _api(
                "POST",
                "/latency_event",
                json={
                    "server_name": scenario.spam_server,
                    "factor": scenario.spam_factor,
                    "duration_seconds": scenario.spam_duration,
                },
            )
            logger.info(
                f"    ⚡ t={tick}: Latency event → {scenario.spam_server} "
                f"(×{scenario.spam_factor}, {scenario.spam_duration}s)"
            )
        chosen = None
        sr = _api("GET", "/steering?_DASH_pathway=cloud")
        if sr and sr.status_code == 200:
            try:
                prio = sr.json().get("PATHWAY-PRIORITY", [])
                for p in prio:
                    if p.startswith("video-streaming-cache"):
                        chosen = p
                        break
            except Exception:
                pass
        payload = {
            "time": tick,
            "lat": round(lat, 5),
            "long": round(lon, 5),
        }
        if chosen:
            payload["server_used"] = chosen
            payload["rt"] = 50
        _api("POST", "/coords", json=payload)
        if tick % 10 == 0 or tick == 1 or tick == duration:
            bar = _progress_bar(tick, duration)
            logger.info(
                f"    [{strategy.upper():>19}] {scenario.name:>42} "
                f"Run {run_idx}/{num_runs}  {bar}  "
                f"pos=({lat:+.1f},{lon:+.1f})  srv={chosen or 'N/A'}"
            )
        elapsed = time.time() - t0
        time.sleep(max(0, TICK_INTERVAL - elapsed))
    expected_path = os.path.join(
        _scenario_raw_dir(scenario), f"log_{strategy}_{run_idx}.csv"
    )
    if os.path.isfile(expected_path):
        logger.info(f"  Saved run log → {expected_path}")
    else:
        logger.warning(f"  Expected run log not found: {expected_path}")
    return True


def _start_service(strategy: str, log_suffix: str) -> Optional[subprocess.Popen]:
    try:
        subprocess.run(
            ["pkill", "-f", "steering-service/src/app.py"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        pass

    cmd = [
        sys.executable,
        os.path.join(STEERING_SRC_DIR, "app.py"),
        "--strategy",
        strategy,
        "--log_suffix",
        log_suffix,
    ]
    logger.info(f"  Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=STEERING_SRC_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    time.sleep(STARTUP_WAIT)
    if _wait_for_service(expected_strategy=strategy):
        logger.info("  Steering service ready.")
        return proc
    logger.error(
        f"  Steering service failed to start with expected strategy '{strategy}'."
    )
    _kill_service(proc)
    return None


def _kill_service(proc: subprocess.Popen):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=3)
        except Exception:
            pass


def _docker_up():
    script = os.path.join(PROJECT_ROOT, "starting_streaming.sh")
    if not os.path.isfile(script):
        logger.warning(
            "starting_streaming.sh not found — assuming containers already running."
        )
        return True
    logger.info("Starting Docker cache containers (may require sudo)…")
    try:
        subprocess.run(["sudo", "bash", script], check=True, timeout=60)
        logger.info("Containers started successfully.")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"starting_streaming.sh failed: {exc}")
        return False


def _docker_down():
    script = os.path.join(PROJECT_ROOT, "stop_streaming.sh")
    if not os.path.isfile(script):
        return
    logger.info("Stopping Docker cache containers…")
    try:
        subprocess.run(["sudo", "bash", script], check=True, timeout=30)
    except Exception as exc:
        logger.warning(f"stop_streaming.sh issue: {exc}")


def _docker_running() -> bool:
    for i in range(1, 4):
        name = f"video-streaming-cache-{i}"
        r = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"name={name}",
                "--filter",
                "status=running",
                "-q",
            ],
            capture_output=True,
            text=True,
        )
        if not r.stdout.strip():
            return False
    return True


def _run_analysis(strategies: list[str], scenarios: list[Scenario]):
    logger.info("\n" + "=" * 70)
    logger.info("  ANALYSIS PIPELINE")
    logger.info("=" * 70)
    py = sys.executable
    graphs_script = os.path.join(PLOTTING_DIR, "generate_graphs.py")
    agg_script = os.path.join(ANALYSIS_DIR, "aggregate_logs.py")
    agg_graphs = os.path.join(PLOTTING_DIR, "generate_aggregated_graphs.py")
    boxplots_script = os.path.join(PLOTTING_DIR, "generate_boxplots.py")
    compare_script = os.path.join(PLOTTING_DIR, "generate_compare_graphs.py")
    accuracy_script = os.path.join(ANALYSIS_DIR, "analyze_server_choices.py")
    logger.info("  [1/6] Individual run graphs…")
    for sc in scenarios:
        raw_dir = _scenario_raw_dir(sc)
        if not os.path.isdir(raw_dir):
            continue
        scenario_results_dir = os.path.join(RESULTS_INDIVIDUAL_DIR, sc.raw_subdir)
        for fn in sorted(os.listdir(raw_dir)):
            if fn.startswith("log_") and fn.endswith(".csv") and "_average" not in fn:
                _safe_run(
                    [
                        py,
                        graphs_script,
                        os.path.join(raw_dir, fn),
                        "--output_dir",
                        scenario_results_dir,
                    ],
                    label=fn,
                )
    logger.info("  [2/6] Aggregating logs…")
    for strat in strategies:
        for sc in scenarios:
            scenario_input_dir = _scenario_raw_dir(sc)
            cmd = [
                py,
                agg_script,
                strat,
                "--input_dir",
                scenario_input_dir,
                "--output_dir",
                LOG_AGGREGATED_DIR,
            ]
            ok = _safe_run(cmd, label=f"agg {strat} [{sc.raw_subdir}]")
            if not ok:
                continue
            default_output = os.path.join(LOG_AGGREGATED_DIR, f"log_{strat}_average.csv")
            scenario_output = os.path.join(
                LOG_AGGREGATED_DIR,
                f"log_{strat}{sc.suffix}_average.csv",
            )
            if os.path.exists(default_output):
                if os.path.exists(scenario_output):
                    os.remove(scenario_output)
                os.replace(default_output, scenario_output)
                logger.info(f"    ✓ saved {os.path.basename(scenario_output)}")
            else:
                logger.warning(
                    f"    ✗ expected aggregated output not found: {default_output}"
                )
    logger.info("  [3/6] Aggregated graphs…")
    for strat in strategies:
        for sc in scenarios:
            csv_name = f"log_{strat}{sc.suffix}_average.csv"
            csv_path = os.path.join(LOG_AGGREGATED_DIR, csv_name)
            if os.path.exists(csv_path):
                _safe_run(
                    [
                        py,
                        agg_graphs,
                        csv_path,
                        "--output_dir",
                        os.path.join(RESULTS_CONSOLIDATED_DIR, sc.raw_subdir),
                    ],
                    label=csv_name,
                )
    logger.info("  [4/6] Boxplots…")
    _safe_run([py, boxplots_script])
    logger.info("  [5/6] Comparison graphs…")
    _safe_run(
        [
            py,
            compare_script,
            "--agg_dir",
            LOG_AGGREGATED_DIR,
            "--output_dir",
            RESULTS_DIR,
        ]
    )
    logger.info("  [6/6] Server choice accuracy analysis…")
    _safe_run(
        [
            py,
            accuracy_script,
            "--logs_dir",
            LOG_AGGREGATED_DIR,
            "--output_csv",
            os.path.join(LOG_AGGREGATED_DIR, "dynamic_best_choice_accuracy.csv"),
            "--output_img",
            os.path.join(RESULTS_DIR, "analysis", "dynamic_best_choice_accuracy_table.png"),
            "--comparison_output_dir",
            RESULTS_COMPARATIVE_DIR,
        ]
    )
    logger.info("  Analysis pipeline complete.\n")


def _safe_run(cmd: list, label: str = "", timeout: int = 120) -> bool:
    try:
        subprocess.run(cmd, check=True, timeout=timeout, capture_output=True, text=True)
        if label:
            logger.info(f"    ✓ {label}")
        return True
    except subprocess.CalledProcessError as exc:
        tag = label or " ".join(cmd[-2:])
        logger.warning(f"    ✗ {tag}: {exc.stderr[:200] if exc.stderr else exc}")
        return False
    except Exception as exc:
        logger.warning(f"    ✗ {label}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Scenario-based experiment runner (4 scenarios × N algorithms × R runs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Scenarios:\n"
            "  Default: All scenarios run for configured duration.\n"
            "  1  Baseline       — static client, stable network (300 s)\n"
            "  2  Mobility       — client moves toward Cache 2 (Chile) from t=120s (2:00) to t=180s (3:00)\n"
            "  3  Latency Shock  — 15× spike on Cache 1 from t=120s (2:00) to t=180s (3:00), then recovery\n"
            "  4  Extreme Shock  — 1000× spike on Cache 1 (extreme stress test)\n"
        ),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=ALGORITHMS,
        help=f"Algorithms to test (default: {' '.join(ALGORITHMS)})",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help="Which scenarios to run (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Runs per (algorithm × scenario) (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=SCENARIO_DURATION,
        help=f"Duration per run in seconds (default: {SCENARIO_DURATION})",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip the post-experiment analysis pipeline",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Assume Docker containers are already running",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG-level logging",
    )
    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    selected_scenarios = [SCENARIOS[i - 1] for i in args.scenarios]
    for sc in selected_scenarios:
        sc.duration = args.duration
    total_runs = len(args.strategies) * len(selected_scenarios) * args.runs
    total_secs = total_runs * args.duration
    logger.info("=" * 70)
    logger.info("  Content Steering — Scenario-Based Experiment Runner")
    logger.info("=" * 70)
    logger.info(f"  Algorithms : {args.strategies}")
    logger.info(f"  Scenarios  : {[sc.name for sc in selected_scenarios]}")
    logger.info(f"  Runs/combo : {args.runs}")
    logger.info(f"  Duration   : {args.duration}s per run")
    logger.info(f"  Total runs : {total_runs}")
    logger.info(f"  Est. time  : ~{total_secs // 60}m {total_secs % 60}s (+ overhead)")
    logger.info("=" * 70 + "\n")
    os.makedirs(LOG_ROOT_DIR, exist_ok=True)
    os.makedirs(LOG_RAW_DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_INDIVIDUAL_DIR, exist_ok=True)
    for sc in selected_scenarios:
        os.makedirs(_scenario_raw_dir(sc), exist_ok=True)
    if not args.skip_docker:
        if not _docker_running():
            if not _docker_up():
                logger.error("Cannot start Docker containers. Aborting.")
                sys.exit(1)
        else:
            logger.info("Docker containers already running — skipping startup.\n")
    else:
        logger.info("--skip-docker: assuming containers are running.\n")
    successes: list[str] = []
    failures: list[str] = []
    for strat_idx, strategy in enumerate(args.strategies, 1):
        for sc_idx, scenario in enumerate(selected_scenarios, 1):
            tag = f"{strategy} / {scenario.name}"
            logger.info("─" * 70)
            logger.info(
                f"  [{strat_idx}/{len(args.strategies)}] Algorithm: {strategy.upper()}"
            )
            logger.info(
                f"  [{sc_idx}/{len(selected_scenarios)}] Scenario : {scenario.name}"
            )
            logger.info("─" * 70)
            proc = _start_service(strategy, scenario.suffix)
            if proc is None:
                failures.append(tag)
                continue
            try:
                all_ok = True
                for run_idx in range(1, args.runs + 1):
                    logger.info(f"\n  ▶ Run {run_idx}/{args.runs}")
                    ok = _run_scenario_tick_loop(scenario, strategy, run_idx, args.runs)
                    if not ok:
                        all_ok = False
                    if run_idx < args.runs:
                        time.sleep(INTER_RUN_PAUSE)
                if all_ok:
                    successes.append(tag)
                else:
                    failures.append(tag)
            finally:
                _kill_service(proc)
                logger.info(f"  Steering service stopped for {strategy}.\n")
            time.sleep(INTER_STRATEGY_PAUSE)
    logger.info("\n" + "=" * 70)
    logger.info("  EXPERIMENT BATCH COMPLETE")
    logger.info(f"  Succeeded : {len(successes)}/{total_runs // args.runs} combos")
    if failures:
        logger.warning(f"  Failed    : {failures}")
    logger.info("=" * 70)
    if not args.skip_analysis and successes:
        _run_analysis(args.strategies, selected_scenarios)
    elif args.skip_analysis:
        logger.info("Analysis skipped (--skip-analysis).")
    if not args.skip_docker:
        _docker_down()
    logger.info(
        "\nAll done. Check logs/raw_data/<scenario>/ for CSV data and results/ for figures."
    )


if __name__ == "__main__":
    main()
