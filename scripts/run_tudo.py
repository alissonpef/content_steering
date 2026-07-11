import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml
from playwright.sync_api import sync_playwright

CONFIG_PATH = Path(__file__).parent / "experiment_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


def setup_logger(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"experiment_{timestamp}.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logger = logging.getLogger("content_steering")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_file)
    return logger


def _apply_scenario(page, scenario_cfg: dict, scenario_name: str) -> None:
    movement_target = scenario_cfg.get("movement_target", "none")
    page.select_option("id=simMovementTarget", movement_target)
    if movement_target != "none":
        page.fill("id=simMovementStartTime", str(scenario_cfg["movement_start_time"]))
        page.fill("id=simMovementDuration", str(scenario_cfg["movement_duration"]))
    spam_target_1 = scenario_cfg.get("spam_target_1", "none")
    page.select_option("id=simSpamTarget_1", spam_target_1)
    if spam_target_1 != "none":
        page.fill("id=simSpamStartTime_1", str(scenario_cfg["spam_start_time_1"]))
        page.fill("id=simSpamDuration_1", str(scenario_cfg["spam_duration_1"]))
    spam_target_2 = scenario_cfg.get("spam_target_2", "none")
    page.select_option("id=simSpamTarget_2", spam_target_2)
    if spam_target_2 != "none":
        page.fill("id=simSpamStartTime_2", str(scenario_cfg["spam_start_time_2"]))
        page.fill("id=simSpamDuration_2", str(scenario_cfg["spam_duration_2"]))


def _wait_for_simulation_end(
    page,
    logger: logging.Logger,
    completion_event: dict,
    sim_duration_s: int,
    timeout_grace_s: int,
    poll_ms: int,
) -> bool:
    hard_timeout_ms = (sim_duration_s + timeout_grace_s) * 1000
    elapsed_ms = 0
    while not completion_event["done"]:
        page.wait_for_timeout(poll_ms)
        elapsed_ms += poll_ms
        if elapsed_ms >= hard_timeout_ms:
            logger.warning(
                "Hard timeout reached after %ds — simulation did not emit completion signal. Run data may be incomplete.",
                sim_duration_s + timeout_grace_s,
            )
            return False
    completion_event["done"] = False
    return True


def _validate_log(log_path: Path, logger: logging.Logger, min_rows: int = 10) -> bool:
    if not log_path.exists():
        logger.warning("Expected log not found: %s", log_path)
        return False
    try:
        df = pd.read_csv(log_path)
        if len(df) < min_rows:
            logger.warning(
                "Log %s has only %d rows (expected >= %d). Simulation may have ended prematurely.",
                log_path.name,
                len(df),
                min_rows,
            )
            return False
        logger.debug("Log validated: %s (%d rows)", log_path.name, len(df))
        return True
    except Exception as exc:
        logger.warning("Could not parse log %s: %s", log_path, exc)
        return False


def run_all_experiments(cfg: dict, logger: logging.Logger) -> None:
    exp_cfg = cfg["experiment"]
    infra_cfg = cfg["infra"]
    scenarios_cfg = cfg["scenarios"]
    strategies = cfg["strategies"]
    num_runs = exp_cfg["num_runs"]
    sim_duration_s = exp_cfg["simulation_duration_seconds"]
    timeout_grace_s = exp_cfg["timeout_grace_seconds"]
    page_load_wait_ms = exp_cfg["page_load_wait_ms"]
    poll_ms = exp_cfg["completion_poll_interval_ms"]
    random_seed = exp_cfg["random_seed"]
    client_url = infra_cfg["client_url"]
    raw_log_dir = Path(cfg["output"]["logs_raw_dir"])
    logger.info("=" * 60)
    logger.info("PHASE 1 — Browser simulations")
    logger.info("Strategies : %s", strategies)
    logger.info("Scenarios  : %s", list(scenarios_cfg.keys()))
    logger.info("Runs/combo : %d", num_runs)
    logger.info("Seed       : %d", random_seed)
    logger.info("=" * 60)
    suspect_runs: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
            ],
        )
        for run_id in range(1, num_runs + 1):
            logger.info("--- RUN %d/%d ---", run_id, num_runs)
            for strategy in strategies:
                logger.info("[%d/%d] Strategy: %s", run_id, num_runs, strategy)
                for scenario_name, scenario_cfg in scenarios_cfg.items():
                    label = f"[{run_id}/{num_runs}][{strategy}][{scenario_name}]"
                    logger.info("%s Starting simulation...", label)
                    context = browser.new_context(ignore_https_errors=True)
                    page = context.new_page()
                    completion_event: dict = {"done": False}

                    def _on_console(msg, ev=completion_event, lbl=label):
                        text = msg.text
                        logger.debug("%s [browser:%s] %s", lbl, msg.type, text)
                        if "All runs completed." in text:
                            ev["done"] = True

                    page.on("console", _on_console)
                    log_filename = f"log_{strategy}_{scenario_name}_{run_id}.csv"

                    def handle_reset_route(
                        route,
                        strat=strategy,
                        scen=scenario_name,
                        rid=run_id,
                        fname=log_filename,
                        seed=random_seed,
                    ):
                        post_data = route.request.post_data
                        if post_data:
                            try:
                                data_dict = json.loads(post_data)
                                data_dict["log_filename"] = fname
                                data_dict["random_seed"] = seed
                                route.continue_(post_data=json.dumps(data_dict))
                            except json.JSONDecodeError:
                                route.continue_(post_data=post_data)
                        else:
                            route.continue_()

                    page.route("**/reset_simulation", handle_reset_route)
                    try:
                        page.goto(client_url)
                        page.click("#load-button")
                        page.wait_for_timeout(page_load_wait_ms)
                        page.select_option("id=simStrategy", strategy)
                        page.check('input[name="runMode"][value="duration"]')
                        page.fill("id=simDuration", str(sim_duration_s))
                        _apply_scenario(page, scenario_cfg, scenario_name)
                        page.click("id=button_StartControlledSim")
                        ended_cleanly = _wait_for_simulation_end(
                            page=page,
                            logger=logger,
                            completion_event=completion_event,
                            sim_duration_s=sim_duration_s,
                            timeout_grace_s=timeout_grace_s,
                            poll_ms=poll_ms,
                        )
                        if ended_cleanly:
                            logger.info(
                                "%s Simulation ended via completion signal.", label
                            )
                        else:
                            suspect_runs.append(log_filename)
                        page.wait_for_timeout(2000)
                        expected_log = raw_log_dir / log_filename
                        _validate_log(expected_log, logger)
                    except Exception as exc:
                        logger.error(
                            "%s Unexpected error: %s", label, exc, exc_info=True
                        )
                        suspect_runs.append(log_filename)
                    finally:
                        context.close()
        browser.close()
    logger.info("Phase 1 complete.")
    if suspect_runs:
        logger.warning(
            "%d suspect run(s) detected (timeout fallback or validation failure): %s",
            len(suspect_runs),
            suspect_runs,
        )
    else:
        logger.info("All runs ended cleanly via completion signal.")


def extract_logs(cfg: dict, logger: logging.Logger) -> None:
    infra_cfg = cfg["infra"]
    out_cfg = cfg["output"]
    label_selector = infra_cfg["kubernetes_label_selector"]
    namespace = infra_cfg["kubernetes_namespace"]
    remote_path = infra_cfg["pod_logs_remote_path"]
    logger.info("=" * 60)
    logger.info("PHASE 2 — Extracting logs from Kubernetes")
    logger.info("=" * 60)
    for d in [
        out_cfg["logs_raw_dir"],
        out_cfg["logs_aggregated_dir"],
        out_cfg["results_dir"],
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)
    try:
        pod_name = (
            subprocess.check_output(
                f"kubectl get pod -l {label_selector} -n {namespace} -o jsonpath='{{.items[0].metadata.name}}'",
                shell=True,
            )
            .decode("utf-8")
            .strip()
        )
        logger.info("Target pod: %s", pod_name)
        subprocess.run(
            f"kubectl cp {namespace}/{pod_name}:{remote_path} ./{out_cfg['logs_raw_dir']}/",
            shell=True,
            check=True,
        )
        logger.info("Logs copied successfully.")
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "kubectl cp failed (code %d). Logs may already be local.", exc.returncode
        )
    except Exception as exc:
        logger.warning("Could not copy logs from pod: %s", exc)


def run_analysis_scripts(cfg: dict, logger: logging.Logger) -> None:
    strategies = cfg["strategies"]
    scenarios = list(cfg["scenarios"].keys())
    logger.info("=" * 60)
    logger.info("PHASE 3 — Running analysis scripts")
    logger.info("=" * 60)
    logger.info("3.1 Aggregating logs per (strategy, scenario)...")
    for strategy in strategies:
        for scenario in scenarios:
            cmd = f"uv run python analysis/aggregate_logs.py {strategy} --suffix_pattern _{scenario}"
            logger.debug("CMD: %s", cmd)
            try:
                subprocess.run(
                    cmd, shell=True, check=True, capture_output=True, text=True
                )
                logger.info("  OK  aggregate %s / %s", strategy, scenario)
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "  FAIL aggregate %s / %s (code %d)\n%s",
                    strategy,
                    scenario,
                    exc.returncode,
                    exc.stderr,
                )
    scripts = [
        (
            "3.2 Server choice tables",
            "uv run python analysis/analyze_server_choices.py",
        ),
        (
            "3.3 Individual run graphs",
            "uv run python analysis/plotting/generate_graphs.py",
        ),
        (
            "3.4 Comparative graphs",
            "uv run python analysis/plotting/generate_compare_graphs.py",
        ),
        (
            "3.5 Aggregated system graphs",
            "uv run python analysis/plotting/generate_aggregated_graphs.py",
        ),
        (
            "3.6 Statistical boxplots",
            "uv run python analysis/plotting/generate_boxplots.py",
        ),
    ]
    for description, cmd in scripts:
        logger.info("%s", description)
        logger.debug("CMD: %s", cmd)
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            logger.info("  OK  %s", description)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "  FAIL %s (code %d)\n%s", description, exc.returncode, exc.stderr
            )
    logger.info("Phase 3 complete.")


def reorganize_results(cfg: dict, logger: logging.Logger) -> None:
    out_cfg = cfg["output"]
    source_dir = Path(out_cfg["results_dir"])
    target_dir = Path(out_cfg["experiments_results_dir"])
    scenarios = list(cfg["scenarios"].keys())
    logger.info("=" * 60)
    logger.info("PHASE 4 — Reorganizing results: %s → %s", source_dir, target_dir)
    logger.info("=" * 60)
    if not source_dir.exists():
        logger.warning("Source dir %s not found, skipping.", source_dir)
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = source_dir / "comparative_analysis"
    if comp_dir.exists():
        for scenario in scenarios:
            scenario_src = comp_dir / scenario
            if scenario_src.is_dir():
                scenario_target = target_dir / scenario / "comparacao_modelos"
                scenario_target.mkdir(parents=True, exist_ok=True)
                for file in scenario_src.iterdir():
                    shutil.copy(file, scenario_target / file.name)
    cons_dir = source_dir / "consolidated_charts"
    if cons_dir.exists():
        for run_dir in cons_dir.iterdir():
            name = run_dir.name
            if name.startswith("log_") and name.endswith("_average"):
                parts = name.split("_")
                if len(parts) >= 4:
                    scenario = parts[-2]
                    strategy = "_".join(parts[1:-2])
                    if scenario in scenarios and run_dir.is_dir():
                        dest = (
                            target_dir
                            / scenario
                            / "modelos_individuais"
                            / strategy
                            / "consolidado"
                        )
                        dest.mkdir(parents=True, exist_ok=True)
                        for file in run_dir.iterdir():
                            shutil.copy(file, dest / file.name)
    indiv_dir = source_dir / "individual_runs"
    if indiv_dir.exists():
        for run_dir in indiv_dir.iterdir():
            name = run_dir.name
            if name.startswith("log_") and "_average" not in name:
                parts = name.split("_")
                if len(parts) >= 4:
                    strategy = "_".join(parts[1:-2])
                    scenario = parts[-2]
                    if scenario in scenarios and run_dir.is_dir():
                        dest = target_dir / scenario / "modelos_individuais" / strategy
                        dest.mkdir(parents=True, exist_ok=True)
                        for file in run_dir.iterdir():
                            shutil.copy(file, dest / file.name)
    stats_dir = Path(out_cfg["logs_aggregated_dir"]) / "by_scenario"
    if stats_dir.exists():
        for file in stats_dir.iterdir():
            for scenario in scenarios:
                if file.name.endswith(f"_{scenario}.csv"):
                    dest = target_dir / scenario / "estatisticas"
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy(file, dest / file.name)
    global_stat = (
        Path(out_cfg["logs_aggregated_dir"]) / "dynamic_best_choice_accuracy.csv"
    )
    if global_stat.exists():
        for scenario in scenarios:
            dest = target_dir / scenario / "estatisticas"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(global_stat, dest / "dynamic_best_choice_accuracy_all.csv")
    logger.info("Phase 4 complete. Canonical results at: %s", target_dir)


def generate_report(cfg: dict, logger: logging.Logger) -> None:
    out_cfg = cfg["output"]
    agg_dir = Path(out_cfg["logs_aggregated_dir"]) / "by_scenario"
    logger.info("=" * 60)
    logger.info("PHASE 5 — Generating analise_cenarios.md")
    logger.info("=" * 60)
    md_lines = [
        "# Análise de Acurácia por Cenário\n",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        f"_Config: `{CONFIG_PATH.name}` — seed={cfg['experiment']['random_seed']}, runs={cfg['experiment']['num_runs']}_\n\n",
    ]
    for scenario in cfg["scenarios"]:
        csv_path = agg_dir / f"dynamic_best_choice_accuracy_{scenario}.csv"
        md_lines.append(f"## {scenario.capitalize()}\n")
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                md_lines.append(df.to_markdown(index=False))
                md_lines.append("\n\n")
            except Exception as exc:
                md_lines.append(f"_Error reading {csv_path.name}: {exc}_\n\n")
                logger.warning("Could not read %s: %s", csv_path, exc)
        else:
            md_lines.append(f"_File not found: `{csv_path.name}`_\n\n")
            logger.warning("Report CSV missing: %s", csv_path)
    report_path = Path("analise_cenarios.md")
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    cfg = load_config()
    logger = setup_logger(cfg["output"]["run_log_dir"])
    logger.info("Content Steering experiment pipeline starting.")
    logger.info("Config: %s", CONFIG_PATH)
    run_all_experiments(cfg, logger)
    extract_logs(cfg, logger)
    run_analysis_scripts(cfg, logger)
    reorganize_results(cfg, logger)
    generate_report(cfg, logger)
    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("Results: %s", cfg["output"]["experiments_results_dir"])
    logger.info("=" * 60)
