import time
import os
import csv
import json
import logging
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from dash_parser import DashParser
from monitor import ContainerMonitor
from strategies import (EpsilonGreedy, RandomSelector, NoSteeringSelector,
                      UCB1Selector, OracleBestChoiceSelector, D_UCB, LinUCBSelector)
from dynamic_latency_oracle import DynamicLatencyOracle, calculate_haversine_distance
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)
STEERING_PORT = 30500
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "logs", "raw")
CSV_HEADERS = [
    "timestamp_server", "sim_time_client", "client_lat", "client_lon",
    "server_used_for_latency", "experienced_latency_ms_CLIENT",
    "experienced_latency_ms_ORACLE", "experienced_latency_ms",
    "all_servers_oracle_latency_json", "steering_decision_main_server",
    "rl_strategy", "rl_counts_json", "rl_actual_counts_json", "rl_values_json", "gamma_value"
]
selector_instance = None
selector_initialized = False
last_steering_main_server_decision = "N/A"
current_strategy_name = "N/A"
latency_oracle = None
active_log_filename = None
last_client_coords = {'lat': None, 'lon': None, 'time': 0}
MOVEMENT_THRESHOLD_KM = CONFIG.get('simulation', {}).get('movement_threshold_km', 0.05)
CLIENT_COORDS_UPDATE_INTERVAL_SEC = CONFIG.get('simulation', {}).get('client_coords_update_interval_sec', 0.9)
last_decision_contexts = {}
app_logger = logging.getLogger("SteeringApp")
oracle_logger = logging.getLogger("LatencyOracle")
monitor_logger = logging.getLogger("ContainerMonitor")
selector_strategies_logger = logging.getLogger("SelectorStrategies")
def _configure_all_loggers(default_level=logging.WARNING):
    loggers_to_configure = [app_logger, oracle_logger, monitor_logger, selector_strategies_logger]
    formatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
    for logger_instance in loggers_to_configure:
        if not logger_instance.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger_instance.addHandler(handler)
        else:
            logger_instance.handlers[0].setFormatter(formatter)
        logger_instance.setLevel(default_level)
        logger_instance.propagate = False
def _create_strategy_instance(strategy_name: str, monitor_ref, oracle_ref):
    cfg = CONFIG.get('strategies', {}).get(strategy_name, {})
    constructors = {
        "epsilon_greedy":    lambda: EpsilonGreedy(epsilon=cfg.get('epsilon', 0.3),
                                                   counts={}, values={},
                                                   monitor=monitor_ref, latency_oracle=oracle_ref),
        "no_steering":       lambda: NoSteeringSelector(monitor=monitor_ref, latency_oracle=oracle_ref),
        "random":            lambda: RandomSelector(monitor=monitor_ref, latency_oracle=oracle_ref),
        "ucb1":              lambda: UCB1Selector(c=cfg.get('c', 1.0),
                                                   monitor=monitor_ref, latency_oracle=oracle_ref),
        "linucb":            lambda: LinUCBSelector(d=cfg.get('d', 14), alpha=cfg.get('alpha', 0.5),
                                                     monitor=monitor_ref, latency_oracle=oracle_ref),
        "d_ucb":             lambda: D_UCB(monitor=monitor_ref, latency_oracle=oracle_ref),
        "oracle_best_choice": lambda: OracleBestChoiceSelector(monitor=monitor_ref, latency_oracle=oracle_ref),
    }
    builder = constructors.get(strategy_name)
    if builder is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return builder()
def setup_csv_logging(filename: str):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode="w", newline="", buffering=1) as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)
            file.flush()
            os.fsync(file.fileno())
        app_logger.info(f"CSV log configured: {filename}")
    except Exception as e:
        app_logger.critical(f"Error setting up CSV log for {filename}: {e}", exc_info=True)
def log_data_to_csv(data_dict: dict, filename: str):
    row = [data_dict.get(h) for h in CSV_HEADERS]
    try:
        with open(filename, mode="a", newline="", buffering=1) as file:
            csv.writer(file).writerow(row)
            file.flush()
            os.fsync(file.fileno())
    except Exception as e:
        app_logger.error(f"Error writing to CSV {filename}: {e}", exc_info=True)
def get_unique_log_filename(base_name: str, user_suffix: str, directory: str = LOG_DIR) -> str:
    full_base_with_suffix = f"{base_name}{user_suffix}"
    cnt = 1
    while True:
        numbered_filename = f"{full_base_with_suffix}_{cnt}.csv"
        numbered_path = os.path.join(directory, numbered_filename)
        if not os.path.exists(numbered_path):
            return numbered_path
        cnt += 1
class Main:
    def __init__(self, sel_inst, strategy_arg: str, log_file: str, log_suffix: str):
        global selector_instance, current_strategy_name, active_log_filename
        selector_instance, current_strategy_name, active_log_filename = sel_inst, strategy_arg, log_file
        self.log_suffix = log_suffix
        self.app = Flask(__name__)
        CORS(self.app)
        werkzeug_logger = logging.getLogger("werkzeug")
        if app_logger.getEffectiveLevel() > logging.INFO:
            werkzeug_logger.setLevel(logging.ERROR)
        else:
            werkzeug_logger.setLevel(logging.INFO)
        self._register_routes()
    def _initialize_selector_if_needed(self) -> bool:
        global selector_initialized, selector_instance
        if not selector_initialized or not selector_instance.nodes:
            nodes_info = monitor.getNodes()
            if nodes_info:
                node_names = [info[0] for info in nodes_info if info and info[0]]
                if node_names:
                    selector_instance.initialize(node_names)
                    selector_initialized = True
                    app_logger.debug(f"Selector initialized/updated with nodes: {node_names}")
                    return True
                else:
                    app_logger.warning("No node names from monitor to initialize selector.")
            else:
                app_logger.warning("No node info from monitor to initialize selector.")
            return False
        return True
    def _register_routes(self):
        @self.app.route("/reset_simulation", methods=["POST"])
        def reset_simulation():
            global selector_instance, active_log_filename, selector_initialized
            app_logger.info(f"Resetting simulation... Old Selector ID: {id(selector_instance)}")

            data = request.get_json(silent=True) or {}
            requested_subdir = data.get("log_subdir")
            requested_filename = data.get("log_filename")

            target_dir = LOG_DIR
            if requested_subdir:
                subdir = str(requested_subdir).strip().replace("\\", "/")
                subdir = os.path.normpath(subdir)
                if not os.path.isabs(subdir) and not subdir.startswith(".."):
                    target_dir = os.path.join(LOG_DIR, subdir)
                else:
                    app_logger.warning(f"Invalid log_subdir ignored: {requested_subdir}")

            if requested_filename:
                safe_filename = os.path.basename(str(requested_filename).strip())
                if not safe_filename.endswith(".csv"):
                    safe_filename += ".csv"
                active_log_filename = os.path.join(target_dir, safe_filename)
            else:
                active_log_filename = get_unique_log_filename(
                    f"log_{current_strategy_name}",
                    self.log_suffix,
                    directory=target_dir,
                )

            setup_csv_logging(filename=active_log_filename)
            if latency_oracle and hasattr(latency_oracle, 'reset_events'):
                latency_oracle.reset_events()
            selector_instance = _create_strategy_instance(current_strategy_name, monitor, latency_oracle)
            selector_initialized = False
            app_logger.info(f"Simulation reset. New log: {active_log_filename}")
            return jsonify({"message": "Simulation reset", "new_log": os.path.basename(active_log_filename)}), 200
        @self.app.route("/<path:name>", methods=["GET", "POST"])
        def do_remote_steering(name: str):
            global last_steering_main_server_decision, last_decision_contexts
            if not self._initialize_selector_if_needed():
                return jsonify({"error": "Service not ready (selector initialization failed)."}), 503
            ordered_nodes = []
            if isinstance(selector_instance, LinUCBSelector):
                node_names = [info[0] for info in monitor.getNodes() if info and info[0]]
                contexts_for_decision = {}
                for node_name in node_names:
                    context, _ = latency_oracle.get_context_and_final_latency(node_name)
                    contexts_for_decision[node_name] = context
                last_decision_contexts = contexts_for_decision
                ordered_nodes = selector_instance.select_arm(contexts=contexts_for_decision)
            else:
                ordered_nodes = selector_instance.select_arm()
            last_steering_main_server_decision = ordered_nodes[0] if ordered_nodes else "N/A_NO_NODES_FROM_SELECTION"
            if not ordered_nodes:
                app_logger.error("No server selected by strategy.")
                return jsonify({"error": "No selectable server"}), 503
            if latency_oracle and ordered_nodes:
                latency_oracle.track_server_selection(ordered_nodes[0])
            nodes_p = [(n, n) for n in ordered_nodes]
            uri_scheme = request.headers.get('X-Forwarded-Proto', request.scheme)
            service_host = request.headers.get('X-Forwarded-Host', request.host)
            uri = f"{uri_scheme}://{service_host}"
            target = request.args.get("_DASH_pathway", "", str)
            resp = dash_parser.build(target=target, nodes=nodes_p, uri=uri, request=request)
            return jsonify(resp), 200
        @self.app.route("/coords", methods=["POST"])
        def coords_update():
            global last_steering_main_server_decision, last_client_coords
            if not request.json:
                return "Invalid request: Missing JSON body", 400
            data = request.json
            s_t       = data.get("time")
            lat       = data.get("lat")
            lon       = data.get("long")
            rt_c      = data.get("rt")
            srv_u_fb  = data.get("server_used")
            client_is_moving = self._update_client_position(lat, lon)
            oracle_lat_fb = self._get_oracle_feedback_latency(srv_u_fb)
            gamma_val = self._update_ducb_env_state(srv_u_fb, oracle_lat_fb, client_is_moving)
            log_base = self._build_log_base(s_t, lat, lon, gamma_val)
            if srv_u_fb and rt_c is not None and latency_oracle and oracle_lat_fb is not None:
                return self._handle_rl_feedback(srv_u_fb, oracle_lat_fb, log_base)
            elif lat is not None and lon is not None:
                return self._handle_location_only(srv_u_fb, rt_c, log_base)
            else:
                app_logger.warning(f"Invalid /coords data: srv={srv_u_fb}, rt={rt_c}, lat={lat}")
                return "Invalid data: Location or critical info missing", 400
        @self.app.route("/latency_event", methods=["POST"])
        def latency_event_route():
            global latency_oracle
            if not request.json: return "Invalid request: Missing JSON body", 400
            data = request.json
            server, factor, duration = data.get("server_name"), data.get("factor", 2.0), data.get("duration_seconds", 10)
            app_logger.info(f"Latency Event Received: Server={server}, Factor={factor}, Duration={duration}s")
            if not server: return "Server name (server_name) missing", 400
            if not latency_oracle: return "Latency oracle not ready", 503
            try:
                latency_oracle.apply_event_modifier(server, float(factor), int(duration))
                return f"Latency event for {server} applied", 200
            except ValueError: return "Invalid format for factor or duration", 400
            except Exception as e:
                app_logger.error(f"Error in /latency_event: {e}", exc_info=True)
                return "Error applying event", 500
        @self.app.route("/sim_state", methods=["GET"])
        def sim_state():
            oracle_latencies = latency_oracle.get_all_current_latencies() if latency_oracle else {}
            return jsonify({
                "latencies": oracle_latencies,
                "decision": last_steering_main_server_decision,
                "strategy": current_strategy_name,
            }), 200
    def run(self):
        global current_strategy_name
        s_dir = os.path.dirname(os.path.abspath(__file__))
        certs_dir = os.path.join(s_dir, "..", "certs")
        cert, key = os.path.join(certs_dir, "steering-service.pem"), os.path.join(certs_dir, "steering-service-key.pem")
        try:
            if not (os.path.exists(cert) and os.path.exists(key)):
                raise FileNotFoundError("SSL certificate/key not found.")
            app_logger.info(f"Attempting to start HTTPS service (Strategy: {current_strategy_name}) on port {STEERING_PORT}...")
            self.app.run(host="0.0.0.0", port=STEERING_PORT, debug=False, ssl_context=(cert, key))
        except Exception as e:
            app_logger.warning(f"Failed to start SSL: {e}. Falling back to HTTP.")
            app_logger.info(f"Starting HTTP service (Strategy: {current_strategy_name}) on port {STEERING_PORT}...")
            self.app.run(host="0.0.0.0", port=STEERING_PORT, debug=False)
    @staticmethod
    def _update_client_position(lat, lon) -> bool:
        global last_client_coords
        client_is_moving = False
        now = time.time()
        if lat is None or lon is None:
            return False
        if latency_oracle:
            latency_oracle.update_client_location(lat, lon)
        if last_client_coords['lat'] is not None and last_client_coords['lon'] is not None:
            if now - last_client_coords['time'] >= CLIENT_COORDS_UPDATE_INTERVAL_SEC:
                dist = calculate_haversine_distance(
                    last_client_coords['lat'], last_client_coords['lon'], lat, lon)
                if dist > MOVEMENT_THRESHOLD_KM:
                    client_is_moving = True
                    app_logger.debug(f"Movement detected: {dist:.3f} km")
                last_client_coords['lat'] = lat
                last_client_coords['lon'] = lon
                last_client_coords['time'] = now
        elif last_client_coords['lat'] is None:
            last_client_coords['lat'] = lat
            last_client_coords['lon'] = lon
            last_client_coords['time'] = now
        return client_is_moving
    @staticmethod
    def _get_oracle_feedback_latency(srv_name):
        if not srv_name or not latency_oracle:
            return None
        all_lats = latency_oracle.get_all_current_latencies()
        return all_lats.get(srv_name)
    @staticmethod
    def _update_ducb_env_state(srv_name, oracle_lat, client_is_moving):
        if not isinstance(selector_instance, D_UCB):
            return None
        selector_instance.update_environmental_state(False, False)
        return selector_instance.current_gamma
    @staticmethod
    def _build_log_base(s_t, lat, lon, gamma_val):
        all_oracle_lats = latency_oracle.get_all_current_latencies() if latency_oracle else {}
        counts = getattr(selector_instance, "counts", {})
        actual = getattr(selector_instance, "real_counts", counts)
        return {
            "timestamp_server": time.time(),
            "sim_time_client": s_t,
            "client_lat": lat,
            "client_lon": lon,
            "all_servers_oracle_latency_json": json.dumps(all_oracle_lats),
            "steering_decision_main_server": last_steering_main_server_decision,
            "rl_strategy": current_strategy_name,
            "rl_counts_json": json.dumps(counts),
            "rl_actual_counts_json": json.dumps(actual),
            "rl_values_json": json.dumps(getattr(selector_instance, "values", {})),
            "gamma_value": gamma_val,
        }
    def _handle_rl_feedback(self, srv_name, oracle_lat, log_base):
        log_entry = {
            **log_base,
            "server_used_for_latency": srv_name,
            "experienced_latency_ms_CLIENT": request.json.get("rt"),
            "experienced_latency_ms_ORACLE": oracle_lat,
            "experienced_latency_ms": oracle_lat,
        }
        if active_log_filename:
            log_data_to_csv(log_entry, filename=active_log_filename)
        else:
            app_logger.warning("No active log file. Call /reset_simulation first.")
        if not self._initialize_selector_if_needed():
            return "Service not ready (selector in /coords)", 503
        if not hasattr(selector_instance, "update"):
            return "Data logged (no RL update)", 200
        if srv_name not in selector_instance.nodes:
            self._initialize_selector_if_needed()
            if srv_name not in selector_instance.nodes:
                app_logger.info(f"Server {srv_name} not in selector nodes; feedback logged without RL update.")
                return "Feedback logged (server not in selector nodes).", 200
        feedback_value = float(oracle_lat)
        if isinstance(selector_instance, (UCB1Selector, D_UCB, LinUCBSelector)):
            feedback_value = 1000.0 / float(oracle_lat) if float(oracle_lat) > 0 else 0.0
        update_kwargs = {}
        if isinstance(selector_instance, LinUCBSelector):
            ctx = last_decision_contexts.get(srv_name)
            if ctx is not None:
                update_kwargs['context'] = ctx
            else:
                app_logger.warning(f"Context snapshot missing for {srv_name}; LinUCB update may be imprecise.")
        selector_instance.update(srv_name, feedback_value, **update_kwargs)
        return "RL updated and logged", 200
    @staticmethod
    def _handle_location_only(srv_name, rt_c, log_base):
        if active_log_filename:
            log_entry = {
                **log_base,
                "server_used_for_latency": srv_name,
                "experienced_latency_ms_CLIENT": rt_c,
                "experienced_latency_ms_ORACLE": None,
                "experienced_latency_ms": None,
            }
            log_data_to_csv(log_entry, filename=active_log_filename)
        return "Location data logged", 200
dash_parser = DashParser()
monitor = ContainerMonitor()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content Steering Service with RL.")
    parser.add_argument("--strategy", type=str, default="epsilon_greedy",
                        choices=["epsilon_greedy", "no_steering", "random", "ucb1",
                                 "d_ucb", "oracle_best_choice", "linucb"],
                        help="Steering strategy.")
    parser.add_argument("--log_suffix", type=str, default="",
                        help="Optional suffix for CSV log filename (e.g., _testScenarioX).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enables DEBUG level logging.")
    args = parser.parse_args()
    log_level_to_set = logging.DEBUG if args.verbose else logging.WARNING
    _configure_all_loggers(default_level=log_level_to_set)
    app_logger.info(f"Logging level set to {logging.getLevelName(app_logger.getEffectiveLevel())}.")
    current_strategy_name = args.strategy
    app_logger.info(f"Selected strategy: {current_strategy_name}")
    active_log_filename = None
    app_logger.info("Log file will be created when simulation starts (via /reset_simulation).")
    app_logger.info("Starting container monitor...")
    monitor_config = CONFIG.get('monitor', {})
    monitor.start_collecting()
    app_logger.info("Initializing latency oracle...")
    oracle_config = CONFIG.get('oracle', {})
    oracle_interval = oracle_config.get('update_interval_seconds', 1)
    latency_oracle = DynamicLatencyOracle(monitor, update_interval_seconds=oracle_interval)
    latency_oracle.movement_smoothing_factor = oracle_config.get('movement_smoothing_factor', 0.3)
    app_logger.info(f"Oracle configured: smoothing_factor={latency_oracle.movement_smoothing_factor}")
    latency_oracle.start()
    app_logger.info("Briefly waiting for monitor and oracle to gather initial data...")
    time.sleep(max(monitor.interval if hasattr(monitor, 'interval') else 2, oracle_interval) + 1.0)
    selector_instance = _create_strategy_instance(args.strategy, monitor, latency_oracle)
    app_logger.info("Creating Flask application instance...")
    main_app = Main(selector_instance, current_strategy_name, active_log_filename, args.log_suffix)
    app_logger.info(f"Starting Flask service (Strategy: {current_strategy_name})...")
    try:
        main_app.run()
    except KeyboardInterrupt:
        app_logger.info("Service shutting down (Ctrl+C).")
    except Exception as e:
        app_logger.critical(f"Runtime error in main application: {e}", exc_info=True)
    finally:
        app_logger.info("Shutdown procedures...")
        if latency_oracle and hasattr(latency_oracle, 'stop') and callable(latency_oracle.stop):
            app_logger.info("Stopping latency oracle...")
            latency_oracle.stop()
        if monitor and hasattr(monitor, 'stop_collecting') and callable(monitor.stop_collecting):
            app_logger.info("Stopping container monitor...")
            monitor.stop_collecting()
        app_logger.info(f"Service ({current_strategy_name}) stopped.")
