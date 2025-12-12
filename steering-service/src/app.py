import time
import os
import csv
import json
import logging
import argparse
import math

from flask import Flask, request, jsonify
from flask_cors import CORS

from dash_parser import DashParser
from monitor import ContainerMonitor
from strategies import (EpsilonGreedy, RandomSelector, NoSteeringSelector,
                      UCB1Selector, OracleBestChoiceSelector, D_UCB, LinUCBSelector)
from dynamic_latency_oracle import DynamicLatencyOracle

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

def calculate_haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    if None in [lat1, lon1, lat2, lon2]:
        return 0.0
    try:
        lat1_f, lon1_f, lat2_f, lon2_f = float(lat1), float(lon1), float(lat2), float(lon2)
    except (ValueError, TypeError):
        app_logger.warning(f"Invalid coordinates for Haversine: {(lat1, lon1, lat2, lon2)}")
        return 0.0
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1_f, lon1_f, lat2_f, lon2_f])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def setup_csv_logging(filename: str):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
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
            global selector_instance, active_log_filename, current_strategy_name, selector_initialized

            app_logger.info(f"Resetting simulation... Old Selector ID: {id(selector_instance)}")
            if hasattr(selector_instance, 'counts'):
                app_logger.info(f"Old Selector Counts: {selector_instance.counts}")

            active_log_filename = get_unique_log_filename(f"log_{current_strategy_name}", self.log_suffix, directory=LOG_DIR)
            setup_csv_logging(filename=active_log_filename)
            
            if latency_oracle and hasattr(latency_oracle, 'reset_events'):
                latency_oracle.reset_events()

            strategy_config = CONFIG.get('strategies', {}).get(current_strategy_name, {})
            
            if current_strategy_name == "epsilon_greedy":
                epsilon = strategy_config.get('epsilon', 0.1)
                selector_instance = EpsilonGreedy(epsilon=epsilon, counts={}, values={}, monitor=monitor, latency_oracle=latency_oracle)
            elif current_strategy_name == "no_steering":
                selector_instance = NoSteeringSelector(monitor=monitor, latency_oracle=latency_oracle)
            elif current_strategy_name == "random":
                selector_instance = RandomSelector(monitor=monitor, latency_oracle=latency_oracle)
            elif current_strategy_name == "ucb1":
                c = strategy_config.get('c', 2.0)
                selector_instance = UCB1Selector(c=c, monitor=monitor, latency_oracle=latency_oracle)
            elif current_strategy_name == "linucb":
                d = strategy_config.get('d', 5)
                alpha = strategy_config.get('alpha', 1.0)
                selector_instance = LinUCBSelector(d=d, alpha=alpha, monitor=monitor, latency_oracle=latency_oracle)
            elif current_strategy_name == "d_ucb":
                gamma_min = strategy_config.get('gamma_min', 0.1)
                gamma_max = strategy_config.get('gamma_max', 1.0)
                movement_weight = strategy_config.get('movement_weight', 0.4)
                latency_shock_weight = strategy_config.get('latency_shock_weight', 0.6)
                selector_instance = D_UCB(gamma_min=gamma_min, gamma_max=gamma_max, 
                                         movement_weight=movement_weight, latency_shock_weight=latency_shock_weight,
                                         monitor=monitor, latency_oracle=latency_oracle)
            elif current_strategy_name == "oracle_best_choice":
                selector_instance = OracleBestChoiceSelector(monitor=monitor, latency_oracle=latency_oracle)
            
            selector_initialized = False
            app_logger.info(f"Simulation reset complete. New Selector ID: {id(selector_instance)}")
            app_logger.info(f"New Log File: {active_log_filename}")
            
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
            global last_steering_main_server_decision, selector_instance, latency_oracle, active_log_filename
            global last_client_coords

            if not request.json: return "Invalid request: Missing JSON body", 400
            data = request.json
            s_t, lat, lon, rt_c, srv_u_feedback = (data.get(k) for k in ["time", "lat", "long", "rt", "server_used"])

            client_is_moving = False
            current_time_for_move_check = time.time()

            if lat is not None and lon is not None:
                if latency_oracle: latency_oracle.update_client_location(lat, lon)
                if last_client_coords['lat'] is not None and \
                   last_client_coords['lon'] is not None:
                    if (current_time_for_move_check - last_client_coords['time'] >= CLIENT_COORDS_UPDATE_INTERVAL_SEC):
                        dist_moved = calculate_haversine_distance(last_client_coords['lat'], last_client_coords['lon'], lat, lon)
                        if dist_moved > MOVEMENT_THRESHOLD_KM:
                            client_is_moving = True
                            app_logger.debug(f"Movement detected: {dist_moved:.3f} km")
                        last_client_coords['lat'], last_client_coords['lon'], last_client_coords['time'] = lat, lon, current_time_for_move_check
                elif last_client_coords['lat'] is None:
                    last_client_coords['lat'], last_client_coords['lon'], last_client_coords['time'] = lat, lon, current_time_for_move_check

            latency_shock_detected = False
            oracle_lat_for_feedback = None
            if srv_u_feedback and latency_oracle:
                 all_lats_temp = latency_oracle.get_all_current_latencies()
                 oracle_lat_for_feedback = all_lats_temp.get(srv_u_feedback, latency_oracle.get_current_latency(srv_u_feedback))

            current_gamma_val = None
            if isinstance(selector_instance, D_UCB):
                if srv_u_feedback and oracle_lat_for_feedback is not None:
                    if hasattr(selector_instance, '_check_latency_shock'):
                        latency_shock_detected = selector_instance._check_latency_shock(srv_u_feedback, oracle_lat_for_feedback)
                selector_instance.update_environmental_state(client_is_moving, latency_shock_detected)
                current_gamma_val = selector_instance.current_gamma

            all_oracle_lats_for_log = latency_oracle.get_all_current_latencies() if latency_oracle else {}
            all_srv_json = json.dumps(all_oracle_lats_for_log)

            counts_to_log = getattr(selector_instance, "counts", {})
            actual_counts_to_log = {}
            if hasattr(selector_instance, "real_counts"):
                 actual_counts_to_log = getattr(selector_instance, "real_counts", {})
            elif hasattr(selector_instance, "counts"):
                 actual_counts_to_log = getattr(selector_instance, "counts", {})

            log_base = {
                "timestamp_server": time.time(), "sim_time_client": s_t,
                "client_lat": lat, "client_lon": lon,
                "all_servers_oracle_latency_json": all_srv_json,
                "steering_decision_main_server": last_steering_main_server_decision,
                "rl_strategy": current_strategy_name,
                "rl_counts_json": json.dumps(counts_to_log),
                "rl_actual_counts_json": json.dumps(actual_counts_to_log),
                "rl_values_json": json.dumps(getattr(selector_instance, "values", {})),
                "gamma_value": current_gamma_val
            }

            if srv_u_feedback and rt_c is not None and latency_oracle:
                log_entry = {**log_base, "server_used_for_latency": srv_u_feedback,
                             "experienced_latency_ms_CLIENT": rt_c,
                             "experienced_latency_ms_ORACLE": oracle_lat_for_feedback,
                             "experienced_latency_ms": oracle_lat_for_feedback}
                
                if active_log_filename:
                    log_data_to_csv(log_entry, filename=active_log_filename)
                else:
                    app_logger.warning("Attempted to log data but no log file is active. Call /reset_simulation first.")

                if not self._initialize_selector_if_needed():
                    return "Service not ready (selector in /coords)", 503

                if hasattr(selector_instance, "update"):
                    if srv_u_feedback not in selector_instance.nodes:
                        app_logger.warning(f"Server {srv_u_feedback} not in nodes ({selector_instance.nodes}). Re-initializing...")
                        self._initialize_selector_if_needed()
                        if srv_u_feedback not in selector_instance.nodes:
                            app_logger.error(f"Server {srv_u_feedback} still not recognized. RL update not performed.")
                            return "Server not recognized, RL not updated.", 400
                    
                    feedback_value = float(oracle_lat_for_feedback) 
                    if isinstance(selector_instance, (UCB1Selector, D_UCB, LinUCBSelector)):
                        reward = 1000.0 / float(oracle_lat_for_feedback) if float(oracle_lat_for_feedback) > 0 else 0.0
                        feedback_value = reward

                    update_kwargs = {}
                    if isinstance(selector_instance, LinUCBSelector):
                        context_for_update = last_decision_contexts.get(srv_u_feedback)
                        if context_for_update is not None:
                            update_kwargs['context'] = context_for_update
                        else:
                            app_logger.warning(f"Contexto para {srv_u_feedback} não encontrado no último snapshot de decisão. Update do LinUCB pode ser impreciso.")

                    selector_instance.update(srv_u_feedback, feedback_value, **update_kwargs)
                    return "RL updated and logged", 200
                return "Data logged (no RL update)", 200
            elif lat is not None and lon is not None:
                if active_log_filename:
                    log_entry = {**log_base, "server_used_for_latency": srv_u_feedback,
                                 "experienced_latency_ms_CLIENT": rt_c,
                                 "experienced_latency_ms_ORACLE": None, "experienced_latency_ms": None}
                    log_data_to_csv(log_entry, filename=active_log_filename)
                return "Location data logged", 200
            else:
                app_logger.warning(f"Invalid or missing data in /coords: srv_u={srv_u_feedback}, rt_c={rt_c}, lat={lat}, lon={lon}")
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

    strategy_config = CONFIG.get('strategies', {}).get(args.strategy, {})
    
    if args.strategy == "epsilon_greedy":
        epsilon = strategy_config.get('epsilon', 0.1)
        selector_instance = EpsilonGreedy(epsilon=epsilon, counts={}, values={}, monitor=monitor, latency_oracle=latency_oracle)
    elif args.strategy == "no_steering":
        selector_instance = NoSteeringSelector(monitor=monitor, latency_oracle=latency_oracle)
    elif args.strategy == "random":
        selector_instance = RandomSelector(monitor=monitor, latency_oracle=latency_oracle)
    elif args.strategy == "ucb1":
        c = strategy_config.get('c', 2.0)
        selector_instance = UCB1Selector(c=c, monitor=monitor, latency_oracle=latency_oracle)
    elif args.strategy == "linucb":
        d = strategy_config.get('d', 5)
        alpha = strategy_config.get('alpha', 1.0)
        selector_instance = LinUCBSelector(d=d, alpha=alpha, monitor=monitor, latency_oracle=latency_oracle)
    elif args.strategy == "d_ucb":
        gamma_min = strategy_config.get('gamma_min', 0.1)
        gamma_max = strategy_config.get('gamma_max', 1.0)
        movement_weight = strategy_config.get('movement_weight', 0.4)
        latency_shock_weight = strategy_config.get('latency_shock_weight', 0.6)
        selector_instance = D_UCB(gamma_min=gamma_min, gamma_max=gamma_max,
                                 movement_weight=movement_weight, latency_shock_weight=latency_shock_weight,
                                 monitor=monitor, latency_oracle=latency_oracle)
    elif args.strategy == "oracle_best_choice":
        selector_instance = OracleBestChoiceSelector(monitor=monitor, latency_oracle=latency_oracle)
    else:
        app_logger.critical(f"Unknown strategy: {args.strategy}. Defaulting to EpsilonGreedy.")
        current_strategy_name = "epsilon_greedy"
        epsilon = CONFIG.get('strategies', {}).get('epsilon_greedy', {}).get('epsilon', 0.1)
        selector_instance = EpsilonGreedy(epsilon=epsilon, counts={}, values={}, monitor=monitor, latency_oracle=latency_oracle)

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