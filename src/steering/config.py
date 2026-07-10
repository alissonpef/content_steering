import os
import json
import logging

PROJECT_ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
LOG_DIR = os.getenv("STEERING_LOG_DIR", "/app/logs/raw")
CONFIG_PATH = os.path.join(PROJECT_ROOT_DIR, "config", "strategies.json")
try:
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = {}
STEERING_PORT = 30500
MOVEMENT_THRESHOLD_KM = CONFIG.get("simulation", {}).get("movement_threshold_km", 0.05)
CLIENT_COORDS_UPDATE_INTERVAL_SEC = CONFIG.get("simulation", {}).get(
    "client_coords_update_interval_sec", 0.9
)
app_logger = logging.getLogger("SteeringApp")
monitor_logger = logging.getLogger("KubernetesMonitor")
selector_strategies_logger = logging.getLogger("SelectorStrategies")


def _configure_all_loggers(default_level=logging.WARNING):
    loggers_to_configure = [app_logger, monitor_logger, selector_strategies_logger]
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    for logger_instance in loggers_to_configure:
        if not logger_instance.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger_instance.addHandler(handler)
        else:
            logger_instance.handlers[0].setFormatter(formatter)
        logger_instance.setLevel(default_level)
        logger_instance.propagate = False
