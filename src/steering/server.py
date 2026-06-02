import time
import os
import json
import argparse
import asyncio
import logging
import numpy as np
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .dash_parser import DashParser
from .monitor import KubernetesMonitor
from .real_latency import get_all_latencies
from .strategies import (
    EpsilonGreedy,
    UCB1Selector,
    LinUCBSelector,
    ThompsonSamplingSelector,
    PPOHybridSelector,
    SACHybridSelector,
    RandomSelector,
    BestSelector,
)

from .config import CONFIG, STEERING_PORT, LOG_DIR, app_logger, _configure_all_loggers
from .logging_csv import setup_csv_logging, log_data_to_csv, get_unique_log_filename
from .context import (
    update_client_position,
    get_simple_context,
    update_spam_target,
    get_dynamic_penalty,
)

from typing import Any

AVAILABLE_STRATEGIES = [
    "epsilon_greedy",
    "ucb1",
    "linucb",
    "thompson_sampling",
    "ppo_hybrid",
    "sac_hybrid",
    "random",
    "best",
]

selector_instance: Any = None
selector_initialized = False
last_steering_main_server_decision = "N/A"
current_strategy_name = "N/A"
active_log_filename = None

last_real_latencies = {}
real_latency_history = {}
last_real_latency_probe_time = 0.0
last_decision_contexts_by_ip = {}


def _create_strategy_instance(strategy_name: str, monitor_ref):
    cfg = CONFIG.get("strategies", {}).get(strategy_name, {})
    constructors = {
        "epsilon_greedy": lambda: EpsilonGreedy(
            epsilon=cfg.get("epsilon", 0.2), counts={}, values={}, monitor=monitor_ref
        ),
        "ucb1": lambda: UCB1Selector(c=cfg.get("c", 1.0), monitor=monitor_ref),
        "linucb": lambda: LinUCBSelector(
            d=cfg.get("d", 14), alpha=cfg.get("alpha", 0.5), monitor=monitor_ref
        ),
        "thompson_sampling": lambda: ThompsonSamplingSelector(
            d=cfg.get("d", 14),
            alpha=cfg.get("alpha", 0.8),
            reward_scale=cfg.get("reward_scale", 10.0),
            prior_precision=cfg.get("prior_precision", 1.0),
            learning_rate=cfg.get("learning_rate", 0.75),
            update_steps=cfg.get("update_steps", 1),
            min_precision=cfg.get("min_precision", 1e-3),
            random_state=cfg.get("random_state"),
            monitor=monitor_ref,
        ),
        "ppo_hybrid": lambda: PPOHybridSelector(
            hidden_dim=cfg.get("hidden_dim", 64),
            learning_rate=cfg.get("learning_rate", 3e-4),
            gamma=cfg.get("gamma", 0.99),
            clip_ratio=cfg.get("clip_ratio", 0.2),
            entropy_coef=cfg.get("entropy_coef", 0.01),
            value_coef=cfg.get("value_coef", 0.5),
            batch_size=cfg.get("batch_size", 32),
            update_epochs=cfg.get("update_epochs", 4),
            reward_scale=cfg.get("reward_scale", 10.0),
            min_std=cfg.get("min_std", 0.1),
            max_std=cfg.get("max_std", 1.5),
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            random_state=cfg.get("random_state"),
            quality_levels=cfg.get("quality_levels"),
            policy_path=cfg.get("policy_path"),
            monitor=monitor_ref,
        ),
        "sac_hybrid": lambda: SACHybridSelector(
            hidden_dim=cfg.get("hidden_dim", 64),
            critic_hidden_dim=cfg.get("critic_hidden_dim", 64),
            actor_learning_rate=cfg.get("actor_learning_rate", 3e-4),
            critic_learning_rate=cfg.get("critic_learning_rate", 3e-4),
            gamma=cfg.get("gamma", 0.99),
            tau=cfg.get("tau", 0.02),
            entropy_coef=cfg.get("entropy_coef", 0.2),
            batch_size=cfg.get("batch_size", 32),
            replay_size=cfg.get("replay_size", 4096),
            update_steps=cfg.get("update_steps", 2),
            reward_scale=cfg.get("reward_scale", 10.0),
            min_std=cfg.get("min_std", 0.1),
            max_std=cfg.get("max_std", 1.5),
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            random_state=cfg.get("random_state"),
            quality_levels=cfg.get("quality_levels"),
            policy_path=cfg.get("policy_path"),
            monitor=monitor_ref,
        ),
        "random": lambda: RandomSelector(monitor=monitor_ref),
        "best": lambda: BestSelector(monitor=monitor_ref),
    }
    builder = constructors.get(strategy_name)
    if builder is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return builder()


dash_parser = DashParser()
monitor: Any = None

fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SteeringServer:
    def __init__(
        self,
        log_suffix: str = "",
        host_suffix: str = "",
        gateway_mode: bool = False,
        port: int = STEERING_PORT,
    ):
        self.log_suffix = log_suffix
        self.host_suffix = host_suffix
        self.gateway_mode = gateway_mode
        self.port = port
        self.app = fastapi_app

        app_logger.info("Server starting. Strategy will be selected from the UI.")
        self._register_routes()

    def _initialize_selector_if_needed(self) -> bool:
        global selector_initialized, selector_instance
        if selector_instance is None:
            return False
        if not selector_initialized or not selector_instance.nodes:
            nodes_info = monitor.get_nodes()
            if nodes_info:
                node_names = [info[0] for info in nodes_info if info and info[0]]
                if node_names:
                    selector_instance.initialize(node_names)
                    selector_initialized = True
                    app_logger.debug(
                        f"Selector initialized/updated with nodes: {node_names}"
                    )
                    return True
                app_logger.warning("No node names from monitor to initialize selector.")
            else:
                app_logger.warning("No node info from monitor to initialize selector.")
            return False
        return True

    def _register_routes(self):
        @self.app.get("/health")
        async def health_check():
            node_count = len(monitor.get_nodes()) if monitor else 0
            return JSONResponse(
                {
                    "status": "healthy",
                    "strategy": current_strategy_name,
                    "selector_initialized": selector_initialized,
                    "active_nodes": node_count,
                    "available_strategies": AVAILABLE_STRATEGIES,
                }
            )

        @self.app.get("/strategies")
        async def list_strategies():
            return JSONResponse(
                {
                    "strategies": AVAILABLE_STRATEGIES,
                    "current": current_strategy_name,
                }
            )

        @self.app.post("/reset_simulation")
        async def reset_simulation(request: Request):
            global selector_instance, active_log_filename, selector_initialized
            global last_real_latencies, real_latency_history, current_strategy_name
            app_logger.info(
                f"Resetting simulation... Old Selector ID: {id(selector_instance)}"
            )
            last_real_latencies = {}
            real_latency_history = {}
            try:
                data = await request.json()
            except Exception:
                data = {}

            requested_strategy = data.get("strategy")
            if requested_strategy and requested_strategy in AVAILABLE_STRATEGIES:
                current_strategy_name = requested_strategy
                app_logger.info(f"Strategy changed to: {current_strategy_name}")
            elif requested_strategy:
                return JSONResponse(
                    {
                        "error": f"Unknown strategy: {requested_strategy}",
                        "available": AVAILABLE_STRATEGIES,
                    },
                    status_code=400,
                )

            if current_strategy_name == "N/A":
                return JSONResponse(
                    {
                        "error": "No strategy selected",
                        "available": AVAILABLE_STRATEGIES,
                    },
                    status_code=400,
                )

            requested_subdir = data.get("log_subdir")
            requested_filename = data.get("log_filename")
            target_dir = LOG_DIR
            if requested_subdir:
                subdir = str(requested_subdir).strip().replace("\\", "/")
                subdir = os.path.normpath(subdir)
                if not os.path.isabs(subdir) and not subdir.startswith(".."):
                    target_dir = os.path.join(LOG_DIR, subdir)
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
            selector_instance = _create_strategy_instance(
                current_strategy_name, monitor
            )
            selector_initialized = False
            app_logger.info(
                f"Simulation reset. Strategy: {current_strategy_name}, "
                f"Log: {active_log_filename}"
            )
            return JSONResponse(
                {
                    "message": "Simulation reset",
                    "strategy": current_strategy_name,
                    "new_log": os.path.basename(active_log_filename),
                }
            )

        @self.app.post("/coords")
        async def coords_update(request: Request):
            try:
                data = await request.json()
            except Exception:
                return JSONResponse({"error": "Missing JSON body"}, status_code=400)
            client_ip = request.client.host if request.client else "unknown"
            s_t, lat, lon, rt_c, srv_u_fb, d_id, stall_t, spam_tgt = (
                data.get("time"),
                data.get("lat"),
                data.get("long"),
                data.get("rt"),
                data.get("server_used"),
                data.get("decision_id"),
                data.get("stall_time", 0),
                data.get("spam_target"),
            )
            update_client_position(lat, lon)
            update_spam_target(spam_tgt)
            log_base = self._build_log_base(s_t, lat, lon)
            if srv_u_fb and rt_c is not None:
                penalty = get_dynamic_penalty(
                    self._normalize_server_name(srv_u_fb), monitor
                )
                rt_c_with_penalty = rt_c + penalty
                self._record_real_latency(srv_u_fb, rt_c_with_penalty)
                return await self._handle_rl_feedback(
                    srv_u_fb,
                    rt_c_with_penalty,
                    log_base,
                    client_latency=rt_c_with_penalty,
                    client_ip=client_ip,
                    decision_id=d_id,
                    stall_time=stall_t,
                )
            elif lat is not None and lon is not None:
                return await self._handle_location_only(srv_u_fb, rt_c, log_base)
            else:
                return JSONResponse({"error": "Invalid data"}, status_code=400)

        @self.app.get("/sim_state")
        async def sim_state():
            oracle_latencies = await self._get_real_latency_snapshot(
                probe_if_empty=False
            )
            snapshot = {}
            if isinstance(selector_instance, PPOHybridSelector):
                snapshot["ppo_snapshot"] = selector_instance.policy_snapshot(
                    explore=False
                )
            elif isinstance(selector_instance, SACHybridSelector):
                snapshot["sac_snapshot"] = selector_instance.policy_snapshot(
                    explore=False
                )
            return JSONResponse(
                {
                    "latencies": oracle_latencies,
                    "decision": last_steering_main_server_decision,
                    "strategy": current_strategy_name,
                    **snapshot,
                }
            )

        @self.app.get("/{name:path}")
        @self.app.post("/{name:path}")
        async def do_remote_steering(name: str, request: Request):
            global last_steering_main_server_decision, last_decision_contexts_by_ip
            client_ip = request.client.host if request.client else "unknown"
            ordered_nodes = []
            decision_id = "N/A"

            if self._initialize_selector_if_needed():
                if isinstance(
                    selector_instance,
                    (
                        LinUCBSelector,
                        ThompsonSamplingSelector,
                        PPOHybridSelector,
                        SACHybridSelector,
                        BestSelector,
                    ),
                ):
                    node_names = [
                        info[0] for info in monitor.get_nodes() if info and info[0]
                    ]
                    contexts_for_decision, latencies_for_decision = {}, {}
                    for node_name in node_names:
                        latency = await self._get_real_latency_for_node(node_name)
                        history = real_latency_history.get(
                            self._normalize_server_name(node_name), [latency]
                        )
                        context = get_simple_context(
                            self._normalize_server_name(node_name),
                            latency,
                            history,
                            monitor,
                            selector_instance,
                            last_real_latencies,
                        )
                        contexts_for_decision[node_name] = context
                        latencies_for_decision[node_name] = latency
                    last_decision_contexts_by_ip[client_ip] = contexts_for_decision
                    decision_id = str(uuid.uuid4())
                    ordered_nodes = selector_instance.select_arm(
                        contexts=contexts_for_decision,
                        latencies=latencies_for_decision,
                        decision_id=decision_id,
                    )
                else:
                    decision_id = str(uuid.uuid4())
                    ordered_nodes = selector_instance.select_arm(
                        decision_id=decision_id
                    )

            last_steering_main_server_decision = (
                ordered_nodes[0] if ordered_nodes else "N/A_NO_NODES_FROM_SELECTION"
            )

            nodes_p = [(n, n) for n in ordered_nodes]
            uri_scheme = request.headers.get("X-Forwarded-Proto", request.url.scheme)
            service_host = request.headers.get("X-Forwarded-Host", request.url.netloc)
            forwarded_prefix = request.headers.get("X-Forwarded-Prefix", "")
            if forwarded_prefix and not forwarded_prefix.startswith("/"):
                forwarded_prefix = f"/{forwarded_prefix}"
            uri = f"{uri_scheme}://{service_host}{forwarded_prefix}"
            target = request.query_params.get("_DASH_pathway", "")

            resp = dash_parser.build(
                target=target,
                nodes=nodes_p,
                uri=uri,
                request=request,
                host_suffix=self.host_suffix,
                gateway_mode=self.gateway_mode,
                request_host=service_host,
            )
            resp["MEASURED-LATENCIES-MS"] = await self._get_real_latency_snapshot()
            resp["DECISION-ID"] = decision_id
            last_act = getattr(selector_instance, "last_action", None)
            if last_act is not None and isinstance(last_act, dict):
                if "quality_level" in last_act:
                    resp["RL-QUALITY-LEVEL"] = last_act["quality_level"]
            response = JSONResponse(resp)
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, max-age=0"
            )
            return response

    def run(self):
        s_dir = os.path.dirname(os.path.abspath(__file__))
        certs_dir = os.path.join(s_dir, "..", "..", "certs", "steering-service")
        cert = os.path.join(certs_dir, "steering-server.pem")
        key = os.path.join(certs_dir, "steering-server-key.pem")
        try:
            if not (os.path.exists(cert) and os.path.exists(key)):
                raise FileNotFoundError("SSL certificate/key not found.")
            app_logger.info(f"Starting HTTPS FastAPI service on port {self.port}...")
            uvicorn.run(
                self.app,
                host="0.0.0.0",
                port=self.port,
                ssl_certfile=cert,
                ssl_keyfile=key,
            )
        except Exception as e:
            app_logger.warning(f"Failed to start SSL: {e}. Falling back to HTTP.")
            uvicorn.run(self.app, host="0.0.0.0", port=self.port)

    @staticmethod
    def _normalize_server_name(server_name):
        if not server_name:
            return server_name
        name = str(server_name)
        if name.startswith("node") and name[4:].isdigit():
            return f"delivery-node-{name[4:]}"
        return name

    @staticmethod
    def _coerce_latency_ms(latency_value):
        try:
            latency = float(latency_value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(latency) or latency <= 0:
            return None
        return latency

    def _record_real_latency(self, server_name, latency_value):
        global last_real_latencies, real_latency_history
        normalized_name = self._normalize_server_name(server_name)
        latency = self._coerce_latency_ms(latency_value)
        if not normalized_name or normalized_name == "cloud" or latency is None:
            return

        latency = min(latency, 1000.0)

        prev_latency = last_real_latencies.get(normalized_name)
        if prev_latency is not None:
            smoothed = 0.25 * latency + 0.75 * prev_latency
        else:
            smoothed = latency

        last_real_latencies[normalized_name] = smoothed
        history = real_latency_history.setdefault(normalized_name, [])
        history.append(smoothed)
        if len(history) > 30:
            del history[:-30]

    async def _probe_real_latencies(self):
        global last_real_latencies, last_real_latency_probe_time
        now = time.time()
        if now - last_real_latency_probe_time < 2.0 and last_real_latencies:
            return dict(last_real_latencies)
        node_names = [info[0] for info in monitor.get_nodes() if info and info[0]]
        if not node_names:
            node_names = None
        probed = await asyncio.to_thread(
            get_all_latencies, nodes=node_names, timeout_seconds=0.75
        )
        last_real_latency_probe_time = now
        for node_name, latency in probed.items():
            if latency != 9999.0:
                penalty = get_dynamic_penalty(
                    self._normalize_server_name(node_name), monitor
                )
                latency_with_penalty = latency + penalty
                self._record_real_latency(node_name, latency_with_penalty)
                probed[node_name] = latency_with_penalty
        return dict(last_real_latencies) or probed

    async def _get_real_latency_snapshot(self, probe_if_empty=True):
        if last_real_latencies:
            return dict(last_real_latencies)
        if probe_if_empty:
            return await self._probe_real_latencies()
        return {}

    async def _get_real_latency_for_node(self, node_name):
        normalized_name = self._normalize_server_name(node_name)
        latency = last_real_latencies.get(normalized_name)
        if latency is not None:
            return latency
        probed = await self._probe_real_latencies()
        return probed.get(normalized_name, 50.0)

    @staticmethod
    def _build_log_base(s_t, lat, lon):
        counts = getattr(selector_instance, "counts", {})
        actual = getattr(selector_instance, "real_counts", counts)
        return {
            "timestamp_server": time.time(),
            "sim_time_client": s_t,
            "client_lat": lat,
            "client_lon": lon,
            "dynamic_best_server_latency": None,
            "all_servers_oracle_latency_json": "{}",
            "steering_decision_main_server": last_steering_main_server_decision,
            "rl_strategy": current_strategy_name,
            "rl_counts_json": json.dumps(counts),
            "rl_actual_counts_json": json.dumps(actual),
            "rl_values_json": json.dumps(getattr(selector_instance, "values", {})),
            "gamma_value": None,
        }

    async def _handle_rl_feedback(
        self,
        srv_name,
        feedback_latency,
        log_base,
        client_latency=None,
        client_ip=None,
        decision_id=None,
        stall_time=0,
    ):
        selector_srv_name = self._normalize_server_name(srv_name)
        log_entry = {
            **log_base,
            "server_used_for_latency": srv_name,
            "experienced_latency_ms_CLIENT": client_latency,
            "experienced_latency_ms_ORACLE": None,
            "experienced_latency_ms": feedback_latency,
            "stall_time_ms": stall_time,
        }
        oracle_latencies = await self._probe_real_latencies()
        if oracle_latencies:
            log_entry["all_servers_oracle_latency_json"] = json.dumps(oracle_latencies)
            best_server = min(oracle_latencies, key=lambda k: oracle_latencies[k])
            log_entry["experienced_latency_ms_ORACLE"] = oracle_latencies.get(
                self._normalize_server_name(srv_name)
            )
            log_entry["dynamic_best_server_latency"] = oracle_latencies[best_server]
        if active_log_filename:
            log_data_to_csv(log_entry, filename=active_log_filename)

        if not self._initialize_selector_if_needed():
            return JSONResponse({"error": "Service not ready"}, status_code=503)
        if not hasattr(selector_instance, "update"):
            return JSONResponse({"message": "Data logged"}, status_code=200)

        feedback_value = float(feedback_latency)
        effective_latency = feedback_value + float(stall_time) * 10.0

        if isinstance(
            selector_instance,
            (
                UCB1Selector,
                LinUCBSelector,
                EpsilonGreedy,
                ThompsonSamplingSelector,
                PPOHybridSelector,
                SACHybridSelector,
            ),
        ):
            feedback_value = (
                1000.0 / effective_latency if effective_latency > 0 else 0.0
            )

        update_kwargs = {}
        if decision_id:
            update_kwargs["decision_id"] = decision_id
        if isinstance(selector_instance, (LinUCBSelector, ThompsonSamplingSelector)):
            ctx = None
            if client_ip in last_decision_contexts_by_ip:
                ctx = last_decision_contexts_by_ip[client_ip].get(srv_name)
                if ctx is None:
                    ctx = last_decision_contexts_by_ip[client_ip].get(selector_srv_name)
            if ctx is not None:
                update_kwargs["context"] = ctx
            else:
                latency = await self._get_real_latency_for_node(selector_srv_name)
                history = real_latency_history.get(selector_srv_name, [latency])
                update_kwargs["context"] = get_simple_context(
                    selector_srv_name,
                    latency,
                    history,
                    monitor,
                    selector_instance,
                    last_real_latencies,
                )

        if isinstance(selector_instance, (PPOHybridSelector, SACHybridSelector)):
            update_kwargs["done"] = False
        await asyncio.to_thread(
            selector_instance.update, selector_srv_name, feedback_value, **update_kwargs
        )
        return JSONResponse({"message": "RL updated and logged"}, status_code=200)

    async def _handle_location_only(self, srv_name, rt_c, log_base):
        if active_log_filename:
            log_entry = {
                **log_base,
                "server_used_for_latency": srv_name,
                "experienced_latency_ms_CLIENT": rt_c,
                "experienced_latency_ms_ORACLE": None,
                "experienced_latency_ms": None,
            }
            oracle_latencies = await self._probe_real_latencies()
            if oracle_latencies:
                log_entry["all_servers_oracle_latency_json"] = json.dumps(
                    oracle_latencies
                )
                best_server = min(oracle_latencies, key=lambda k: oracle_latencies[k])
                log_entry["experienced_latency_ms_ORACLE"] = oracle_latencies.get(
                    self._normalize_server_name(srv_name)
                )
                log_entry["dynamic_best_server_latency"] = oracle_latencies[best_server]
            log_data_to_csv(log_entry, filename=active_log_filename)
        return JSONResponse({"message": "Location data logged"}, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content Steering Service with RL.")
    parser.add_argument("--log_suffix", type=str, default="")
    parser.add_argument("--host_suffix", type=str, default=".default.svc.cluster.local")
    parser.add_argument("--gateway-mode", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _configure_all_loggers(
        default_level=logging.DEBUG if args.verbose else logging.INFO
    )
    app_logger.info("Steering server starting. Strategy will be selected from the UI.")

    monitor_config = CONFIG.get("monitor", {})
    monitor = KubernetesMonitor(
        interval_seconds=monitor_config.get("interval_seconds", 2),
        namespace=monitor_config.get("namespace", "default"),
        label_selector=monitor_config.get(
            "kubernetes_label_selector", "app=delivery-node"
        ),
    )
    monitor.start_collecting()

    main_app = SteeringServer(
        log_suffix=args.log_suffix,
        host_suffix=args.host_suffix,
        gateway_mode=args.gateway_mode,
        port=args.port or STEERING_PORT,
    )
    main_app.run()
