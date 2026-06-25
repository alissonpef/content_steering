import time
import os
import json
import argparse
import asyncio
import logging
import numpy as np
import uuid

from contextlib import asynccontextmanager

from .network_emulator import NetworkEmulatorDaemon
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .dash_parser import DashParser
from .monitor import KubernetesMonitor
from .real_latency import get_all_latencies, warmup_nodes
from .strategies import (
    EpsilonGreedy,
    UCB1Selector,
    LinUCBSelector,
    ThompsonSamplingSelector,
    PPOHybridSelector,
    SACHybridSelector,
    RandomSelector,
    BestSelector,
    RoundRobin,
)

from .config import CONFIG, STEERING_PORT, LOG_DIR, app_logger, _configure_all_loggers
from .logging_csv import setup_csv_logging, log_data_to_csv, get_unique_log_filename
from .context import (
    update_client_position,
    get_simple_context,
    update_spam_target,
)

AVAILABLE_STRATEGIES = [
    "epsilon_greedy",
    "ucb1",
    "linucb",
    "thompson_sampling",
    "ppo_hybrid",
    "sac_hybrid",
    "random",
    "round_robin",
    "best",
]

_MAX_CONTEXT_CACHE = 200
_STALL_PENALTY_FACTOR = 100.0


def _create_strategy_instance(strategy_name: str, monitor_ref):
    cfg = CONFIG.get("strategies", {}).get(strategy_name, {})
    constructors = {
        "epsilon_greedy": lambda: EpsilonGreedy(
            epsilon=cfg.get("epsilon", 0.2), counts={}, values={}, gamma=cfg.get("gamma", 0.95), monitor=monitor_ref
        ),
        "ucb1": lambda: UCB1Selector(c=cfg.get("c", 1.0), gamma=cfg.get("gamma", 0.95), monitor=monitor_ref),
        "linucb": lambda: LinUCBSelector(
            d=cfg.get("d", 14), alpha=cfg.get("alpha", 0.5), gamma=cfg.get("gamma", 0.95), monitor=monitor_ref
        ),
        "thompson_sampling": lambda: ThompsonSamplingSelector(
            d=cfg.get("d", 14),
            alpha=cfg.get("alpha", 0.8),
            gamma=cfg.get("gamma", 0.95),
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
        "round_robin": lambda: RoundRobin(monitor=monitor_ref),
        "best": lambda: BestSelector(monitor=monitor_ref),
    }
    builder = constructors.get(strategy_name)
    if builder is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return builder()


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    if hasattr(app.state, "network_emulator"):
        app.state.network_emulator.start()
    yield


fastapi_app = FastAPI(lifespan=app_lifespan)
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
        monitor_ref,
        log_suffix: str = "",
        host_suffix: str = "",
        gateway_mode: bool = False,
        port: int = STEERING_PORT,
    ):
        self.monitor = monitor_ref
        self.log_suffix = log_suffix
        self.host_suffix = host_suffix
        self.gateway_mode = gateway_mode
        self.port = port
        self.app = fastapi_app
        self.selector_instance = None
        self.selector_initialized = False
        self.last_steering_main_server_decision = "N/A"
        self.current_strategy_name = "N/A"
        self.active_log_filename = None
        self.last_real_latencies = {}
        self.real_latency_history = {}
        self.last_real_latency_probe_time = 0.0
        self.last_decision_contexts = {}
        self._steering_lock = asyncio.Lock()
        self.dash_parser = DashParser()

        app_logger.info("Server starting. Strategy will be selected from the UI.")
        self._register_routes()

    def _initialize_selector_if_needed(self) -> bool:
        if self.selector_instance is None:
            return False
        if not self.selector_initialized or not self.selector_instance.nodes:
            nodes_info = self.monitor.get_nodes()
            if nodes_info:
                node_names = [info[0] for info in nodes_info if info and info[0]]
                if node_names:
                    self.selector_instance.initialize(node_names)
                    self.selector_initialized = True
                    app_logger.debug(
                        f"Selector initialized/updated with nodes: {node_names}"
                    )
                    return True
                app_logger.warning(
                    "No node names from self.monitor to initialize selector."
                )
            else:
                app_logger.warning(
                    "No node info from self.monitor to initialize selector."
                )
            return False
        return True

    def _register_routes(self):
        @self.app.get("/health")
        async def health_check():
            node_count = len(self.monitor.get_nodes()) if self.monitor else 0
            return JSONResponse(
                {
                    "status": "healthy",
                    "strategy": self.current_strategy_name,
                    "self.selector_initialized": self.selector_initialized,
                    "active_nodes": node_count,
                    "available_strategies": AVAILABLE_STRATEGIES,
                }
            )

        @self.app.get("/strategies")
        async def list_strategies():
            return JSONResponse(
                {
                    "strategies": AVAILABLE_STRATEGIES,
                    "current": self.current_strategy_name,
                }
            )

        @self.app.post("/reset_simulation")
        async def reset_simulation(request: Request):
            app_logger.info(
                f"Resetting simulation... Old Selector ID: {id(self.selector_instance)}"
            )
            self.last_real_latencies = {}
            self.real_latency_history = {}

            if self.monitor:
                nodes = [n for n, _ in self.monitor.get_nodes() if n]
                if nodes:
                    await asyncio.to_thread(warmup_nodes, nodes)

            try:
                data = await request.json()
            except Exception:
                from fastapi import HTTPException

                raise HTTPException(status_code=422, detail="Invalid JSON body")

            requested_strategy = data.get("strategy")
            if requested_strategy and requested_strategy in AVAILABLE_STRATEGIES:
                self.current_strategy_name = requested_strategy
                app_logger.info(f"Strategy changed to: {self.current_strategy_name}")
            elif requested_strategy:
                return JSONResponse(
                    {
                        "error": f"Unknown strategy: {requested_strategy}",
                        "available": AVAILABLE_STRATEGIES,
                    },
                    status_code=400,
                )

            if self.current_strategy_name == "N/A":
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
                self.active_log_filename = os.path.join(target_dir, safe_filename)
            else:
                self.active_log_filename = get_unique_log_filename(
                    f"log_{self.current_strategy_name}",
                    self.log_suffix,
                    directory=target_dir,
                )
            setup_csv_logging(filename=self.active_log_filename)
            self.selector_instance = _create_strategy_instance(
                self.current_strategy_name, self.monitor
            )
            self.selector_initialized = False
            app_logger.info(
                f"Simulation reset. Strategy: {self.current_strategy_name}, "
                f"Log: {self.active_log_filename}"
            )
            return JSONResponse(
                {
                    "message": "Simulation reset",
                    "strategy": self.current_strategy_name,
                    "new_log": os.path.basename(self.active_log_filename),
                }
            )

        @self.app.post("/coords")
        async def coords_update(request: Request):
            try:
                data = await request.json()
            except Exception:
                from fastapi import HTTPException

                raise HTTPException(status_code=422, detail="Invalid JSON body")
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
                return await self._handle_rl_feedback(
                    srv_u_fb,
                    rt_c,
                    log_base,
                    client_latency=rt_c,
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
            if isinstance(self.selector_instance, PPOHybridSelector):
                snapshot["ppo_snapshot"] = self.selector_instance.policy_snapshot(
                    explore=False
                )
            elif isinstance(self.selector_instance, SACHybridSelector):
                snapshot["sac_snapshot"] = self.selector_instance.policy_snapshot(
                    explore=False
                )
            return JSONResponse(
                {
                    "latencies": oracle_latencies,
                    "decision": self.last_steering_main_server_decision,
                    "strategy": self.current_strategy_name,
                    **snapshot,
                }
            )

        @self.app.get("/{name:path}")
        @self.app.post("/{name:path}")
        async def do_remote_steering(name: str, request: Request):
            ordered_nodes = []
            decision_id = "N/A"

            if self._initialize_selector_if_needed():
                assert self.selector_instance is not None
                if isinstance(
                    self.selector_instance,
                    (
                        LinUCBSelector,
                        ThompsonSamplingSelector,
                        PPOHybridSelector,
                        SACHybridSelector,
                        BestSelector,
                    ),
                ):
                    async with self._steering_lock:
                        node_names = [
                            info[0]
                            for info in self.monitor.get_nodes()
                            if info and info[0]
                        ]
                        contexts_for_decision, latencies_for_decision = {}, {}
                        for node_name in node_names:
                            latency = await self._get_real_latency_for_node(node_name)
                            history = self.real_latency_history.get(
                                self._normalize_server_name(node_name), [latency]
                            )
                            context = get_simple_context(
                                self._normalize_server_name(node_name),
                                latency,
                                history,
                                self.monitor,
                                self.selector_instance,
                                self.last_real_latencies,
                            )
                            contexts_for_decision[node_name] = context
                            latencies_for_decision[node_name] = latency
                        decision_id = str(uuid.uuid4())
                        ordered_nodes = self.selector_instance.select_arm(
                            contexts=contexts_for_decision,
                            latencies=latencies_for_decision,
                            decision_id=decision_id,
                        )
                        if len(self.last_decision_contexts) >= _MAX_CONTEXT_CACHE:
                            oldest_key = next(iter(self.last_decision_contexts))
                            del self.last_decision_contexts[oldest_key]
                        self.last_decision_contexts[decision_id] = contexts_for_decision
                else:
                    decision_id = str(uuid.uuid4())
                    ordered_nodes = self.selector_instance.select_arm(
                        decision_id=decision_id
                    )

            self.last_steering_main_server_decision = (
                ordered_nodes[0] if ordered_nodes else "N/A_NO_NODES_FROM_SELECTION"
            )

            nodes_p = [(n, n) for n in ordered_nodes]
            uri_scheme = request.headers.get("X-Forwarded-Proto", request.url.scheme)
            service_host = request.headers.get("X-Forwarded-Host", request.url.netloc)
            forwarded_prefix = request.headers.get("X-Forwarded-Prefix", "")
            if forwarded_prefix and not forwarded_prefix.startswith("/"):
                forwarded_prefix = f"/{forwarded_prefix}"
            uri = f"{uri_scheme}://{service_host}{forwarded_prefix}"
            target = (
                request.query_params.get("_DASH_pathway", "")
                .strip()
                .strip('"')
                .strip("'")
            )

            resp = self.dash_parser.build(
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
            last_act = getattr(self.selector_instance, "last_action", None)
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

    def _resolve_arm_name_for_selector(self, raw_name: str) -> str:
        """Return the arm name exactly as it is stored in the selector's self.nodes.

        The client may report a server as 'node1' while the monitor registers it as
        'delivery-node-1' (or vice-versa).  This method resolves the ambiguity by
        checking the selector's own node list, trying both forms.  If the selector is
        not yet initialised, it falls back to the raw name so callers can still log.
        """
        if not raw_name:
            return raw_name
        selector = self.selector_instance
        if selector is None or not getattr(selector, "nodes", None):
            return raw_name
        nodes = selector.nodes
        if raw_name in nodes:
            return raw_name
        normalized = self._normalize_server_name(raw_name)
        if normalized in nodes:
            return normalized
        return raw_name

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
        normalized_name = self._normalize_server_name(server_name)
        latency = self._coerce_latency_ms(latency_value)
        if not normalized_name or normalized_name == "cloud" or latency is None:
            return

        prev_latency = self.last_real_latencies.get(normalized_name)
        if prev_latency is not None:
            smoothed = 0.15 * latency + 0.85 * prev_latency
        else:
            smoothed = latency

        self.last_real_latencies[normalized_name] = smoothed
        history = self.real_latency_history.setdefault(normalized_name, [])
        history.append(smoothed)
        if len(history) > 30:
            del history[:-30]
        if server_name != normalized_name:
            self.last_real_latencies[server_name] = smoothed
            raw_history = self.real_latency_history.setdefault(server_name, [])
            raw_history.append(smoothed)
            if len(raw_history) > 30:
                del raw_history[:-30]

    async def _probe_real_latencies(self):
        now = time.time()
        if now - self.last_real_latency_probe_time < 2.0 and self.last_real_latencies:
            return dict(self.last_real_latencies)
        node_names = [info[0] for info in self.monitor.get_nodes() if info and info[0]]
        if not node_names:
            node_names = None
        probed = await asyncio.to_thread(
            get_all_latencies, nodes=node_names, timeout_seconds=0.75
        )
        self.last_real_latency_probe_time = now
        for node_name, latency in probed.items():
            if latency != 9999.0:
                self._record_real_latency(node_name, latency)
        return dict(self.last_real_latencies) or probed

    async def _get_real_latency_snapshot(self, probe_if_empty=True):
        if probe_if_empty or self.last_real_latencies:
            return await self._probe_real_latencies()
        return {}

    async def _get_real_latency_for_node(self, node_name):
        latency = self.last_real_latencies.get(node_name)
        if latency is not None:
            return latency
        normalized_name = self._normalize_server_name(node_name)
        latency = self.last_real_latencies.get(normalized_name)
        if latency is not None:
            return latency
        probed = await self._probe_real_latencies()
        return probed.get(normalized_name, probed.get(node_name, 50.0))

    def _build_log_base(self, s_t, lat, lon):
        counts = getattr(self.selector_instance, "counts", {})
        actual = getattr(self.selector_instance, "real_counts", counts)
        return {
            "timestamp_server": time.time(),
            "sim_time_client": s_t,
            "client_lat": lat,
            "client_lon": lon,
            "dynamic_best_server_latency": None,
            "all_servers_oracle_latency_json": "{}",
            "steering_decision_main_server": self.last_steering_main_server_decision,
            "rl_strategy": self.current_strategy_name,
            "rl_counts_json": json.dumps(counts),
            "rl_actual_counts_json": json.dumps(actual),
            "rl_values_json": json.dumps(getattr(self.selector_instance, "values", {})),
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
        selector_arm_name = self._resolve_arm_name_for_selector(srv_name)
        normalized_latency_name = self._normalize_server_name(srv_name)

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
            oracle_val = oracle_latencies.get(normalized_latency_name)
            if oracle_val is None:
                oracle_val = oracle_latencies.get(srv_name)
            log_entry["experienced_latency_ms_ORACLE"] = oracle_val
            log_entry["dynamic_best_server_latency"] = oracle_latencies[best_server]
        if self.active_log_filename:
            log_data_to_csv(log_entry, filename=self.active_log_filename)

        if not self._initialize_selector_if_needed():
            return JSONResponse({"error": "Service not ready"}, status_code=503)
        assert self.selector_instance is not None
        if not hasattr(self.selector_instance, "update"):
            return JSONResponse({"message": "Data logged"}, status_code=200)

        feedback_value = float(feedback_latency)
        effective_latency = feedback_value + float(stall_time) * _STALL_PENALTY_FACTOR

        if isinstance(
            self.selector_instance,
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
                10000.0 / effective_latency if effective_latency > 0 else 0.0
            )

        async with self._steering_lock:
            update_kwargs = {}
            if decision_id:
                update_kwargs["decision_id"] = decision_id
            if isinstance(
                self.selector_instance, (LinUCBSelector, ThompsonSamplingSelector)
            ):
                ctx = None
                if decision_id and decision_id in self.last_decision_contexts:
                    stored = self.last_decision_contexts.pop(decision_id, {})
                    ctx = (
                        stored.get(selector_arm_name)
                        or stored.get(srv_name)
                        or stored.get(normalized_latency_name)
                    )
                if ctx is not None:
                    update_kwargs["context"] = ctx
                else:
                    app_logger.warning(
                        f"[Server] Context not found for decision_id={decision_id}, "
                        "recomputing from current state."
                    )
                    latency = await self._get_real_latency_for_node(selector_arm_name)
                    history = (
                        self.real_latency_history.get(selector_arm_name)
                        or self.real_latency_history.get(normalized_latency_name)
                        or [latency]
                    )
                    update_kwargs["context"] = get_simple_context(
                        selector_arm_name,
                        latency,
                        history,
                        self.monitor,
                        self.selector_instance,
                        self.last_real_latencies,
                    )

            if isinstance(
                self.selector_instance, (PPOHybridSelector, SACHybridSelector)
            ):
                update_kwargs["done"] = True
            self.selector_instance.update(
                selector_arm_name, feedback_value, **update_kwargs
            )
        return JSONResponse({"message": "RL updated and logged"}, status_code=200)

    async def _handle_location_only(self, srv_name, rt_c, log_base):
        if self.active_log_filename:
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
            log_data_to_csv(log_entry, filename=self.active_log_filename)
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

    network_emulator = NetworkEmulatorDaemon(monitor=monitor)
    fastapi_app.state.network_emulator = network_emulator

    main_app = SteeringServer(
        monitor_ref=monitor,
        log_suffix=args.log_suffix,
        host_suffix=args.host_suffix,
        gateway_mode=args.gateway_mode,
        port=args.port or STEERING_PORT,
    )
    main_app.run()
