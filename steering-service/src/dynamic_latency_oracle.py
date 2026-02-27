import time
import random
import threading
import numpy as np
import logging
import math

logger = logging.getLogger("LatencyOracle")


def calculate_haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    if None in (lat1, lon1, lat2, lon2):
        return 0.0
    try:
        lat1_f, lon1_f = float(lat1), float(lon1)
        lat2_f, lon2_f = float(lat2), float(lon2)
    except (ValueError, TypeError):
        return 0.0
    dLat = math.radians(lat2_f - lat1_f)
    dLon = math.radians(lon2_f - lon1_f)
    lat1_rad = math.radians(lat1_f)
    lat2_rad = math.radians(lat2_f)
    a = (
        math.sin(dLat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dLon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class _OrnsteinUhlenbeckProcess:
    __slots__ = ("theta", "mu", "sigma", "state", "dt")

    def __init__(self, theta=0.15, mu=0.0, sigma=1.0, initial_state=0.0, dt=1.0):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.state = initial_state
        self.dt = dt

    def sample(self) -> float:
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * math.sqrt(
            self.dt
        ) * random.gauss(0, 1)
        self.state += dx
        return self.state

    def reset(self):
        self.state = self.mu


class _MicroBurstGenerator:
    __slots__ = (
        "mean_interval",
        "mean_duration",
        "intensity_range",
        "_next_burst_time",
        "_burst_end_time",
        "_current_intensity",
    )

    def __init__(
        self, mean_interval_sec=45.0, mean_duration_sec=1.5, intensity_range=(1.2, 2.5)
    ):
        self.mean_interval = mean_interval_sec
        self.mean_duration = mean_duration_sec
        self.intensity_range = intensity_range
        now = time.time()
        self._next_burst_time = now + random.expovariate(1.0 / mean_interval_sec)
        self._burst_end_time = 0.0
        self._current_intensity = 1.0

    def get_multiplier(self) -> float:
        now = time.time()
        if now < self._burst_end_time:
            remaining = max(
                0.0, (self._burst_end_time - now) / max(0.01, self.mean_duration)
            )
            return 1.0 + (self._current_intensity - 1.0) * min(1.0, remaining)
        if now >= self._next_burst_time:
            duration = max(0.2, min(random.expovariate(1.0 / self.mean_duration), 5.0))
            self._burst_end_time = now + duration
            self._current_intensity = random.uniform(*self.intensity_range)
            self._next_burst_time = (
                now + duration + random.expovariate(1.0 / self.mean_interval)
            )
            return self._current_intensity
        return 1.0

    def reset(self):
        now = time.time()
        self._next_burst_time = now + random.expovariate(1.0 / self.mean_interval)
        self._burst_end_time = 0.0
        self._current_intensity = 1.0


class _RouteFlappingSimulator:
    __slots__ = (
        "mean_interval",
        "recovery_duration",
        "_next_flap_time",
        "_flap_start_time",
        "_flap_end_time",
        "_flap_penalty_ms",
    )

    def __init__(self, mean_interval_sec=120.0, recovery_duration_sec=5.0):
        self.mean_interval = mean_interval_sec
        self.recovery_duration = recovery_duration_sec
        now = time.time()
        self._next_flap_time = now + random.expovariate(1.0 / mean_interval_sec)
        self._flap_start_time = 0.0
        self._flap_end_time = 0.0
        self._flap_penalty_ms = 0.0

    def get_penalty_ms(self) -> float:
        now = time.time()
        if now < self._flap_end_time:
            elapsed = now - self._flap_start_time
            tau = self.recovery_duration * 0.4
            decay = math.exp(-elapsed / tau)
            oscillation = 1.0 + 0.3 * math.sin(elapsed * 4.0)
            return self._flap_penalty_ms * decay * oscillation
        if now >= self._next_flap_time:
            self._flap_penalty_ms = random.uniform(5, 18)
            self._flap_start_time = now
            self._flap_end_time = now + self.recovery_duration
            self._next_flap_time = (
                now
                + self.recovery_duration
                + random.expovariate(1.0 / self.mean_interval)
            )
            return self._flap_penalty_ms
        return 0.0

    def reset(self):
        now = time.time()
        self._next_flap_time = now + random.expovariate(1.0 / self.mean_interval)
        self._flap_start_time = 0.0
        self._flap_end_time = 0.0
        self._flap_penalty_ms = 0.0


class _TCPConnectionSimulator:
    __slots__ = ("_last_request_times", "_connection_warm", "_cooldown_seconds")

    def __init__(self, cooldown_seconds=30.0):
        self._last_request_times = {}
        self._connection_warm = {}
        self._cooldown_seconds = cooldown_seconds

    def get_penalty_ms(self, server_name: str, base_rtt_ms: float) -> float:
        now = time.time()
        last_time = self._last_request_times.get(server_name, 0.0)
        was_warm = self._connection_warm.get(server_name, False)
        self._last_request_times[server_name] = now
        if not was_warm or (now - last_time) > self._cooldown_seconds:
            self._connection_warm[server_name] = True
            return base_rtt_ms * 0.5
        self._connection_warm[server_name] = True
        return 0.0

    def reset(self):
        self._last_request_times.clear()
        self._connection_warm.clear()


class DynamicLatencyOracle:
    SPEED_OF_LIGHT_FIBER_KMS = 200_000
    DEFAULT_INITIAL_CLIENT_LAT = -23.0
    DEFAULT_INITIAL_CLIENT_LON = -47.0
    _NORM_LATENCY_MS = 300.0
    _NORM_DISTANCE_KM = 12_000.0
    _NORM_JITTER_MS = 40.0
    _NORM_FLAP_MS = 80.0
    CONTEXT_DIM = 14
    DEFAULT_SERVER_COORDS = {
        "video-streaming-cache-1": {"lat": -23.0, "lon": -47.0},
        "video-streaming-cache-2": {"lat": -33.0, "lon": -71.0},
        "video-streaming-cache-3": {"lat": 5.0, "lon": -74.0},
    }

    def __init__(self, monitor, update_interval_seconds: int = 2):
        self.monitor = monitor
        self.server_latencies: dict[str, float] = {}
        self.server_base_latencies_config = {
            "video-streaming-cache-1": 52,
            "video-streaming-cache-2": 56,
            "video-streaming-cache-3": 72,
        }
        self.server_base_jitter_config = {
            "video-streaming-cache-1": 5,
            "video-streaming-cache-2": 6,
            "video-streaming-cache-3": 7,
        }
        self.server_base_packet_loss_config = {
            "video-streaming-cache-1": 0.002,
            "video-streaming-cache-2": 0.003,
            "video-streaming-cache-3": 0.005,
        }
        self.server_peering_quality = {
            "video-streaming-cache-1": 1.0,
            "video-streaming-cache-2": 1.30,
            "video-streaming-cache-3": 1.40,
        }
        self.server_geo_coords: dict = {}
        self.client_latitude = self.DEFAULT_INITIAL_CLIENT_LAT
        self.client_longitude = self.DEFAULT_INITIAL_CLIENT_LON
        self.server_event_modifiers: dict[str, tuple] = {}
        self.update_interval_seconds = max(0.5, update_interval_seconds)
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        self.min_simulated_latency = 45.0
        self.movement_smoothing_factor = 0.35
        self._jitter_processes: dict[str, _OrnsteinUhlenbeckProcess] = {}
        self._micro_burst_gens: dict[str, _MicroBurstGenerator] = {}
        self._route_flap_sims: dict[str, _RouteFlappingSimulator] = {}
        self._tcp_sim = _TCPConnectionSimulator(cooldown_seconds=30.0)
        self.latency_history: dict[str, list] = {}
        self.server_selection_counts: dict[str, int] = {}
        self.previous_latencies: dict[str, float] = {}
        self.burst_active = False
        self._burst_start_time = 0.0
        self.burst_end_time = 0.0
        self.burst_affected_servers: list = []
        self._burst_intensity = 2.5
        self._backbone_groups = {
            "south_america": [
                "video-streaming-cache-1",
                "video-streaming-cache-2",
                "video-streaming-cache-3",
            ]
        }
        self._backbone_congestion: dict[str, float] = {}
        self._last_diurnal_time = 0.0
        self._cached_diurnal_mult = 1.0
        self._update_server_geo_coordinates()

    def _update_server_geo_coordinates(self):
        coords = {}
        if self.monitor:
            monitor_coords = self.monitor.get_node_coordinates()
            if isinstance(monitor_coords, dict):
                coords.update(monitor_coords)
        for server_name, fallback in self.DEFAULT_SERVER_COORDS.items():
            current = coords.get(server_name, {})
            lat = current.get("lat") if isinstance(current, dict) else None
            lon = current.get("lon") if isinstance(current, dict) else None
            if lat is None or lon is None:
                coords[server_name] = fallback
        with self.lock:
            self.server_geo_coords = coords

    def _ensure_server_components(self, server_name: str):
        if server_name not in self._jitter_processes:
            base_jitter = self.server_base_jitter_config.get(server_name, 10)
            self._jitter_processes[server_name] = _OrnsteinUhlenbeckProcess(
                theta=0.15,
                mu=0.0,
                sigma=base_jitter * 0.6,
                dt=self.update_interval_seconds,
            )
        if server_name not in self._micro_burst_gens:
            base_lat = self.server_base_latencies_config.get(server_name, 30)
            burst_interval = max(20.0, 60.0 - base_lat * 0.3)
            self._micro_burst_gens[server_name] = _MicroBurstGenerator(
                mean_interval_sec=burst_interval,
                mean_duration_sec=1.5,
                intensity_range=(1.05, 1.35),
            )
        if server_name not in self._route_flap_sims:
            self._route_flap_sims[server_name] = _RouteFlappingSimulator(
                mean_interval_sec=120.0,
                recovery_duration_sec=5.0,
            )

    def _initialize_server_states(self):
        current_nodes_info = self.monitor.getNodes() if self.monitor else []
        if not current_nodes_info:
            return
        current_node_names = [
            info[0] for info in current_nodes_info if info and info[0]
        ]
        if not current_node_names:
            return
        self._update_server_geo_coordinates()
        with self.lock:
            for name in current_node_names:
                if name not in self.server_latencies:
                    initial = self.server_base_latencies_config.get(
                        name, random.uniform(10, 30)
                    )
                    self.server_latencies[name] = initial
                    self.server_event_modifiers[name] = (1.0, 0)
                    self._ensure_server_components(name)
                    logger.info(
                        f"Oracle: Server {name} initialised (base: {initial:.1f}ms)"
                    )
            stale = [n for n in self.server_latencies if n not in current_node_names]
            for name in stale:
                self.server_latencies.pop(name, None)
                self.server_event_modifiers.pop(name, None)
                self._jitter_processes.pop(name, None)
                self._micro_burst_gens.pop(name, None)
                self._route_flap_sims.pop(name, None)
                logger.info(f"Oracle: Server {name} removed.")

    def _get_diurnal_multiplier(self) -> float:
        now = time.time()
        if now - self._last_diurnal_time < 0.5:
            return self._cached_diurnal_mult
        t = time.localtime(now)
        hour_frac = t.tm_hour + t.tm_min / 60.0 + t.tm_sec / 3600.0
        t_rad = (hour_frac / 24.0) * 2.0 * math.pi
        primary = 0.12 * math.sin(t_rad - math.radians(300))
        secondary = 0.05 * math.sin(2.0 * t_rad - math.radians(135))
        tertiary = 0.02 * math.sin(3.0 * t_rad - math.radians(180))
        noise = random.gauss(0, 0.02)
        mult = max(0.80, min(1.25, 1.0 + primary + secondary + tertiary + noise))
        self._cached_diurnal_mult = mult
        self._last_diurnal_time = now
        return mult

    def _update_backbone_congestion(self):
        total_sel = max(1, sum(self.server_selection_counts.values()))
        for group, members in self._backbone_groups.items():
            group_load = sum(self.server_selection_counts.get(m, 0) for m in members)
            rho = group_load / total_sel
            if rho < 0.8:
                factor = 1.0 + 0.1 * rho
            else:
                factor = 1.0 + 0.5 / max(0.01, 1.0 - rho)
            self._backbone_congestion[group] = min(factor, 3.0)

    def _get_backbone_factor(self, server_name: str) -> float:
        for group, members in self._backbone_groups.items():
            if server_name in members:
                return self._backbone_congestion.get(group, 1.0)
        return 1.0

    def _compute_propagation_delay_ms(self, distance_km: float) -> float:
        if distance_km <= 0:
            return 0.0
        one_way_ms = (distance_km / self.SPEED_OF_LIGHT_FIBER_KMS) * 1000.0
        return 2.0 * one_way_ms * 1.10

    def _compute_distance_penalty_ms(self, distance_km: float) -> float:
        if distance_km <= 0:
            return 0.0
        linear = 0.025 * distance_km
        quadratic = 0.0000015 * (distance_km**2)
        return linear + quadratic

    def _compute_proximity_bonus_ms(self, distance_km: float) -> float:
        if distance_km >= 800.0:
            return 0.0
        closeness = 1.0 - (distance_km / 800.0)
        return 6.0 * max(0.0, closeness)

    def _compute_retransmission_penalty(
        self, base_rtt_ms: float, loss_rate: float
    ) -> float:
        if loss_rate <= 0 or base_rtt_ms <= 0:
            return 0.0
        penalty = 0.0
        backoff = 1.0
        for _ in range(10):
            if random.random() < loss_rate:
                penalty += base_rtt_ms * backoff
                backoff = min(backoff * 1.5, 4.0)
        return penalty

    def _compute_queue_delay_ms(self, server_name: str) -> float:
        total = max(1, sum(self.server_selection_counts.values()))
        rho = min(0.95, self.server_selection_counts.get(server_name, 0) / total)
        if rho < 0.1:
            return 0.0
        return (rho / max(0.05, 1.0 - rho)) * 5.0

    def _get_burst_multiplier(self, server_name: str) -> float:
        now = time.time()
        if self.burst_active and now < self.burst_end_time:
            if server_name in self.burst_affected_servers:
                total_dur = max(1.0, self.burst_end_time - self._burst_start_time)
                remaining_ratio = max(0.0, (self.burst_end_time - now) / total_dur)
                return 1.0 + (self._burst_intensity - 1.0) * remaining_ratio
        elif self.burst_active:
            self.burst_active = False
        return 1.0

    def _apply_movement_smoothing(self, server_name: str, raw: float) -> float:
        prev = self.previous_latencies.get(server_name)
        if prev is None:
            self.previous_latencies[server_name] = raw
            return raw
        alpha = self.movement_smoothing_factor
        delta = abs(raw - prev)
        if delta > 60:
            alpha *= 0.2
        elif delta > 30:
            alpha *= 0.5
        smoothed = alpha * prev + (1.0 - alpha) * raw
        self.previous_latencies[server_name] = max(self.min_simulated_latency, smoothed)
        return max(self.min_simulated_latency, smoothed)

    def _compute_latency_internal(self, server_name: str):
        self._ensure_server_components(server_name)
        base_latency = self.server_base_latencies_config.get(server_name, 30)
        mod_factor, mod_expiry = self.server_event_modifiers.get(server_name, (1.0, 0))
        if mod_expiry != 0 and time.time() >= mod_expiry:
            mod_factor = 1.0
            if self.server_event_modifiers.get(server_name) != (1.0, 0):
                self.server_event_modifiers[server_name] = (1.0, 0)
        diurnal_mult = self._get_diurnal_multiplier()
        burst_mult = self._get_burst_multiplier(server_name)
        micro_burst_mult = self._micro_burst_gens[server_name].get_multiplier()
        combined_mod = min(
            4.0, mod_factor * diurnal_mult * burst_mult * micro_burst_mult
        )
        distance_km = 0.0
        if self.client_latitude is not None and self.client_longitude is not None:
            sc = self.server_geo_coords.get(server_name)
            if sc and sc.get("lat") is not None and sc.get("lon") is not None:
                distance_km = calculate_haversine_distance(
                    self.client_latitude,
                    self.client_longitude,
                    sc["lat"],
                    sc["lon"],
                )
        propagation_ms = self._compute_propagation_delay_ms(distance_km)
        distance_penalty_ms = self._compute_distance_penalty_ms(distance_km)
        proximity_bonus_ms = self._compute_proximity_bonus_ms(distance_km)
        processing_noise = random.gauss(0, max(1, base_latency) * 0.05)
        server_load = max(
            self.min_simulated_latency, (base_latency + processing_noise) * combined_mod
        )
        ou_sample = self._jitter_processes[server_name].sample()
        jitter_amp = 1.0 + (combined_mod - 1.0) * 0.4
        dist_jitter = 1.0 + (distance_km / 2000.0) * 0.15
        jitter_mag = abs(ou_sample) * jitter_amp * dist_jitter
        queue_ms = self._compute_queue_delay_ms(server_name)
        base_rtt = propagation_ms + base_latency
        tcp_ms = self._tcp_sim.get_penalty_ms(server_name, base_rtt)
        flap_ms = self._route_flap_sims[server_name].get_penalty_ms()
        backbone_f = self._get_backbone_factor(server_name)
        base_loss = self.server_base_packet_loss_config.get(server_name, 0.01)
        eff_loss = min(0.3, base_loss * combined_mod * backbone_f)
        retx_ms = self._compute_retransmission_penalty(base_rtt, eff_loss)
        latency_estimate = (
            propagation_ms + distance_penalty_ms + server_load * backbone_f
        )
        hist = self.latency_history.setdefault(server_name, [])
        hist.append(latency_estimate)
        if len(hist) > 10:
            del hist[:-10]
        t = time.localtime()
        time_of_day = (t.tm_hour + t.tm_min / 60.0) / 24.0
        total_sel = max(1, sum(self.server_selection_counts.values()))
        popularity = self.server_selection_counts.get(server_name, 0) / total_sel
        congestion = min(1.0, max(0.0, (diurnal_mult - 0.6) / 0.9))
        micro_burst_indicator = min(1.0, max(0.0, (micro_burst_mult - 1.0) / 2.5))
        route_instability = min(1.0, flap_ms / self._NORM_FLAP_MS)
        tcp_warm = self._tcp_sim._connection_warm.get(server_name, False)
        if len(hist) >= 2:
            recent_avg = float(np.mean(hist[-5:]))
        else:
            recent_avg = latency_estimate
        norm_recent_avg = min(1.0, recent_avg / self._NORM_LATENCY_MS)

        if len(hist) >= 4:
            mid = len(hist) // 2
            old_half = float(np.mean(hist[:mid]))
            new_half = float(np.mean(hist[mid:]))
            trend_raw = (new_half - old_half) / max(1.0, old_half)
            latency_trend = max(-1.0, min(1.0, trend_raw))
        else:
            latency_trend = 0.0
        context_vector = np.array(
            [
                1.0,
                min(1.0, propagation_ms / self._NORM_LATENCY_MS),
                min(1.0, distance_km / self._NORM_DISTANCE_KM),
                min(1.0, server_load / self._NORM_LATENCY_MS),
                min(1.0, jitter_mag / self._NORM_JITTER_MS),
                min(1.0, eff_loss * 10.0),
                time_of_day,
                congestion,
                micro_burst_indicator,
                route_instability,
                1.0 if tcp_warm else 0.0,
                popularity,
                norm_recent_avg,
                latency_trend,
            ]
        )
        peering_q = self.server_peering_quality.get(server_name, 1.0)
        jitter_sign = random.choice([-0.3, 0.5, 0.7, 1.0])
        raw_latency = (
            propagation_ms
            + distance_penalty_ms * peering_q
            + server_load * backbone_f
            + jitter_mag * jitter_sign
            + queue_ms
            + tcp_ms
            + flap_ms
            + retx_ms
            - proximity_bonus_ms
        )
        raw_latency = max(self.min_simulated_latency, raw_latency)
        final = self._apply_movement_smoothing(server_name, raw_latency)
        return context_vector, final

    def _update_latencies(self):
        self._initialize_server_states()
        with self.lock:
            self._update_backbone_congestion()
            for name in list(self.server_latencies):
                _, lat = self._compute_latency_internal(name)
                self.server_latencies[name] = lat
                logger.debug(f"Oracle tick: {name} → {lat:.1f}ms")

    def update_client_location(self, lat: float, lon: float):
        if lat is None or lon is None:
            return
        with self.lock:
            try:
                new_lat, new_lon = float(lat), float(lon)
                if new_lat != self.client_latitude or new_lon != self.client_longitude:
                    self.client_latitude = new_lat
                    self.client_longitude = new_lon
                    logger.debug(f"Oracle: Client → ({new_lat}, {new_lon})")
            except (ValueError, TypeError):
                logger.warning(f"Oracle: Invalid coords lat={lat}, lon={lon}")

    def get_context_and_final_latency(self, server_name: str):
        with self.lock:
            if server_name not in self.server_latencies:
                self._initialize_server_states()
            self._ensure_server_components(server_name)
            return self._compute_latency_internal(server_name)

    def get_current_latency(self, server_name: str) -> float:
        with self.lock:
            lat = self.server_latencies.get(server_name)
            if lat is None:
                logger.warning(f"Oracle: Latency unknown for {server_name}")
                return random.uniform(50, 150)
            return lat

    def get_all_current_latencies(self) -> dict:
        with self.lock:
            if not self.server_latencies and self.monitor and self.monitor.getNodes():
                self._initialize_server_states()
            return dict(self.server_latencies)

    def apply_event_modifier(
        self, server_name: str, factor: float, duration_seconds: int
    ):
        with self.lock:
            if server_name in self.server_latencies:
                expiry = time.time() + duration_seconds if duration_seconds > 0 else 0
                self.server_event_modifiers[server_name] = (factor, expiry)
                logger.info(
                    f"Oracle: Event → {server_name}  factor={factor:.2f}  dur={duration_seconds}s"
                )
            else:
                logger.warning(f"Oracle: Unknown server '{server_name}' for event.")

    def is_any_event_active(self) -> bool:
        with self.lock:
            now = time.time()
            for _, (factor, expiry) in self.server_event_modifiers.items():
                if factor != 1.0 and (expiry == 0 or now < expiry):
                    return True
        return False

    def reset_events(self):
        with self.lock:
            self.server_event_modifiers = {k: (1.0, 0) for k in self.server_latencies}
            self.previous_latencies.clear()
            self.latency_history.clear()
            self.server_selection_counts.clear()
            self._tcp_sim.reset()
            for p in self._jitter_processes.values():
                p.reset()
            for g in self._micro_burst_gens.values():
                g.reset()
            for s in self._route_flap_sims.values():
                s.reset()
            self._backbone_congestion.clear()
            self.burst_active = False
            self.burst_end_time = 0.0
            self._burst_start_time = 0.0
            self.burst_affected_servers = []
            logger.info("Oracle: Full state reset.")

    def track_server_selection(self, server_name: str):
        with self.lock:
            self.server_selection_counts[server_name] = (
                self.server_selection_counts.get(server_name, 0) + 1
            )

    def trigger_traffic_burst(self, duration_seconds=30, intensity=2.5):
        with self.lock:
            self.burst_active = True
            self._burst_start_time = time.time()
            self.burst_end_time = self._burst_start_time + duration_seconds
            self._burst_intensity = intensity
            servers = list(self.server_latencies.keys())
            if servers:
                self.burst_affected_servers = random.sample(
                    servers, k=min(2, len(servers))
                )
                logger.info(
                    f"Traffic burst: {duration_seconds}s on {self.burst_affected_servers}"
                )

    def apply_correlated_failure(
        self, primary_server: str, cascade_factor: float = 1.8, duration: int = 20
    ):
        with self.lock:
            if primary_server not in self.server_latencies:
                return
            now = time.time()
            self.server_event_modifiers[primary_server] = (10.0, now + duration)
            for s in self.server_latencies:
                if s != primary_server:
                    self.server_event_modifiers[s] = (cascade_factor, now + duration)
            logger.warning(
                f"Correlated failure: {primary_server} → cascade={cascade_factor}"
            )

    def run_update_loop(self):
        logger.info("Oracle: Update loop started.")
        try:
            while self.running:
                self._update_latencies()
                steps = int(self.update_interval_seconds * 10)
                for _ in range(steps):
                    if not self.running:
                        break
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Oracle: Critical error in loop: {e}", exc_info=True)
        finally:
            logger.info("Oracle: Update loop ended.")

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self._update_server_geo_coordinates()
            self.thread = threading.Thread(target=self.run_update_loop, daemon=True)
            self.thread.start()
            logger.info("Oracle: Thread started.")

    def stop(self):
        logger.info("Oracle: Requesting stop.")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=self.update_interval_seconds + 1)
        if self.thread and self.thread.is_alive():
            logger.warning("Oracle: Thread did not terminate in time.")
        self.thread = None


if __name__ == "__main__":
    import json

    _fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _hdl = logging.StreamHandler()
    _hdl.setFormatter(_fmt)
    logger.addHandler(_hdl)
    logger.setLevel(logging.DEBUG)

    class MockMonitor:
        def getNodes(self):
            return [
                ("video-streaming-cache-1", "ip1"),
                ("video-streaming-cache-2", "ip2"),
                ("video-streaming-cache-3", "ip3"),
            ]

        def get_node_coordinates(self):
            return {
                "video-streaming-cache-1": {"lat": -23.0, "lon": -47.0},
                "video-streaming-cache-2": {"lat": -33.0, "lon": -71.0},
                "video-streaming-cache-3": {"lat": 5.0, "lon": -74.0},
            }

        def start_collecting(self):
            pass

        def stop_collecting(self):
            pass

        @property
        def interval(self):
            return 2

    logger.info("Starting ultra-realistic DynamicLatencyOracle test...")
    mock = MockMonitor()
    oracle = DynamicLatencyOracle(monitor=mock, update_interval_seconds=1)
    oracle.start()
    try:
        for i in range(15):
            time.sleep(1)
            lats = oracle.get_all_current_latencies()
            logger.info(
                f"Tick {i + 1:>2} | {json.dumps({k: round(v, 1) for k, v in lats.items()})}"
            )
            if i == 3:
                oracle.apply_event_modifier("video-streaming-cache-1", 5.0, 5)
                logger.info(">>> Event modifier on cache-1 (5×, 5s)")
            if i == 7:
                oracle.update_client_location(-30.0, -60.0)
                logger.info(">>> Client moved to (-30, -60)")
            if i == 10:
                oracle.trigger_traffic_burst(3, 3.0)
                logger.info(">>> Traffic burst triggered")
            logger.info(f"     Event active? {oracle.is_any_event_active()}")
    except KeyboardInterrupt:
        logger.info("Test interrupted.")
    finally:
        oracle.stop()
        logger.info("Test finished.")
