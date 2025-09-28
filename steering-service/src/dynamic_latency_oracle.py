import time
import random
import threading
import numpy as np
import logging
import math
import json 

logger = logging.getLogger("LatencyOracle")

def calculate_haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    if None in [lat1, lon1, lat2, lon2]:
        return 0.0
    try:
        lat1_f, lon1_f, lat2_f, lon2_f = float(lat1), float(lon1), float(lat2), float(lon2)
        dLat, dLon, lat1_rad, lat2_rad = map(math.radians, [lat2_f - lat1_f, lon2_f - lon1_f, lat1_f, lat2_f])
    except (ValueError, TypeError):
        return 0.0
    a = math.sin(dLat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class DynamicLatencyOracle:
    DEFAULT_USE_DISTANCE_PENALTY: bool = True
    DEFAULT_MS_PER_KM_FACTOR: float = 0.0250
    DEFAULT_INITIAL_CLIENT_LAT: float = -23.0
    DEFAULT_INITIAL_CLIENT_LON: float = -47.0

    def __init__(self, monitor, update_interval_seconds: int = 2):
        self.monitor = monitor
        self.server_latencies = {}
        # Configurações base para latência
        self.server_base_latencies_config = {
            "video-streaming-cache-1": 30,
            "video-streaming-cache-2": 25,
            "video-streaming-cache-3": 125
        }
        # Novas configurações base para jitter e perda de pacotes
        self.server_base_jitter_config = {
            "video-streaming-cache-1": 5,   # ms
            "video-streaming-cache-2": 10,  # ms
            "video-streaming-cache-3": 20   # ms
        }
        self.server_base_packet_loss_config = {
            "video-streaming-cache-1": 0.001, # 0.1%
            "video-streaming-cache-2": 0.005, # 0.5%
            "video-streaming-cache-3": 0.01   # 1.0%
        }
        self.server_geo_coords = {}
        self.client_latitude = DynamicLatencyOracle.DEFAULT_INITIAL_CLIENT_LAT
        self.client_longitude = DynamicLatencyOracle.DEFAULT_INITIAL_CLIENT_LON
        self.server_event_modifiers = {}
        self.update_interval_seconds = max(0.5, update_interval_seconds)
        self.ms_per_km_factor = DynamicLatencyOracle.DEFAULT_MS_PER_KM_FACTOR
        self.use_distance_penalty = DynamicLatencyOracle.DEFAULT_USE_DISTANCE_PENALTY
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.noise_std_dev_factor = 0.15
        self.min_simulated_latency = 5
        self._update_server_geo_coordinates()

    def _update_server_geo_coordinates(self):
        if self.monitor:
            coords = self.monitor.get_node_coordinates()
            with self.lock:
                self.server_geo_coords = coords if isinstance(coords, dict) else {}

    def update_client_location(self, lat: float, lon: float):
        if lat is None or lon is None:
            return
        with self.lock:
            try:
                new_lat, new_lon = float(lat), float(lon)
                if new_lat != self.client_latitude or new_lon != self.client_longitude:
                    self.client_latitude = new_lat
                    self.client_longitude = new_lon
                    logger.debug(f"Oracle: Client location updated to Lat={new_lat}, Lon={new_lon}")
            except (ValueError, TypeError):
                logger.warning(f"Oracle: Invalid client coordinates received: lat={lat}, lon={lon}")

    def _initialize_server_states(self):
        current_nodes_info = self.monitor.getNodes()
        if not current_nodes_info:
            logger.debug("Oracle: No monitor nodes to initialize states.")
            return

        current_node_names = [info[0] for info in current_nodes_info if info and info[0]]
        if not current_node_names:
            logger.debug("Oracle: No valid node names from monitor.")
            return

        self._update_server_geo_coordinates()

        with self.lock:
            for name in current_node_names:
                if name not in self.server_latencies:
                    initial_lat = self.server_base_latencies_config.get(name, random.uniform(10, 30))
                    self.server_latencies[name] = initial_lat
                    self.server_event_modifiers[name] = (1.0, 0)
                    logger.info(f"Oracle: Server {name} added (initial lat: {initial_lat:.2f}ms).")

            servers_in_oracle = list(self.server_latencies.keys())
            removed_servers = [name for name in servers_in_oracle if name not in current_node_names]
            for name in removed_servers:
                del self.server_latencies[name]
                if name in self.server_event_modifiers:
                    del self.server_event_modifiers[name]
                logger.info(f"Oracle: Server {name} removed.")

    def _update_latencies(self):
        self._initialize_server_states()
        with self.lock:
            for server_name in list(self.server_latencies.keys()):
                _, final_latency = self.get_context_and_final_latency(server_name)
                self.server_latencies[server_name] = final_latency
                logger.debug(f"Oracle: Updated Latency {server_name}: {final_latency:.2f}ms")

    def get_context_and_final_latency(self, server_name: str) -> tuple[np.ndarray, float]:
        """
        Calcula e retorna o vetor de contexto para LinUCB e a latência final simulada.
        """
        # 1. Obter os componentes base da latência
        if server_name not in self.server_latencies:
            self._initialize_server_states()
        
        base_latency_config = self.server_base_latencies_config.get(server_name, 30)
        current_modifier_factor, current_expiry_time = self.server_event_modifiers.get(server_name, (1.0, 0))
        
        final_modifier_to_apply = current_modifier_factor
        if current_expiry_time != 0 and time.time() >= current_expiry_time:
            final_modifier_to_apply = 1.0
            if self.server_event_modifiers.get(server_name) != (1.0, 0):
                self.server_event_modifiers[server_name] = (1.0, 0)
        
        # 2. Construir as features para o vetor de contexto
        # Feature 1: Distância Geográfica
        feature_distance_km = 0.0
        if self.use_distance_penalty and self.client_latitude is not None and self.client_longitude is not None:
            server_coords = self.server_geo_coords.get(server_name)
            if server_coords and server_coords.get('lat') is not None and server_coords.get('lon') is not None:
                feature_distance_km = calculate_haversine_distance(
                    self.client_latitude, self.client_longitude,
                    server_coords['lat'], server_coords['lon']
                )

        # Feature 2: "Carga" do Servidor (latência base + ruído + spam)
        noise = np.random.normal(loc=0, scale=max(1, base_latency_config) * self.noise_std_dev_factor)
        feature_server_load = max(self.min_simulated_latency, (base_latency_config + noise)) * final_modifier_to_apply
        
        # Feature 3: Jitter Simulado
        base_jitter = self.server_base_jitter_config.get(server_name, 10)
        jitter_multiplier = 1.0 + (final_modifier_to_apply - 1.0) * 1.5 # Spam aumenta o jitter
        feature_jitter = np.random.uniform(base_jitter * 0.5, base_jitter * 1.5) * jitter_multiplier

        # Feature 4: Perda de Pacotes Simulada
        base_loss = self.server_base_packet_loss_config.get(server_name, 0.01)
        feature_packet_loss = min(1.0, base_loss * final_modifier_to_apply) # Spam aumenta a perda
        
        # Vetor de contexto ENRIQUECIDO (agora com 5 dimensões!)
        context_vector = np.array([
            1.0,
            feature_distance_km,
            feature_server_load,
            feature_jitter,
            feature_packet_loss 
        ])

        # 3. Calcular a Latência Final (ground truth)
        distance_penalty = feature_distance_km * self.ms_per_km_factor
        jitter_effect_on_latency = feature_jitter * np.random.choice([-1, 1]) # Variação aleatória
        packet_loss_penalty = 500 if np.random.rand() < feature_packet_loss else 0 # Simula retransmissão
        
        final_latency = feature_server_load + distance_penalty + jitter_effect_on_latency + packet_loss_penalty
        
        return context_vector, final_latency


    def get_current_latency(self, server_name: str) -> float:
        with self.lock:
            latency = self.server_latencies.get(server_name)
            if latency is None:
                logger.warning(f"Oracle: Latency not found for {server_name}. Returning random value.")
                return random.uniform(50, 150)
            return latency

    def get_all_current_latencies(self) -> dict:
        with self.lock:
            if not self.server_latencies and self.monitor and self.monitor.getNodes():
                 self._initialize_server_states()
            return dict(self.server_latencies)

    def apply_event_modifier(self, server_name: str, factor: float, duration_seconds: int):
        with self.lock:
            if server_name in self.server_latencies:
                expiry_timestamp = time.time() + duration_seconds if duration_seconds > 0 else 0
                self.server_event_modifiers[server_name] = (factor, expiry_timestamp)
                logger.info(f"Oracle: Latency event applied to {server_name}. Factor: {factor:.2f}, Duration: {duration_seconds}s.")
            else:
                logger.warning(f"Oracle: Attempt to apply event to unknown server '{server_name}'.")

    def is_any_event_active(self) -> bool:
        with self.lock:
            current_time = time.time()
            for server_name, (factor, expiry_time) in self.server_event_modifiers.items():
                if factor != 1.0 and (expiry_time == 0 or current_time < expiry_time) :
                    logger.debug(f"Oracle: Active event detected for {server_name} (factor: {factor}, expires at: {expiry_time})")
                    return True
        return False

    def run_update_loop(self):
        logger.info("Oracle: Starting latency update loop.")
        try:
            while self.running:
                self._update_latencies()
                for _ in range(int(self.update_interval_seconds * 10)):
                    if not self.running:
                        break
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Oracle: Critical error in update loop: {e}", exc_info=True)
        finally:
            logger.info("Oracle: Latency update loop ended.")

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self._update_server_geo_coordinates()
            self.thread = threading.Thread(target=self.run_update_loop, daemon=True)
            self.thread.start()
            logger.info("Oracle: Update thread started.")

    def stop(self):
        logger.info("Oracle: Requesting stop of update thread.")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=self.update_interval_seconds + 1)
        if self.thread and self.thread.is_alive():
            logger.warning("Oracle: Update thread did not terminate in the expected time.")
        self.thread = None
        
if __name__ == '__main__':
    _formatter_standalone = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler_standalone = logging.StreamHandler()
    _handler_standalone.setFormatter(_formatter_standalone)

    logger.addHandler(_handler_standalone)
    logger.setLevel(logging.DEBUG)

    class MockMonitor:
        def getNodes(self): return [("video-streaming-cache-1", "ip1"), ("video-streaming-cache-2", "ip2")]
        def get_node_coordinates(self): return {"video-streaming-cache-1": {"lat": -23.0, "lon": -47.0},"video-streaming-cache-2": {"lat": -33.0, "lon": -71.0}}
        def start_collecting(self): logger.info("MockMonitor: start_collecting()")
        def stop_collecting(self): logger.info("MockMonitor: stop_collecting()")
        @property
        def interval(self): return 2

    logger.info("Starting standalone test of DynamicLatencyOracle...")
    mock_monitor = MockMonitor()
    oracle = DynamicLatencyOracle(monitor=mock_monitor, update_interval_seconds=1)
    oracle.start()
    try:
        for i in range(10):
            time.sleep(1)
            all_lats = oracle.get_all_current_latencies()
            logger.info(f"Tick {i+1:>2} - Latencies: {json.dumps({k: round(v,1) for k,v in all_lats.items()})}")
            if i == 2:
                oracle.apply_event_modifier("video-streaming-cache-1", 5.0, 4)
                logger.info("Applied SPAM to video-streaming-cache-1 for 4s")
            is_spam = oracle.is_any_event_active()
            logger.info(f"Tick {i+1:>2} - Spam event active? {is_spam}")
            if i == 7: oracle.update_client_location(-30.0, -60.0); logger.info("Client moved.")
    except KeyboardInterrupt: logger.info("Oracle test interrupted.")
    finally: oracle.stop(); logger.info("Oracle test finished.")