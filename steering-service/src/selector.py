import random
import math
import logging
import time
import numpy as np

selector_logger = logging.getLogger("SelectorStrategies")

class Selector:
    def __init__(self, monitor=None, latency_oracle=None):
        self.monitor = monitor
        self.latency_oracle = latency_oracle
        self.nodes = []

    def initialize(self, arms_names: list):
        self.nodes = [str(arm) for arm in arms_names if arm is not None] if arms_names else []
        selector_logger.debug(f"Selector {self.__class__.__name__} initialized with nodes: {self.nodes}")

    def select_arm(self, **kwargs) -> list:
        raise NotImplementedError

    def update(self, chosen_arm_name: str, feedback_value: float, **kwargs):
        pass

class EpsilonGreedy(Selector):
    def __init__(self, epsilon: float, counts: dict, values: dict, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.epsilon = epsilon
        self.counts = counts if isinstance(counts, dict) else {}
        self.values = values if isinstance(values, dict) else {}

    def initialize(self, arms_names: list):
        super().initialize(arms_names)
        new_counts = {arm: self.counts.get(arm, 0) for arm in self.nodes}
        new_values = {arm: self.values.get(arm, float('inf')) for arm in self.nodes}
        self.counts = new_counts
        self.values = new_values

    def select_arm(self, **kwargs) -> list:
        if self.monitor:
            current_monitor_node_names = [name for name, _ in self.monitor.getNodes() if name]
            if not current_monitor_node_names and not self.nodes: return []
            if set(current_monitor_node_names) != set(self.nodes):
                self.initialize(current_monitor_node_names)
        if not self.nodes: return []
        unvisited_arms = [arm for arm in self.nodes if self.counts.get(arm, 0) == 0]
        if unvisited_arms:
            random.shuffle(unvisited_arms)
            chosen_unvisited = unvisited_arms[0]
            other_nodes = [n for n in self.nodes if n != chosen_unvisited]
            if not other_nodes: return [chosen_unvisited]
            if random.random() > self.epsilon:
                sorted_remaining = sorted(other_nodes, key=lambda node: self.values.get(node, float('inf')))
            else:
                sorted_remaining = random.sample(other_nodes, len(other_nodes))
            return [chosen_unvisited] + sorted_remaining
        if random.random() > self.epsilon:
            return sorted(list(self.nodes), key=lambda node: self.values.get(node, float('inf')))
        else:
            return random.sample(self.nodes, len(self.nodes))

    def update(self, chosen_arm_name: str, punishment: float, **kwargs):
        str_arm = str(chosen_arm_name)
        if str_arm not in self.nodes:
            if self.monitor:
                nodes = [name for name, _ in self.monitor.getNodes() if name]
                if str_arm in nodes: self.initialize(nodes)
                if str_arm not in self.nodes:
                    selector_logger.warning(f"[EpsilonGreedy] Update: Arm {str_arm} not in self.nodes. Ignoring.")
                    return
            else:
                selector_logger.warning(f"[EpsilonGreedy] Update: Arm {str_arm} not in self.nodes (no monitor). Ignoring.")
                return

        if str_arm not in self.counts: self.counts[str_arm] = 0
        if str_arm not in self.values: self.values[str_arm] = float('inf')

        self.counts[str_arm] += 1
        n = self.counts[str_arm]
        current_avg_latency = self.values[str_arm]

        if current_avg_latency == float('inf'):
            self.values[str_arm] = float(punishment)
        else:
            self.values[str_arm] = ((n - 1) * current_avg_latency + float(punishment)) / n

class NoSteeringSelector(Selector):
    def __init__(self, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
    def select_arm(self, **kwargs) -> list:
        if self.monitor:
            nodes = [name for name, _ in self.monitor.getNodes() if name]
            if set(nodes) != set(self.nodes): self.initialize(nodes)
        return sorted(list(self.nodes)) if self.nodes else []

class RandomSelector(Selector):
    def __init__(self, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
    def select_arm(self, **kwargs) -> list:
        if self.monitor:
            nodes = [name for name, _ in self.monitor.getNodes() if name]
            if set(nodes) != set(self.nodes): self.initialize(nodes)
        if not self.nodes: return []
        return random.sample(self.nodes, len(self.nodes))

class UCB1Selector(Selector):
    def __init__(self, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.counts = {}
        self.values = {}
        self.total_pulls = 0
    def initialize(self, arms_names: list):
        super().initialize(arms_names)
        new_counts = {arm: self.counts.get(arm, 0) for arm in self.nodes}
        new_values = {arm: self.values.get(arm, 0.0) for arm in self.nodes}
        self.counts = new_counts
        self.values = new_values
    def select_arm(self, **kwargs) -> list:
        if self.monitor:
            nodes = [name for name, _ in self.monitor.getNodes() if name]
            if not nodes and not self.nodes: return []
            if set(nodes) != set(self.nodes): self.initialize(nodes)
        if not self.nodes: return []
        for arm_name in self.nodes:
            if self.counts.get(arm_name, 0) == 0:
                other_nodes = [n for n in self.nodes if n != arm_name]
                random.shuffle(other_nodes)
                return [arm_name] + other_nodes
        ucb_scores = {}
        current_total_pulls_for_log = self.total_pulls if self.total_pulls > 0 else sum(self.counts.values())
        log_total_pulls = math.log(max(1, current_total_pulls_for_log) + 1e-5)
        for arm in self.nodes:
            count = max(1e-5, self.counts.get(arm, 1e-5))
            sum_reward = self.values.get(arm, 0.0)
            avg_reward = sum_reward / count
            exploration_bonus = math.sqrt((2 * log_total_pulls) / count)
            ucb_scores[arm] = avg_reward + exploration_bonus
        return sorted(ucb_scores, key=ucb_scores.get, reverse=True)
    def update(self, chosen_arm_name: str, reward: float, **kwargs):
        str_arm = str(chosen_arm_name)
        if str_arm not in self.nodes:
            if self.monitor:
                nodes = [name for name, _ in self.monitor.getNodes() if name]
                if str_arm in nodes: self.initialize(nodes)
                if str_arm not in self.nodes:
                    selector_logger.warning(f"[UCB1] Update: Arm {str_arm} not in self.nodes. Ignoring.")
                    return
            else:
                selector_logger.warning(f"[UCB1] Update: Arm {str_arm} not in self.nodes (no monitor). Ignoring.")
                return

        if str_arm not in self.counts: self.counts[str_arm] = 0
        if str_arm not in self.values: self.values[str_arm] = 0.0

        self.counts[str_arm] += 1
        self.values[str_arm] += reward
        self.total_pulls += 1

class D_UCB(Selector):
    GAMMA_STILL = 0.995
    GAMMA_MOVEMENT_PERSISTENT = 0.75
    GAMMA_LATENCY_SHOCK = 0.60
    LATENCY_SHOCK_RECOVERY_DURATION_SECONDS = 7
    LATENCY_SHOCK_THRESHOLD_FACTOR = 2.5
    MIN_SAMPLES_FOR_SHOCK_DETECTION = 5

    def __init__(self, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.current_gamma = self.GAMMA_STILL
        self.discounted_counts = {}
        self.discounted_values = {}
        self.time_step = 0
        self.last_gamma_update_log_time = 0
        self.has_moved_ever = False
        self.latency_shock_recovery_active_until_time = 0
        self.raw_latency_sums = {}
        self.raw_pull_counts = {}
        self.actual_pull_counts = {}

    def initialize(self, arms_names: list):
        super().initialize(arms_names)
        new_discounted_counts = {}
        new_discounted_values = {}
        new_raw_latency_sums = {}
        new_raw_pull_counts = {}
        new_actual_pull_counts = {}

        for arm in self.nodes:
            new_discounted_counts[arm] = self.discounted_counts.get(arm, 0.0)
            new_discounted_values[arm] = self.discounted_values.get(arm, 0.0)
            new_raw_latency_sums[arm] = self.raw_latency_sums.get(arm, 0.0)
            new_raw_pull_counts[arm] = self.raw_pull_counts.get(arm, 0)
            new_actual_pull_counts[arm] = self.actual_pull_counts.get(arm, 0)

        self.discounted_counts = new_discounted_counts
        self.discounted_values = new_discounted_values
        self.raw_latency_sums = new_raw_latency_sums
        self.raw_pull_counts = new_raw_pull_counts
        self.actual_pull_counts = new_actual_pull_counts
        selector_logger.debug(f"[D_UCB] Initialized/Re-initialized. Actual Pull Counts: {self.actual_pull_counts}")

    def _check_latency_shock(self, arm_name: str, current_latency_ms: float) -> bool:
        if arm_name not in self.raw_pull_counts: self.raw_pull_counts[arm_name] = 0
        if arm_name not in self.raw_latency_sums: self.raw_latency_sums[arm_name] = 0.0

        if self.raw_pull_counts[arm_name] < self.MIN_SAMPLES_FOR_SHOCK_DETECTION:
            selector_logger.debug(f"[D_UCB] Shock not checked for {arm_name}: insufficient samples ({self.raw_pull_counts.get(arm_name, 0)}/{self.MIN_SAMPLES_FOR_SHOCK_DETECTION})")
            return False
        avg_raw_latency = self.raw_latency_sums[arm_name] / self.raw_pull_counts[arm_name]
        threshold_latency = avg_raw_latency * self.LATENCY_SHOCK_THRESHOLD_FACTOR
        if avg_raw_latency < 10:
            threshold_latency = max(threshold_latency, avg_raw_latency + 15)
        if current_latency_ms > threshold_latency:
            selector_logger.info(f"[D_UCB] Latency shock detected for {arm_name}! "
                                 f"Current: {current_latency_ms:.2f}ms vs Raw Avg: {avg_raw_latency:.2f}ms (Threshold: {threshold_latency:.2f}ms)")
            return True
        selector_logger.debug(f"[D_UCB] No shock for {arm_name}. Current: {current_latency_ms:.2f}ms, Raw Avg: {avg_raw_latency:.2f}ms, Threshold: {threshold_latency:.2f}ms")
        return False

    def update_environmental_state(self, client_is_moving_now: bool, latency_shock_detected: bool):
        old_gamma = self.current_gamma
        now = time.time()
        if client_is_moving_now:
            self.has_moved_ever = True
        if latency_shock_detected:
            self.current_gamma = self.GAMMA_LATENCY_SHOCK
            self.latency_shock_recovery_active_until_time = now + self.LATENCY_SHOCK_RECOVERY_DURATION_SECONDS
            log_reason = f"Latency Shock Detected"
        elif now < self.latency_shock_recovery_active_until_time:
            self.current_gamma = self.GAMMA_LATENCY_SHOCK
            log_reason = f"Post-Shock Recovery (remaining {self.latency_shock_recovery_active_until_time - now:.1f}s)"
        elif self.has_moved_ever:
            self.current_gamma = self.GAMMA_MOVEMENT_PERSISTENT
            log_reason = f"Persistent Movement"
        else:
            self.current_gamma = self.GAMMA_STILL
            log_reason = f"Normal State"
        if old_gamma != self.current_gamma or \
           ( (self.current_gamma != self.GAMMA_STILL) and \
             (now - self.last_gamma_update_log_time > 5) ):
            selector_logger.info(f"[D_UCB] Gamma: {self.current_gamma:.2f}. Reason: {log_reason}")
            self.last_gamma_update_log_time = now

    def select_arm(self, **kwargs) -> list:
        if self.monitor:
            nodes = [name for name, _ in self.monitor.getNodes() if name]
            if not nodes and not self.nodes: return []
            if set(nodes) != set(self.nodes):
                self.initialize(nodes)
        if not self.nodes: return []
        for arm_name in self.nodes:
            if arm_name not in self.discounted_counts: self.discounted_counts[arm_name] = 0.0
            if self.discounted_counts[arm_name] < 1e-5 :
                other_nodes = [n for n in self.nodes if n != arm_name]
                random.shuffle(other_nodes)
                selector_logger.debug(f"[D_UCB] Selecting unpulled arm: {arm_name}")
                return [arm_name] + other_nodes
        ucb_scores = {}
        log_t = math.log(self.time_step + 1e-5)
        exploration_coefficient = 2.0
        if self.current_gamma == self.GAMMA_LATENCY_SHOCK:
            exploration_coefficient = 1.5
        for arm in self.nodes:
            discounted_n_i = max(1e-5, self.discounted_counts.get(arm, 1e-5))
            current_sum_rewards = self.discounted_values.get(arm, 0.0)
            avg_reward = current_sum_rewards / discounted_n_i
            exploration_bonus = math.sqrt((exploration_coefficient * log_t) / discounted_n_i)
            ucb_scores[arm] = avg_reward + exploration_bonus
            selector_logger.debug(
                f"[D_UCB] Select Arm: {arm} | "
                f"AvgRew: {avg_reward:.3f} (SumRew: {current_sum_rewards:.3f} / DiscCnt: {discounted_n_i:.3f}) | "
                f"Bonus: {exploration_bonus:.3f} (Coeff: {exploration_coefficient:.1f}, log_t: {log_t:.3f}) | "
                f"UCB: {ucb_scores[arm]:.3f}"
            )
        sorted_arms = sorted(ucb_scores, key=ucb_scores.get, reverse=True)
        selector_logger.debug(f"[D_UCB] Sorted UCB: {[(arm, '{:.3f}'.format(ucb_scores[arm])) for arm in sorted_arms]}")
        return sorted_arms

    def update(self, chosen_arm_name: str, reward: float, **kwargs):
        str_arm = str(chosen_arm_name)
        if str_arm not in self.nodes:
            if self.monitor:
                nodes_now = [name for name, _ in self.monitor.getNodes() if name]
                if str_arm in nodes_now:
                    self.initialize(nodes_now)
                if str_arm not in self.nodes: return
            else: return

        if str_arm not in self.raw_pull_counts: self.raw_pull_counts[str_arm] = 0
        if str_arm not in self.raw_latency_sums: self.raw_latency_sums[str_arm] = 0.0
        if str_arm not in self.actual_pull_counts: self.actual_pull_counts[str_arm] = 0
        if str_arm not in self.discounted_counts: self.discounted_counts[str_arm] = 0.0
        if str_arm not in self.discounted_values: self.discounted_values[str_arm] = 0.0

        latency_ms = 1000.0 / reward if reward > 0 else float('inf')
        self.raw_pull_counts[str_arm] += 1
        self.raw_latency_sums[str_arm] += latency_ms
        self.actual_pull_counts[str_arm] += 1
        
        self.time_step += 1
        for arm_node in self.nodes:
            self.discounted_counts[arm_node] = self.discounted_counts.get(arm_node, 0.0) * self.current_gamma
            self.discounted_values[arm_node] = self.discounted_values.get(arm_node, 0.0) * self.current_gamma
        self.discounted_counts[str_arm] += 1.0
        self.discounted_values[str_arm] += reward
        selector_logger.debug(
            f"[D_UCB] Update: Arm={str_arm}, Latency={latency_ms:.2f}, Reward={reward:.2f}, Gamma={self.current_gamma:.2f} | "
            f"NewDiscCnt: {self.discounted_counts[str_arm]:.2f}, NewActualCnt: {self.actual_pull_counts.get(str_arm,0)} | "
            f"TimeStep: {self.time_step}"
        )

    @property
    def counts(self):
        return self.discounted_counts

    @property
    def real_counts(self):
        return self.actual_pull_counts

    @property
    def values(self):
        avg_rewards = {}
        for arm in self.nodes:
            count = self.discounted_counts.get(arm, 0.0)
            value_sum = self.discounted_values.get(arm, 0.0)
            avg_rewards[arm] = value_sum / count if count > 1e-6 else 0.0
        return avg_rewards

class LinUCBSelector(Selector):
    def __init__(self, d: int, alpha: float, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.d = d
        self.alpha = alpha
        
        self.A = {} 
        self.b = {} 
        
        selector_logger.info(f"LinUCBSelector inicializado com d={d} e alpha={alpha}")

    def initialize(self, arms_names: list):
        super().initialize(arms_names)
        for arm in self.nodes:
            if arm not in self.A:
                self.A[arm] = np.identity(self.d)
                self.b[arm] = np.zeros((self.d, 1))
                selector_logger.debug(f"[LinUCB] Braço '{arm}' inicializado.")

    def select_arm(self, **kwargs) -> list:
        contexts = kwargs.get("contexts")
        if not contexts:
            selector_logger.error("[LinUCB] 'contexts' não fornecidos para select_arm. Retornando lista vazia.")
            return []

        if set(contexts.keys()) != set(self.nodes):
            self.initialize(list(contexts.keys()))
        
        ucb_scores = {}
        for arm in self.nodes:
            if arm not in contexts:
                continue

            x_a = contexts[arm].reshape(-1, 1) 

            try:
                A_inv = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                selector_logger.warning(f"Matriz A para o braço {arm} é singular. Usando identidade.")
                A_inv = np.identity(self.d)

            theta_hat = A_inv.dot(self.b[arm])
            
            predicted_reward = theta_hat.T.dot(x_a)
            confidence_bonus = self.alpha * np.sqrt(x_a.T.dot(A_inv).dot(x_a))
            
            ucb_scores[arm] = predicted_reward + confidence_bonus

        if not ucb_scores:
            return []
        return sorted(ucb_scores, key=ucb_scores.get, reverse=True)

    def update(self, chosen_arm_name: str, reward: float, **kwargs):
        context = kwargs.get("context")
        if context is None:
            selector_logger.error(f"[LinUCB] 'context' não fornecido para update do braço {chosen_arm_name}.")
            return

        if chosen_arm_name not in self.nodes:
            selector_logger.warning(f"[LinUCB] Update: Braço {chosen_arm_name} não está nos nós conhecidos. Ignorando.")
            return
            
        x_chosen = context.reshape(-1, 1) 
        
        self.A[chosen_arm_name] += x_chosen.dot(x_chosen.T)
        self.b[chosen_arm_name] += reward * x_chosen
        
        selector_logger.debug(f"[LinUCB] Modelo para o braço '{chosen_arm_name}' atualizado com recompensa {reward:.2f}.")

class OracleBestChoiceSelector(Selector):
    def __init__(self, monitor=None, latency_oracle=None):
        if latency_oracle is None: raise ValueError("OracleBestChoiceSelector requires DynamicLatencyOracle.")
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)

    def select_arm(self, **kwargs) -> list:
        if not self.latency_oracle:
            return sorted(list(self.nodes)) if self.nodes else []
        if self.monitor:
            nodes = [name for name, _ in self.monitor.getNodes() if name]
            if not nodes and not self.nodes: return []
            if set(nodes) != set(self.nodes): self.initialize(nodes)
        if not self.nodes: return []
        latencies = self.latency_oracle.get_all_current_latencies()
        node_lats = {}
        for node_name in self.nodes:
            if node_name in latencies:
                node_lats[node_name] = latencies[node_name]
            else:
                node_lats[node_name] = float('inf')
        if not node_lats:
             return sorted(list(self.nodes)) if self.nodes else []
        return sorted(node_lats, key=node_lats.get)