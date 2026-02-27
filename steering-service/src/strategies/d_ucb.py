import math
import random
from .base import Selector, selector_logger


class D_UCB(Selector):
    def __init__(self, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.GAMMA_BASE = 0.97
        self.GAMMA_ADAPT = 0.86
        self.GAMMA_SHOCK = 0.72
        self.EXPLORATION_COEFF = 2.0
        self.EMA_BETA = 0.15
        self.SHOCK_Z_THRESHOLD = 2.5
        self.ADAPT_Z_THRESHOLD = 1.4
        self.SHOCK_COOLDOWN_STEPS = 15
        self.ADAPT_COOLDOWN_STEPS = 8
        self.current_gamma = self.GAMMA_BASE
        self.discounted_counts = {}
        self.discounted_values = {}
        self.time_step = 0
        self.raw_latency_sums = {}
        self.raw_pull_counts = {}
        self.actual_pull_counts = {}
        self._latency_ema = {}
        self._latency_var_ema = {}
        self._adaptive_cooldown_steps = 0

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
        for arm in self.nodes:
            if arm not in self._latency_ema:
                self._latency_ema[arm] = None
            if arm not in self._latency_var_ema:
                self._latency_var_ema[arm] = 1.0
        selector_logger.debug(
            f"[D_UCB] Initialized/Re-initialized. Actual Pull Counts: {self.actual_pull_counts}"
        )

    def _detect_non_stationarity(self, arm_name: str, latency_ms: float):
        ema = self._latency_ema.get(arm_name)
        var = self._latency_var_ema.get(arm_name, 1.0)
        if ema is None:
            self._latency_ema[arm_name] = latency_ms
            self._latency_var_ema[arm_name] = 1.0
            return 0.0
        residual = latency_ms - ema
        beta = self.EMA_BETA
        ema_new = (1.0 - beta) * ema + beta * latency_ms
        var_new = (1.0 - beta) * var + beta * (residual ** 2)
        self._latency_ema[arm_name] = ema_new
        self._latency_var_ema[arm_name] = max(1e-6, var_new)
        z_score = abs(residual) / math.sqrt(self._latency_var_ema[arm_name])
        return z_score

    def _update_gamma_from_latency(self, z_score: float):
        if z_score >= self.SHOCK_Z_THRESHOLD:
            self._adaptive_cooldown_steps = max(
                self._adaptive_cooldown_steps, self.SHOCK_COOLDOWN_STEPS
            )
        elif z_score >= self.ADAPT_Z_THRESHOLD:
            self._adaptive_cooldown_steps = max(
                self._adaptive_cooldown_steps, self.ADAPT_COOLDOWN_STEPS
            )
        if self._adaptive_cooldown_steps > self.ADAPT_COOLDOWN_STEPS:
            self.current_gamma = self.GAMMA_SHOCK
        elif self._adaptive_cooldown_steps > 0:
            self.current_gamma = self.GAMMA_ADAPT
        else:
            self.current_gamma = self.GAMMA_BASE

    def update_environmental_state(
        self, client_is_moving_now: bool, latency_shock_detected: bool
    ):
        return

    def select_arm(self, **kwargs) -> list:
        if self.monitor:
            nodes = [name for name, _ in self.monitor.getNodes() if name]
            if not nodes and not self.nodes:
                return []
            if set(nodes) != set(self.nodes):
                self.initialize(nodes)
        if not self.nodes:
            return []
        for arm_name in self.nodes:
            if arm_name not in self.discounted_counts:
                self.discounted_counts[arm_name] = 0.0
            if self.discounted_counts[arm_name] < 1e-5:
                other_nodes = [n for n in self.nodes if n != arm_name]
                random.shuffle(other_nodes)
                selector_logger.debug(f"[D_UCB] Selecting unpulled arm: {arm_name}")
                return [arm_name] + other_nodes
        ucb_scores = {}
        log_t = math.log(max(1, self.time_step))
        exploration_coefficient = self.EXPLORATION_COEFF
        for arm in self.nodes:
            discounted_n_i = max(1e-5, self.discounted_counts.get(arm, 1e-5))
            current_sum_rewards = self.discounted_values.get(arm, 0.0)
            avg_reward = current_sum_rewards / discounted_n_i
            exploration_bonus = math.sqrt(
                (exploration_coefficient * log_t) / discounted_n_i
            )
            ucb_scores[arm] = avg_reward + exploration_bonus
            selector_logger.debug(
                f"[D_UCB] Select Arm: {arm} | "
                f"AvgRew: {avg_reward:.3f} (SumRew: {current_sum_rewards:.3f} / DiscCnt: {discounted_n_i:.3f}) | "
                f"Bonus: {exploration_bonus:.3f} (Coeff: {exploration_coefficient:.1f}, log_t: {log_t:.3f}) | "
                f"UCB: {ucb_scores[arm]:.3f}"
            )
        sorted_arms = sorted(ucb_scores, key=ucb_scores.get, reverse=True)
        selector_logger.debug(
            f"[D_UCB] Sorted UCB: {[(arm, '{:.3f}'.format(ucb_scores[arm])) for arm in sorted_arms]}"
        )
        return sorted_arms

    def update(self, chosen_arm_name: str, reward: float, **kwargs):
        str_arm = str(chosen_arm_name)
        if str_arm not in self.nodes:
            if self.monitor:
                nodes_now = [name for name, _ in self.monitor.getNodes() if name]
                if str_arm in nodes_now:
                    self.initialize(nodes_now)
                if str_arm not in self.nodes:
                    return
            else:
                return
        if str_arm not in self.raw_pull_counts:
            self.raw_pull_counts[str_arm] = 0
        if str_arm not in self.raw_latency_sums:
            self.raw_latency_sums[str_arm] = 0.0
        if str_arm not in self.actual_pull_counts:
            self.actual_pull_counts[str_arm] = 0
        if str_arm not in self.discounted_counts:
            self.discounted_counts[str_arm] = 0.0
        if str_arm not in self.discounted_values:
            self.discounted_values[str_arm] = 0.0
        latency_ms = 1000.0 / reward if reward > 0 else float("inf")
        self.raw_pull_counts[str_arm] += 1
        self.raw_latency_sums[str_arm] += latency_ms
        self.actual_pull_counts[str_arm] += 1
        self.time_step += 1
        z_score = self._detect_non_stationarity(str_arm, latency_ms)
        self._update_gamma_from_latency(z_score)
        for arm_node in self.nodes:
            self.discounted_counts[arm_node] = (
                self.discounted_counts.get(arm_node, 0.0) * self.current_gamma
            )
            self.discounted_values[arm_node] = (
                self.discounted_values.get(arm_node, 0.0) * self.current_gamma
            )
        self.discounted_counts[str_arm] += 1.0
        self.discounted_values[str_arm] += reward
        if self._adaptive_cooldown_steps > 0:
            self._adaptive_cooldown_steps -= 1
        selector_logger.debug(
            f"[D_UCB] Update: Arm={str_arm}, Latency={latency_ms:.2f}, Reward={reward:.2f}, Gamma={self.current_gamma:.2f}, z={z_score:.2f} | "
            f"NewDiscCnt: {self.discounted_counts[str_arm]:.2f}, NewActualCnt: {self.actual_pull_counts.get(str_arm, 0)} | "
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
