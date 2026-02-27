import math
import random
import time
from .base import Selector, selector_logger


class D_UCB(Selector):
    def __init__(self, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.GAMMA_STILL = 0.97
        self.GAMMA_MOVEMENT = 0.85
        self.GAMMA_LATENCY_SHOCK = 0.70
        self.LATENCY_SHOCK_RECOVERY_DURATION_SECONDS = 10
        self.LATENCY_SHOCK_THRESHOLD_FACTOR = 2.0
        self.MIN_SAMPLES_FOR_SHOCK_DETECTION = 5
        self.MOVEMENT_COOLDOWN_SECONDS = 10
        self.current_gamma = self.GAMMA_STILL
        self.discounted_counts = {}
        self.discounted_values = {}
        self.time_step = 0
        self.last_gamma_update_log_time = 0
        self._last_movement_time = 0.0
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
        selector_logger.debug(
            f"[D_UCB] Initialized/Re-initialized. Actual Pull Counts: {self.actual_pull_counts}"
        )

    def _check_latency_shock(self, arm_name: str, current_latency_ms: float) -> bool:
        if arm_name not in self.raw_pull_counts:
            self.raw_pull_counts[arm_name] = 0
        if arm_name not in self.raw_latency_sums:
            self.raw_latency_sums[arm_name] = 0.0
        if self.raw_pull_counts[arm_name] < self.MIN_SAMPLES_FOR_SHOCK_DETECTION:
            selector_logger.debug(
                f"[D_UCB] Shock not checked for {arm_name}: insufficient samples ({self.raw_pull_counts.get(arm_name, 0)}/{self.MIN_SAMPLES_FOR_SHOCK_DETECTION})"
            )
            return False
        avg_raw_latency = (
            self.raw_latency_sums[arm_name] / self.raw_pull_counts[arm_name]
        )
        threshold_latency = avg_raw_latency * self.LATENCY_SHOCK_THRESHOLD_FACTOR
        if avg_raw_latency < 10:
            threshold_latency = max(threshold_latency, avg_raw_latency + 15)
        if current_latency_ms > threshold_latency:
            selector_logger.info(
                f"[D_UCB] Latency shock detected for {arm_name}! "
                f"Current: {current_latency_ms:.2f}ms vs Raw Avg: {avg_raw_latency:.2f}ms (Threshold: {threshold_latency:.2f}ms)"
            )
            return True
        selector_logger.debug(
            f"[D_UCB] No shock for {arm_name}. Current: {current_latency_ms:.2f}ms, Raw Avg: {avg_raw_latency:.2f}ms, Threshold: {threshold_latency:.2f}ms"
        )
        return False

    def update_environmental_state(
        self, client_is_moving_now: bool, latency_shock_detected: bool
    ):
        old_gamma = self.current_gamma
        now = time.time()
        if client_is_moving_now:
            self._last_movement_time = now
        if latency_shock_detected:
            self.current_gamma = self.GAMMA_LATENCY_SHOCK
            self.latency_shock_recovery_active_until_time = (
                now + self.LATENCY_SHOCK_RECOVERY_DURATION_SECONDS
            )
            log_reason = "Latency Shock Detected"
        elif now < self.latency_shock_recovery_active_until_time:
            self.current_gamma = self.GAMMA_LATENCY_SHOCK
            log_reason = f"Post-Shock Recovery (remaining {self.latency_shock_recovery_active_until_time - now:.1f}s)"
        elif client_is_moving_now:
            self.current_gamma = self.GAMMA_MOVEMENT
            log_reason = "Active Movement"
        elif (now - self._last_movement_time) < self.MOVEMENT_COOLDOWN_SECONDS and self._last_movement_time > 0:
            self.current_gamma = self.GAMMA_MOVEMENT
            log_reason = f"Movement Cooldown (remaining {self.MOVEMENT_COOLDOWN_SECONDS - (now - self._last_movement_time):.1f}s)"
        else:
            self.current_gamma = self.GAMMA_STILL
            log_reason = "Normal State"
        if old_gamma != self.current_gamma or (
            (self.current_gamma != self.GAMMA_STILL)
            and (now - self.last_gamma_update_log_time > 5)
        ):
            selector_logger.info(
                f"[D_UCB] Gamma: {self.current_gamma:.2f}. Reason: {log_reason}"
            )
            self.last_gamma_update_log_time = now

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
        exploration_coefficient = 2.0
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
        for arm_node in self.nodes:
            self.discounted_counts[arm_node] = (
                self.discounted_counts.get(arm_node, 0.0) * self.current_gamma
            )
            self.discounted_values[arm_node] = (
                self.discounted_values.get(arm_node, 0.0) * self.current_gamma
            )
        self.discounted_counts[str_arm] += 1.0
        self.discounted_values[str_arm] += reward
        selector_logger.debug(
            f"[D_UCB] Update: Arm={str_arm}, Latency={latency_ms:.2f}, Reward={reward:.2f}, Gamma={self.current_gamma:.2f} | "
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
