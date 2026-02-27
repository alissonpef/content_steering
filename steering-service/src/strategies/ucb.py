import math
import random
from .base import Selector, selector_logger


class UCB1Selector(Selector):
    def __init__(self, c=2.0, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.c = c
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
            if not nodes and not self.nodes:
                return []
            if set(nodes) != set(self.nodes):
                self.initialize(nodes)
        if not self.nodes:
            return []
        for arm_name in self.nodes:
            if self.counts.get(arm_name, 0) == 0:
                other_nodes = [n for n in self.nodes if n != arm_name]
                random.shuffle(other_nodes)
                return [arm_name] + other_nodes
        ucb_scores = {}
        current_total_pulls_for_log = (
            self.total_pulls if self.total_pulls > 0 else sum(self.counts.values())
        )
        log_total_pulls = math.log(max(1, current_total_pulls_for_log))
        for arm in self.nodes:
            count = max(1e-5, self.counts.get(arm, 1e-5))
            sum_reward = self.values.get(arm, 0.0)
            avg_reward = sum_reward / count
            exploration_bonus = math.sqrt((self.c * log_total_pulls) / count)
            ucb_scores[arm] = avg_reward + exploration_bonus
        return sorted(ucb_scores, key=ucb_scores.get, reverse=True)

    def update(self, chosen_arm_name: str, reward: float, **kwargs):
        str_arm = str(chosen_arm_name)
        if str_arm not in self.nodes:
            if self.monitor:
                nodes = [name for name, _ in self.monitor.getNodes() if name]
                if str_arm in nodes:
                    self.initialize(nodes)
                if str_arm not in self.nodes:
                    selector_logger.warning(
                        f"[UCB1] Update: Arm {str_arm} not in self.nodes. Ignoring."
                    )
                    return
            else:
                selector_logger.warning(
                    f"[UCB1] Update: Arm {str_arm} not in self.nodes (no monitor). Ignoring."
                )
                return
        if str_arm not in self.counts:
            self.counts[str_arm] = 0
        if str_arm not in self.values:
            self.values[str_arm] = 0.0
        self.counts[str_arm] += 1
        self.values[str_arm] += reward
        self.total_pulls += 1
