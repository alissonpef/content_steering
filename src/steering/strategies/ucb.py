import math
import random
from .base import Selector, selector_logger


class UCB1Selector(Selector):
    def __init__(self, c=2.0, gamma=0.99, monitor=None):
        super().__init__(monitor=monitor)
        self.c = c
        self.gamma = gamma
        self.counts = {}
        self.values = {}
        self.total_pulls = 0.0

    def initialize(self, arms_names: list[str]):
        super().initialize(arms_names)
        new_counts = {arm: self.counts.get(arm, 0) for arm in self.nodes}
        new_values = {arm: self.values.get(arm, 0.0) for arm in self.nodes}
        self.counts = new_counts
        self.values = new_values

    def select_arm(self, **kwargs) -> list[str]:
        if self.monitor:
            nodes = [name for name, _ in self.monitor.get_nodes() if name]
            if not nodes and not self.nodes:
                return []
            if set(nodes) != set(self.nodes):
                self.initialize(nodes)
        if not self.nodes:
            return []
        unvisited = [arm for arm in self.nodes if self.counts.get(arm, 0) == 0]
        if unvisited:
            random.shuffle(unvisited)
            chosen = unvisited[0]
            others = [n for n in self.nodes if n != chosen]
            random.shuffle(others)
            return [chosen] + others
        ucb_scores = {}
        current_total_pulls_for_log = (
            self.total_pulls if self.total_pulls > 0 else sum(self.counts.values())
        )
        log_total_pulls = math.log(max(1, current_total_pulls_for_log))
        scale = max(1.0, max(self.values.values()) if self.values else 1.0)
        for arm in self.nodes:
            count = max(1e-5, self.counts.get(arm, 1e-5))
            avg_reward = self.values.get(arm, 0.0)
            exploration_bonus = scale * math.sqrt((self.c * log_total_pulls) / count)
            ucb_scores[arm] = avg_reward + exploration_bonus
        return sorted(ucb_scores, key=lambda k: float(ucb_scores[k]), reverse=True)

    def update(self, chosen_arm_name: str, feedback_value: float, **kwargs):
        str_arm = chosen_arm_name
        if str_arm not in self.nodes:
            if self.monitor:
                nodes = [name for name, _ in self.monitor.get_nodes() if name]
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
            self.counts[str_arm] = 0.0
        if str_arm not in self.values:
            self.values[str_arm] = 0.0
            
        for arm in self.nodes:
            if arm in self.counts:
                self.counts[arm] *= self.gamma
                
        old_count_decayed = self.counts[str_arm]
        self.counts[str_arm] += 1.0
        n = self.counts[str_arm]
        previous_value = self.values.get(str_arm, 0.0)
        self.values[str_arm] = (old_count_decayed * previous_value + feedback_value) / n
        self.total_pulls = self.total_pulls * self.gamma + 1.0
