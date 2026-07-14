import random

from .base import Selector, selector_logger


class EpsilonGreedy(Selector):
    def __init__(
        self,
        epsilon: float,
        counts: dict,
        values: dict,
        gamma: float = 0.9,
        reward_scale: float = 200.0,
        monitor=None,
    ):
        super().__init__(monitor=monitor)
        self.epsilon = epsilon
        self.gamma = gamma
        self.reward_scale = max(1e-06, reward_scale)
        self.counts = counts if isinstance(counts, dict) else {}
        self.values = values if isinstance(values, dict) else {}

    def initialize(self, arms_names: list[str]):
        super().initialize(arms_names)
        new_counts = {arm: self.counts.get(arm, 0) for arm in self.nodes}
        new_values = {arm: self.values.get(arm, 0.0) for arm in self.nodes}
        self.counts = new_counts
        self.values = new_values

    def select_arm(self, **kwargs) -> list[str]:
        if self.monitor:
            current_monitor_node_names = [name for name, _ in self.monitor.get_nodes() if name]
            if not current_monitor_node_names and (not self.nodes):
                return []
            if set(current_monitor_node_names) != set(self.nodes):
                self.initialize(current_monitor_node_names)
        if not self.nodes:
            return []
        unvisited_arms = [arm for arm in self.nodes if self.counts.get(arm, 0) == 0]
        if unvisited_arms:
            random.shuffle(unvisited_arms)
            chosen_unvisited = unvisited_arms[0]
            other_nodes = [n for n in self.nodes if n != chosen_unvisited]
            if not other_nodes:
                return [chosen_unvisited]
            if random.random() > self.epsilon:
                sorted_remaining = sorted(
                    other_nodes,
                    key=lambda node: self.values.get(node, 0.0),
                    reverse=True,
                )
            else:
                sorted_remaining = random.sample(other_nodes, len(other_nodes))
            return [chosen_unvisited] + sorted_remaining
        if random.random() > self.epsilon:
            return sorted(
                list(self.nodes),
                key=lambda node: self.values.get(node, 0.0),
                reverse=True,
            )
        else:
            return random.sample(self.nodes, len(self.nodes))

    def update(self, chosen_arm_name: str, feedback_value: float, **kwargs):
        str_arm = chosen_arm_name
        if str_arm not in self.nodes:
            if self.monitor:
                nodes = [name for name, _ in self.monitor.get_nodes() if name]
                if str_arm in nodes:
                    self.initialize(nodes)
                if str_arm not in self.nodes:
                    selector_logger.warning(
                        f"[EpsilonGreedy] Update: Arm {str_arm} not in self.nodes. Ignoring."
                    )
                    return
            else:
                selector_logger.warning(
                    f"[EpsilonGreedy] Update: Arm {str_arm} not in self.nodes (no monitor). Ignoring."
                )
                return
        if str_arm not in self.counts:
            self.counts[str_arm] = 0
        if str_arm not in self.values:
            self.values[str_arm] = 0.0
        if self.counts[str_arm] == 0:
            self.values[str_arm] = feedback_value / self.reward_scale
        else:
            alpha = 1.0 - self.gamma
            current_avg_reward = self.values[str_arm]
            normalized = feedback_value / self.reward_scale
            self.values[str_arm] = (1.0 - alpha) * current_avg_reward + alpha * normalized
        self.counts[str_arm] += 1
