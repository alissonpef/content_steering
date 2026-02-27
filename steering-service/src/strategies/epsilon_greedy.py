import random
from .base import Selector, selector_logger


class EpsilonGreedy(Selector):
    def __init__(
        self,
        epsilon: float,
        counts: dict,
        values: dict,
        monitor=None,
        latency_oracle=None,
    ):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.epsilon = epsilon
        self.counts = counts if isinstance(counts, dict) else {}
        self.values = values if isinstance(values, dict) else {}

    def initialize(self, arms_names: list):
        super().initialize(arms_names)
        new_counts = {arm: self.counts.get(arm, 0) for arm in self.nodes}
        new_values = {arm: self.values.get(arm, float("inf")) for arm in self.nodes}
        self.counts = new_counts
        self.values = new_values

    def select_arm(self, **kwargs) -> list:
        if self.monitor:
            current_monitor_node_names = [
                name for name, _ in self.monitor.getNodes() if name
            ]
            if not current_monitor_node_names and not self.nodes:
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
                    other_nodes, key=lambda node: self.values.get(node, float("inf"))
                )
            else:
                sorted_remaining = random.sample(other_nodes, len(other_nodes))
            return [chosen_unvisited] + sorted_remaining
        if random.random() > self.epsilon:
            return sorted(
                list(self.nodes), key=lambda node: self.values.get(node, float("inf"))
            )
        else:
            return random.sample(self.nodes, len(self.nodes))

    def update(self, chosen_arm_name: str, feedback_value: float, **kwargs):
        str_arm = str(chosen_arm_name)
        if str_arm not in self.nodes:
            if self.monitor:
                nodes = [name for name, _ in self.monitor.getNodes() if name]
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
            self.values[str_arm] = float("inf")
        self.counts[str_arm] += 1
        n = self.counts[str_arm]
        current_avg_latency = self.values[str_arm]
        if current_avg_latency == float("inf"):
            self.values[str_arm] = float(feedback_value)
        else:
            self.values[str_arm] = (
                (n - 1) * current_avg_latency + float(feedback_value)
            ) / n
