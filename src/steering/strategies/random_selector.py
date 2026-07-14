import random

from .base import Selector


class RandomSelector(Selector):
    def select_arm(self, **kwargs) -> list[str]:
        if self.monitor:
            current_monitor_node_names = [name for name, _ in self.monitor.get_nodes() if name]
            if not current_monitor_node_names and (not self.nodes):
                return []
            if set(current_monitor_node_names) != set(self.nodes):
                self.initialize(current_monitor_node_names)
        if not self.nodes:
            return []
        shuffled_nodes = list(self.nodes)
        random.shuffle(shuffled_nodes)
        return shuffled_nodes
