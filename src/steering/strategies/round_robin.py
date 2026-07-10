from typing import Dict, Any, List
from .base import Selector


class RoundRobin(Selector):
    def __init__(self, monitor=None):
        super().__init__(monitor=monitor)
        self.index = 0

    def select_arm(self, **kwargs) -> List[str]:
        if self.monitor:
            current_monitor_node_names = [
                name for name, _ in self.monitor.get_nodes() if name
            ]
            if not current_monitor_node_names and (not self.nodes):
                return []
            if set(current_monitor_node_names) != set(self.nodes):
                self.initialize(current_monitor_node_names)
        if not self.nodes:
            return []
        n = len(self.nodes)
        curr_idx = self.index % n
        ordered_nodes = [self.nodes[(curr_idx + i) % n] for i in range(n)]
        self.index += 1
        return ordered_nodes

    def get_state(self) -> Dict[str, Any]:
        return {"strategy": "round_robin", "current_index": self.index}
