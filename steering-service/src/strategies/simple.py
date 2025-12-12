import random
from .base import Selector

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
