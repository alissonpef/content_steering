from .base import Selector

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
