class DashParser:
    def __init__(self):
        pass

    def build(self, target: str, nodes: list, uri: str, request) -> dict:
        message = {"VERSION": 1, "TTL": 5, "RELOAD-URI": f"{uri}{request.path}"}
        pathway_priority_nodes = [f"{node[0]}" for node in nodes] if nodes else []
        message["PATHWAY-PRIORITY"] = pathway_priority_nodes + ["cloud"]
        if nodes:
            message["PATHWAY-CLONES"] = self._generate_pathway_clones(nodes)
        return message

    def _generate_pathway_clones(self, nodes: list) -> list:
        clones = []
        for node_info in nodes:
            node_name = node_info[0]
            clone = {
                "BASE-ID": "cloud",
                "ID": f"{node_name}",
                "URI-REPLACEMENT": {"HOST": f"https://{node_name}"},
            }
            clones.append(clone)
        return clones
