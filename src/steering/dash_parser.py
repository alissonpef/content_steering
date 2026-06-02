import typing


class DashParser:
    def __init__(self):
        pass

    def build(
        self,
        target: str,
        nodes: typing.Sequence[tuple[str, typing.Any]],
        uri: str,
        request,
        host_suffix: str = "",
        gateway_mode: bool = False,
        request_host: str = "localhost",
    ) -> dict[str, typing.Any]:
        message: dict[str, typing.Any] = {
            "VERSION": 1,
            "TTL": 5,
            "RELOAD-URI": f"{uri}{request.url.path}",
        }
        pathway_priority_nodes = [f"{node[0]}" for node in nodes] if nodes else []
        message["PATHWAY-PRIORITY"] = pathway_priority_nodes + ["cloud"]
        if nodes:
            message["PATHWAY-CLONES"] = self._generate_pathway_clones(
                nodes=nodes,
                host_suffix=host_suffix,
                gateway_mode=gateway_mode,
                request_host=request_host,
            )
        return message

    def _generate_pathway_clones(
        self,
        nodes: typing.Sequence[tuple[str, typing.Any]],
        host_suffix: str = "",
        gateway_mode: bool = False,
        request_host: str = "localhost",
    ) -> list:
        clones = []
        for node_info in nodes:
            node_name = node_info[0]
            if gateway_mode:
                short_name = node_name.replace("delivery-node-", "node")
                replacement_host = f"{request_host}/{short_name}"
            else:
                replacement_host = f"https://{node_name}{host_suffix}"
            clone = {
                "BASE-ID": "cloud",
                "ID": f"{node_name}",
                "URI-REPLACEMENT": {"HOST": replacement_host},
            }
            clones.append(clone)
        return clones
