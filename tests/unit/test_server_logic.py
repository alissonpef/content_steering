import pytest
from unittest.mock import MagicMock, AsyncMock
from src.steering.server import SteeringServer

@pytest.mark.asyncio
async def test_coords_update_does_not_overwrite_oracle_latency():
    """
    Testa se o envio de métricas de latência pelo cliente (Bufferbloat)
    NÃO sobrescreve a latência oficial (Oráculo) do servidor.
    """
    mock_monitor = MagicMock()
    server = SteeringServer(monitor_ref=mock_monitor)
    server.current_strategy_name = "ucb1"
    
    # Simula latências do oráculo puras e isoladas
    server.last_real_latencies = {"delivery-node-1": 15.0}
    
    # Cria uma estratégia mock
    mock_selector = MagicMock()
    mock_selector.update = MagicMock()
    server.selector_instance = mock_selector
    
    # Mock handlers
    server._handle_rl_feedback = AsyncMock(return_value="OK")
    server._build_log_base = MagicMock(return_value={})
    
    # Simulamos o request como se fosse o _record_real_latency antigo
    # O cliente envia rt=250.0 para node-1
    client_rt = 250.0
    
    class FakeRequest:
        def __init__(self):
            self.client = MagicMock(host="127.0.0.1")
            
        async def json(self):
            return {
                "time": 1,
                "lat": -23.0,
                "long": -47.0,
                "rt": client_rt,
                "server_used": "delivery-node-1",
                "decision_id": "decision-123"
            }
            
    # Chama o endpoint (manualmente via função exposta na rota)
    # Procuramos a função de coords no app
    coords_route = None
    for route in server.app.routes:
        if route.path == "/coords" and "POST" in route.methods:
            coords_route = route
            break
            
    assert coords_route is not None
    await coords_route.endpoint(FakeRequest())
    
    # Verifica se a latência do oráculo PERMANECEU intacta
    assert server.last_real_latencies["delivery-node-1"] == 15.0
    
    # Garante que o feedback RL FOI chamado com os 250ms
    server._handle_rl_feedback.assert_called_once()
    called_args, called_kwargs = server._handle_rl_feedback.call_args
    assert called_args[0] == "delivery-node-1"
    assert called_args[1] == 250.0  # feedback_latency
