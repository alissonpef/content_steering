import numpy as np
from .base import Selector, selector_logger

class LinUCBSelector(Selector):
    def __init__(self, d: int, alpha: float, monitor=None, latency_oracle=None):
        super().__init__(monitor=monitor, latency_oracle=latency_oracle)
        self.d = d
        self.alpha = alpha
        
        self.A = {} 
        self.b = {} 
        
        selector_logger.info(f"LinUCBSelector inicializado com d={d} e alpha={alpha}")

    def initialize(self, arms_names: list):
        super().initialize(arms_names)
        for arm in self.nodes:
            if arm not in self.A:
                self.A[arm] = np.identity(self.d)
                self.b[arm] = np.zeros((self.d, 1))
                selector_logger.debug(f"[LinUCB] Braço '{arm}' inicializado.")

    def select_arm(self, **kwargs) -> list:
        contexts = kwargs.get("contexts")
        if not contexts:
            selector_logger.error("[LinUCB] 'contexts' não fornecidos para select_arm. Retornando lista vazia.")
            return []

        if set(contexts.keys()) != set(self.nodes):
            self.initialize(list(contexts.keys()))
        
        for arm in self.nodes:
            if arm in contexts and (arm not in self.b or np.allclose(self.b[arm], 0)):
                other_arms = [a for a in self.nodes if a != arm]
                import random
                random.shuffle(other_arms)
                selector_logger.info(f"[LinUCB] Explorando braço não testado: {arm}")
                return [arm] + other_arms
        
        ucb_scores = {}
        for arm in self.nodes:
            if arm not in contexts:
                continue

            x_a = contexts[arm].reshape(-1, 1) 

            try:
                A_inv = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                selector_logger.warning(f"Matriz A para o braço {arm} é singular. Usando identidade.")
                A_inv = np.identity(self.d)

            theta_hat = A_inv.dot(self.b[arm])
            
            predicted_reward = theta_hat.T.dot(x_a)
            confidence_bonus = self.alpha * np.sqrt(x_a.T.dot(A_inv).dot(x_a))
            
            ucb_scores[arm] = predicted_reward + confidence_bonus

        if not ucb_scores:
            return []
        return sorted(ucb_scores, key=ucb_scores.get, reverse=True)

    def update(self, chosen_arm_name: str, reward: float, **kwargs):
        context = kwargs.get("context")
        if context is None:
            selector_logger.error(f"[LinUCB] 'context' não fornecido para update do braço {chosen_arm_name}.")
            return

        if chosen_arm_name not in self.nodes:
            selector_logger.warning(f"[LinUCB] Update: Braço {chosen_arm_name} não está nos nós conhecidos. Ignorando.")
            return
            
        x_chosen = context.reshape(-1, 1) 
        
        self.A[chosen_arm_name] += x_chosen.dot(x_chosen.T)
        self.b[chosen_arm_name] += reward * x_chosen
        
        selector_logger.debug(f"[LinUCB] Modelo para o braço '{chosen_arm_name}' atualizado com recompensa {reward:.2f}.")
