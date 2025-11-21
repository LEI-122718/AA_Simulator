from abc import ABC, abstractmethod
from Observation import Observation

class Environment(ABC):

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # CORREÇÃO:
        # Antes usavam grid = [[None for ...]], invertendo indices [x][y].
        # Agora fica grid[h][w] consistente com NumPy e com Lighthouse.
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

        # NOVO:
        # Ambiente agora mantém posição dos agentes (standard para todos os ambientes).
        self.agent_positions = {}

    # NOVO:
    # Método padrão para obter posição atual do agente.
    def get_agent_position(self, agent):
        return self.agent_positions[agent]

    # NOVO:
    # Método geral para todas as subclasses: gerar observação.
    def get_observation(self, agent):
        # Isso substitui o antigo observation_for()
        return Observation(self, agent)

    @abstractmethod
    def apply_action(self, agent, act):
        """Aplicar ação no ambiente."""
        pass

    @abstractmethod
    def update(self):
        """Atualização do estado do ambiente."""
        pass
