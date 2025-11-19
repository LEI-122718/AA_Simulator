from abc import ABC, abstractmethod
from Observation import Observation

class Environment(ABC):

    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Criação da matriz/grelha que representa o estado do mundo (recursos, obstáculos, etc).
        # Inicialmente vazia (None ou 0). As subclasses preenchem-na.
        self.grid = [[None for _ in range(height)] for _ in range(width)]


    def observation_for(self, agent):
        #oq vês, mapa(positions->item)
        pass

    @abstractmethod
    def update(self):
        #chamado pelo simulation engine para atualizar o ambiente!!!
        pass

    @abstractmethod
    def apply_act(self, agent, act):
        #aplica a acão final, valida se pode ir ...
        pass