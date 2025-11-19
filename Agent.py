from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.sensors = []
        self.last_observation = None

    @abstractmethod
    def observe(self, observation):
        self.last_observation = observation

    @abstractmethod
    def act(self):
        """Retorna um objeto Action"""
        pass

    def install(self, sensor):
        self.sensors.append(sensor)

    def evaluate(self, reward: float):
        """Reforço externo do ambiente (para agentes de aprendizagem)."""
        pass

    def communicate(self, message: str, from_agent):
        """Canal de comunicação simples."""
        pass

    def __repr__(self):
        return f"Agent({self.name})"
