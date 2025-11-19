from abc import ABC, abstractmethod

class Environment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_observation(self, agent):
        pass

    @abstractmethod
    def apply_action(self, agent, action):
        pass

    @abstractmethod
    def update(self):
        pass
