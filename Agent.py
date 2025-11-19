from abc import ABC, abstractmethod
import random

from Action import Action, ActionType
from Sensor import Sensor


class Agent:
    def __init__(self, name,radius, (x,y)):
        self.name = name
        self.sensor = Sensor(radius)
        self.last_obs = None
        self.position= (x,y)

    def observe(self, observation):
        self.last_obs = observation

    def act(self):
        perception = self.last_obs.get_map()

        # só células adjacentes ortogonais
        moves = {
            (-1,0): ActionType.LEFT,
            (1,0):  ActionType.RIGHT,
            (0,-1): ActionType.UP,
            (0,1):  ActionType.DOWN
        }

        # Filtrar posições livres
        free = [(dx,dy) for (dx,dy) in moves if perception[(dx,dy)] == 0]

        if not free:
            return Action(ActionType.STAY)

        dx, dy = random.choice(free)
        return Action(moves[(dx,dy)])


    def avaliate_CurrentState(self):
        perception=self.last_obs.get_map()



    @abstractmethod
    def communicate(self, message: str, from_agent):
        """Canal de comunicação simples."""
        pass

    def __repr__(self):
        return f"Agent({self.name})"
