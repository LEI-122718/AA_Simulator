from Agent import Agent
from Action import Action, ActionType

class SimpleAgent(Agent):
    def __init__(self, name):
        super().__init__(name)

    def observe(self, observation):
        self.last_observation = observation

    def act(self):
        """Comportamento totalmente reativo."""
        if self.last_observation is None:
            return Action(ActionType.STAY)

        direction = self.last_observation.get("direction")

        if direction == "UP":
            return Action(ActionType.UP)
        elif direction == "DOWN":
            return Action(ActionType.DOWN)
        elif direction == "LEFT":
            return Action(ActionType.LEFT)
        elif direction == "RIGHT":
            return Action(ActionType.RIGHT)

        return Action(ActionType.STAY)
