import numpy as np
from Environment import Environment
from Observation import Observation
from Action import ActionType

class Foraging(Environment):
    def __init__(self, grid, nest_pos):
        self.grid = np.array(grid, dtype=int)
        self.nest_pos = np.array(nest_pos, dtype=int)

        self.agent_positions = {}
        self.carrying = {}

        self.height, self.width = self.grid.shape

    def add_agent(self, agent, pos):
        self.agent_positions[agent] = np.array(pos, dtype=int)
        self.carrying[agent] = False

    def get_observation(self, agent):
        x, y = self.agent_positions[agent]
        return Observation({
            "cell_value": self.grid[y, x],
            "carrying": self.carrying[agent]
        })

    def apply_action(self, agent, action):
        x, y = self.agent_positions[agent]

        if action.type == ActionType.UP:    y -= 1
        if action.type == ActionType.DOWN:  y += 1
        if action.type == ActionType.LEFT:  x -= 1
        if action.type == ActionType.RIGHT: x += 1

        x = np.clip(x, 0, self.width-1)
        y = np.clip(y, 0, self.height-1)

        self.agent_positions[agent] = np.array([x, y])

        # Coleta
        if action.type == ActionType.COLLECT and not self.carrying[agent]:
            if self.grid[y, x] > 0:
                self.grid[y, x] -= 1
                self.carrying[agent] = True

        # Depositar
        if action.type == ActionType.DROP:
            if self.carrying[agent] and np.array_equal([x, y], self.nest_pos):
                self.carrying[agent] = False

    def update(self):
        pass
